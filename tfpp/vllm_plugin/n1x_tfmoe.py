# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 hive-ram contributors. This file imports vllm-exl3, which is AGPL-3.0, and is licensed under the same terms
# (LICENSE in this folder). The rest of hive-ram is Apache-2.0.
"""vLLM general plugins for MiniMax-M3 EXL3 packs on WSL2 N1X (hive-ram). Three entry points, all off unless asked:
  register          TFMOE=1       routed experts through TensorFold 0.6.6's grouped EXL3 kernels (see prepare_tensorfold.sh)
  register_linear   TFLIN=1       dense EXL3 linears through TensorFold linear.cu
  register_fixes    always on     expert offset under pipeline parallelism; GPU free memory from CUDA, not psutil

Original docstring of the experts part:

TFMOE=1 turns it on. After vllm-exl3 builds a layer's experts, the layer gets a TensorFold expert table that reads the
same trellis tensors in place (only suh/svh are stacked, a few MiB a layer). Calls of at most TFMOE_MAX_ROWS rows run
TensorFold's grouped launch with the clamped SwiGLU-OAI epilogue; longer calls (prefill), expert-parallel layers and
gateless layers keep vllm-exl3's own path. TFMOE_CHECK=N also runs the original path on the first N eager calls and
logs the difference.
"""
import logging, os, sys

log = logging.getLogger("vllm.tfmoe")
MAXR = int(os.environ.get("TFMOE_MAX_ROWS", "16"))
CHECK = [int(os.environ.get("TFMOE_CHECK", "0"))]
CFG_GU = tuple(int(v) for v in os.environ.get("TFMOE_CFG_GU", "8,4,2,2").split(","))
CFG_D = tuple(int(v) for v in os.environ.get("TFMOE_CFG_D", "8,4,2,2").split(","))
_scratch = {}
_stats = {"tf": 0, "orig": 0}


def _codebook(inner):
    for name in ("mul1", "mcg"):
        if getattr(inner, name, None) is not None:
            return name
    return "3inst"


def _build(layer):
    import torch
    from tensorfold.cuda.exl3 import experts as X
    inners = getattr(layer, "_exl3_inners", None)
    if not inners or any(p.get("gate") is None for p in inners):
        return None
    def as16(t):
        return t.view(torch.int16) if t.dtype == torch.int32 else t
    trip = lambda m: (as16(m.trellis).contiguous(), m.suh, m.svh)
    cb = _codebook(inners[0]["gate"])
    ex = X.prepare([trip(p["gate"]) for p in inners], [trip(p["up"]) for p in inners],
                   [trip(p["down"]) for p in inners], cb)
    slots = int(getattr(layer, "top_k", 0) or getattr(getattr(layer, "moe_config", None), "experts_per_token", 0) or 4)
    key = (ex.count, ex.dims, ex.width, slots)
    if key not in _scratch:
        _scratch[key] = X.Scratch(ex, MAXR, slots, CFG_GU, CFG_D)
    layer._tfmoe = (ex, key)
    return ex


def register():
    if os.environ.get("TFMOE") != "1":
        return
    sys.path.insert(0, os.environ.get("TFMOE_ROOT", "/n1x/tfx"))
    import torch
    import vllm_exl3.exl3 as q
    from tensorfold.cuda.exl3 import experts as X
    orig_apply = q.apply_exl3_experts
    orig_pwal = q.Exl3MoEMethod.process_weights_after_loading

    def pwal(self, layer):
        r = orig_pwal(self, layer)
        try:
            ex = _build(layer)
            if ex is not None and not getattr(pwal, "said", False):
                pwal.said = True
                log.warning("tfmoe: TensorFold experts on (E=%d D=%d I=%d cb=%d k2 gu %s d %s, rows<=%d, cfg %s/%s)",
                            ex.count, ex.dims, ex.width, ex.cb, ex.k2_gu, ex.k2_d, MAXR, CFG_GU, CFG_D)
        except Exception as exc:  # keep vllm-exl3's path for this layer
            log.warning("tfmoe: layer kept on vllm-exl3 (%r)", exc)
        return r

    def apply(x, topk_ids, topk_weights, layer, *, limit=None, fused=None):
        tf = getattr(layer, "_tfmoe", None)
        rows = int(x.numel() // x.shape[-1])
        if tf is None or rows > MAXR or getattr(layer, "expert_map", None) is not None:
            _stats["orig"] += 1
            return orig_apply(x, topk_ids, topk_weights, layer, limit=limit, fused=fused)
        ex, key = tf
        x2d = x.reshape(rows, x.shape[-1]).contiguous()
        ids = topk_ids.reshape(rows, -1).to(torch.int32)
        # padded rows (CUDA graph capture sizes) carry id -1: TensorFold skips picks >= E, so map them there instead
        # of letting the kernel index its pointer tables at -1
        ids = torch.where(ids < 0, torch.full_like(ids, ex.count), ids).contiguous()
        w = topk_weights.reshape(rows, -1).to(torch.float32).contiguous()
        out = X.routed(x2d, ids, w, ex, _scratch[key], None, rows,
                       limit=float(limit) if limit is not None else 0.0, act_mode=X.ACT_OAI).to(x.dtype)
        _stats["tf"] += 1
        if CHECK[0] > 0 and not torch.cuda.is_current_stream_capturing():
            CHECK[0] -= 1
            ref = orig_apply(x, topk_ids, topk_weights, layer, limit=limit, fused=fused).reshape(rows, -1).float()
            o = out.reshape(rows, -1).float()
            cos = torch.nn.functional.cosine_similarity(o, ref, dim=-1).min().item()
            rel = ((o - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
            log.warning("tfmoe check: rows=%d min cos %.6f max rel %.2e | ids %d..%d w sum %.3f |x| %.3g |ref| %.3g |tf| %.3g",
                        rows, cos, rel, int(ids.min()), int(ids.max()), float(w.sum(-1).max()),
                        float(x2d.float().abs().max()), float(ref.abs().max()), float(o.abs().max()))
        return out.reshape(x.shape[:-1] + (ex.dims,))

    q.apply_exl3_experts = apply
    q.Exl3MoEMethod.process_weights_after_loading = pwal


# ---- EXL3 linear layers (TFLIN=1) ---------------------------------------------------------------------------------
# On a WSL2 N1X (8 GiB dedicated GPU memory, the rest shared) ExLlamaV3's small-batch GEMV slows ~9x once the weights
# live outside the dedicated segment (lm_head 3.5 -> 30 ms), while TensorFold's linear.cu reads the same trellis at
# full speed. TFLIN=1 routes vllm-exl3's dense EXL3 forward through linear.cu on the trellis in place ("stored"
# layout, no copy) for 1..128 rows; anything else keeps ExLlamaV3. TFLIN_CHECK=N compares the first N eager calls.

LIN_CHECK = [int(os.environ.get("TFLIN_CHECK", "0"))]
_lin = {}
# TFLIN_SPLITS: JSON file {"K,N,R": [k_splits, warps], ...} from tfx_lintune.py; rows not listed use the
# shape's own default. A split changes the summation order (fp32 throughout), not the arithmetic.
_SPLITS = {}
if os.environ.get("TFLIN_SPLITS"):
    try:
        import json as _json
        _SPLITS = {tuple(int(v) for v in k.split(",")): tuple(v) for k, v in
                   _json.load(open(os.environ["TFLIN_SPLITS"])).items()}
    except Exception as _exc:
        log.warning("tflin: TFLIN_SPLITS not read (%r)", _exc)


def _tf_linear(lin):
    import torch
    from tensorfold.cuda.exl3.linear import Exl3Linear, k2_of
    t = getattr(lin, "trellis", None)
    if t is None or t.dim() != 3 or getattr(lin, "bias", None) is not None:
        return None
    cb = "mul1" if getattr(lin, "mul1", None) is not None else ("mcg" if getattr(lin, "mcg", None) is not None else "3inst")
    k, n = 16 * t.shape[0], 16 * t.shape[1]
    if k % 128 or n % 128:
        return None
    bits = t.shape[-1] / 16
    words = (t if t.dtype == torch.int16 else t.view(torch.int16)).contiguous().view(torch.int32)
    return Exl3Linear(words, lin.suh.reshape(-1).half().contiguous(), lin.svh.reshape(-1).half().contiguous(), None,
                      bits, cb, k, n, "stored")


def register_linear():
    if os.environ.get("TFLIN") != "1":
        return
    sys.path.insert(0, os.environ.get("TFMOE_ROOT", "/n1x/tfx"))
    import torch
    import vllm_exl3.exl3 as q
    orig_dense = q._dense_forward
    orig_pwal = q.Exl3LinearMethod.process_weights_after_loading

    def pwal(self, layer):
        r = orig_pwal(self, layer)
        n = 0
        for lin in getattr(layer, "_exl3_linears", None) or []:
            if lin is None or id(lin) in _lin:
                continue
            try:
                tf = _tf_linear(lin)
            except Exception as exc:
                tf = None
                log.warning("tflin: %s kept on ExLlamaV3 (%r)", getattr(layer, "prefix", "?"), exc)
            if tf is not None:
                _ = tf.counters
                _lin[id(lin)] = (lin, tf)
                n += 1
        if n and not getattr(pwal, "said", False):
            pwal.said = True
            log.warning("tflin: TensorFold linear.cu on for EXL3 linears (first: %s)", getattr(layer, "prefix", "?"))
        return r

    def dense(linear, x_fp16):
        ent = _lin.get(id(linear))
        rows = int(x_fp16.shape[0])
        if ent is None or rows > 128:
            return orig_dense(linear, x_fp16)
        tf = ent[1]
        sp = _SPLITS.get((tf.k, tf.n, rows))
        if sp is not None:
            dflt, tf.split = tf.split, sp
            try:
                out = tf(x_fp16, out_dtype=torch.float32)
            finally:
                tf.split = dflt
        else:
            out = tf(x_fp16, out_dtype=torch.float32)
        if LIN_CHECK[0] > 0 and not torch.cuda.is_current_stream_capturing():
            LIN_CHECK[0] -= 1
            ref = orig_dense(linear, x_fp16).float()
            cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1).min().item()
            log.warning("tflin check: rows=%d K=%d N=%d min cos %.6f", rows, ent[1].k, ent[1].n, cos)
        return out

    q._dense_forward = dense
    q.Exl3LinearMethod.process_weights_after_loading = pwal


# ---- fixes for vllm-exl3 under pipeline parallelism (always on) -----------------------------------------------------
def register_fixes():
    """vllm-exl3 41fdb7d sets a routed layer's checkpoint expert offset from get_ep_group().rank, the GLOBAL rank: on
    pipeline stage 1 (global rank 1, EP size 1) it looks up experts 128..255, the header prescan finds none, and every
    expert is staged in host memory before packing -- twice the stage's weights at once, which a 64 GB node cannot
    hold. The offset is the rank inside the EP group times the local expert count."""
    import vllm_exl3.exl3 as q
    orig = q.Exl3MoEMethod.create_weights

    def create_weights(self, layer, num_experts, *a, **k):
        try:
            from vllm.distributed.parallel_state import get_ep_group
            ep = get_ep_group()
            fixed = int(ep.rank_in_group) * int(num_experts)
            if getattr(layer, "starting_expert_offset", fixed) != fixed or ep.rank != ep.rank_in_group:
                log.warning("n1xfix: %s expert offset %s -> %d (EP rank %d of %d, global rank %d)",
                            getattr(layer, "layer_name", "?"), getattr(layer, "starting_expert_offset", None), fixed,
                            ep.rank_in_group, ep.world_size, ep.rank)
            layer.starting_expert_offset = fixed
        except Exception as exc:
            log.warning("n1xfix: expert offset left as is (%r)", exc)
        return orig(self, layer, num_experts, *a, **k)

    q.Exl3MoEMethod.create_weights = create_weights

    # vLLM's MemorySnapshot replaces cudaMemGetInfo's free memory with psutil's available RAM on integrated GPUs.
    # On a WSL2 N1X the weights live in the GPU's dedicated segment, which the WSL VM's memory cap says nothing
    # about: with .wslconfig memory=12GB that rule reported ~8 GiB "free" of 52.6 and vLLM refused to start. Use the
    # CUDA figure (N1X_UMA_FREE=psutil keeps the stock behaviour).
    if os.environ.get("N1X_UMA_FREE", "cuda") == "cuda":
        try:
            from vllm.platforms import current_platform
            type(current_platform).is_integrated_gpu = classmethod(lambda cls, device_id=0: False)
            log.warning("n1xfix: GPU free memory taken from cudaMemGetInfo, not psutil (WSL cap no longer limits it)")
        except Exception as exc:
            log.warning("n1xfix: could not patch is_integrated_gpu (%r)", exc)
