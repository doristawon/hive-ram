"""A stage's share of the fork's model (spec 3.3, 6.3, 7.7): Model and Engram on a layer range.

``StageModel`` is the fork's Model whose ``w.layers`` holds the stage's layers (load.StageLayers: global numbers):
- ``new_cache``: rings for its own layers (zero-row tensors for the others, so ``sc.ring[lay.idx]`` keeps global
  numbers), the planes it reads (its kv sources and the replicas of earlier ones), indexer keys where something on the
  stage scores them, the compressor inputs of its own ratio>1 sources (F/model.py:744-769);
- ``forward``: F/model.py:1241-1293's KERNELS path with the stage's entry (the embedding on stage 0, the previous
  stage's ACT elsewhere: spec 6.3) and exit (M2's state to the next stage), every stage setting ``sc.host`` (the
  decode rounds always set it, F/multi.py:1074, and HostIds.set asserts start <= len: FO/ops.py:351);
- ``_forward_k``: the fork's, recompiled with M1 (floor from the stage's replay floor) and M2 (a stage without the
  head returns its exit state, the last post applied) -- and F1c / F1d with the carry fallback.

``StageEngram`` serves the Engram layers the stage owns and is a no-op (prefetch, touch) or an error (rows) for any
other, so the fork's own callers run unchanged (F/engine.py:404-411; F/rounds.py:688-697, 602-605, 652-654).
"""

from __future__ import annotations

from pathlib import Path

import torch

from tensorfold.families.deepseek_v41.cuda.model import KV_QUANT, RAW, RING_EXTRA, Engram, Model, SeqCache
from tensorfold.families.deepseek_v41.ops import BF16, F32

from . import env, patch, plan as PL

CARRY = env.cut_post() == "carry"


class StageModel(Model):
    _forward_k = patch.patched(Model._forward_k, patch.group_ids("forward_k", CARRY), "Model._forward_k")

    def __init__(self, w, comm, engram=None, *, plan, io) -> None:
        super().__init__(w, comm, engram)                    # F/model.py:727-741 (its layer loop: the own layers)
        self.plan, self.io = plan, io
        self._tfpp_last = plan.last
        self._tfpp_floor = 0
        self._tfpp_carry_in = None                           # F1: (pending, post, comb) of the stage before
        self._tfpp_carry_out = None
        self.f1 = bool(io.f1_chunk)

    @property
    def _tfpp_local(self) -> bool:
        return self.io.local

    # -- caches ---------------------------------------------------------------------------------------------------
    def new_cache(self, cap: int) -> SeqCache:
        c, P = self.cfg, self.plan
        ring = c.window + RING_EXTRA
        sc = SeqCache(cap=cap, ring_size=ring)
        sc.ring = [torch.zeros((ring if P.lo <= i < P.hi else 0, c.head_dim), dtype=BF16, device="cuda")
                   for i in range(c.n_layers)]
        for i in P.planes:
            r = c.compress_ratios[i]
            rows = cap // r + 2                              # + a scratch row (F/model.py:753)
            if KV_QUANT:
                sc.comp[i] = (torch.zeros((rows, c.head_dim // 2), dtype=torch.uint8, device="cuda"),
                              torch.zeros((rows, c.head_dim // 16), dtype=torch.uint8, device="cuda"))
            else:
                sc.comp[i] = torch.zeros((rows, c.head_dim), dtype=BF16, device="cuda")
            if i in P.index_planes:
                if KV_QUANT:
                    sc.index_k[i] = (torch.zeros((rows, c.idx_dim // 2), dtype=torch.uint8, device="cuda"),
                                     torch.full((rows, c.idx_dim // 32), 127, dtype=torch.uint8, device="cuda"))
                else:
                    sc.index_k[i] = torch.zeros((rows, c.idx_dim), dtype=BF16, device="cuda")
            if i in P.own_raw:
                sc.comp_raw[i] = (torch.zeros((RAW, c.head_dim), dtype=F32, device="cuda"),
                                  torch.zeros((RAW, c.head_dim), dtype=F32, device="cuda"))
        sc.tokens = torch.zeros((cap,), dtype=torch.int64, device="cuda")
        return sc

    # -- one prompt chunk ----------------------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, sc, ids: torch.Tensor, start: int, all_logits: bool = False, taps: list | None = None,
                host_ids: list | None = None, replay: int | None = None, image: tuple | None = None):
        P, c, w = self.plan, self.cfg, self.w
        if image is not None and image[0].numel():
            raise ValueError("tfpp-ds41 v1 serves no images")
        n0, start0 = ids.shape[0], start
        assert start == sc.length, (start, sc.length)                        # F/model.py:1252
        sc.tokens[start:start + n0] = ids                                    # 1253
        sc.host.set(start, ids.tolist() if host_ids is None else host_ids)   # every stage (spec 2.2)
        hashes = self.engram.hashes(sc.host.view(), start, n0) if self.engram is not None else None
        kind, s1, n1, self._tfpp_floor = PL.chunk_in(P, start0, n0, replay)
        if P.first:
            emb = w.embed[ids].to(BF16)                                      # F/model.py:1254-1261
            h = emb[:, None, :].expand(-1, c.hc, -1).contiguous()
            pre = torch.zeros((n0, c.hc), dtype=F32, device="cuda")
            pre[:, 0] = 1.0
            shared: dict = {}
        else:
            if kind == "kv":                                                 # encoder-only past the replay layer
                self.io.recv_kv(sc, start0, n0)
                if not P.last:
                    self.io.send_kv(sc, start0, n0)
                sc.length = start0 + n0
                return None
            h, pre, shared, self._tfpp_carry_in = self.io.recv_chunk(sc, start0, n0, s1, n1, self._tfpp_floor)
            if n1 != n0:                                                     # the replay's rows only (1347-1349)
                ids, start = ids[n0 - n1:], s1
        try:
            out = self._forward_k(sc, ids, start, all_logits, taps, h, pre, hashes, shared, replay, None)
        finally:
            self._tfpp_carry_in = None
        if out is None:                                  # this stage's replay layer made the chunk encoder-only (1344)
            if not P.last:
                self.io.send_kv(sc, start0, n0)
            return None
        if isinstance(out, tuple):                       # M2: the stage's exit
            _, h, pre, shared, floor, s2, n2 = out
            carry, self._tfpp_carry_out = self._tfpp_carry_out, None
            self.io.send_chunk(sc, start0, n0, s2, n2, floor, h, pre, shared, carry)
            return None
        return out                                       # the leader: the head's logits (1420-1427)

    # -- F1 (spec 3.6; the anchors F1c / F1d call these) --------------------------------------------------------
    def _tfpp_pending_in_k(self, post: torch.Tensor, comb: torch.Tensor):
        carry = self._tfpp_carry_in
        if carry is None:
            return None
        pend, p, cm = carry
        post.copy_(p)
        comb.copy_(cm)
        return pend

    def _tfpp_keep_pending_k(self, pending, post, comb) -> bool:
        if self.f1 and not self._tfpp_last:
            self._tfpp_carry_out = (pending, post, comb)
            return True
        return False


class StageEngram(Engram):
    """The fork's Engram (F/model.py:489-723) over the tables of the layers this stage owns."""

    def __init__(self, engram_dir: str, cfg, token_map: list[int], owned) -> None:
        super().__init__(engram_dir, cfg, token_map, 0, 1)             # world=1: all 24 hash columns (506-507)
        self.owned = tuple(int(i) for i in owned)
        want = {f"layers.{i}.engram.embed.{k}" for i in self.owned for k in ("weight", "scale")}
        if set(self.files) != want:
            raise RuntimeError(f"Engram tables in {engram_dir}: {sorted(self.files)}; this stage owns layers "
                               f"{list(self.owned)} and needs exactly {sorted(want)} (the fork globs every "
                               "*.safetensors there, F/model.py:517: give each owner a folder of its own tables)")

    def prefetch(self, layer: int, idx, lane: int = 1) -> None:
        if layer not in self.owned:
            return
        return super().prefetch(layer, idx, lane)

    def touch(self, layer: int, idx) -> None:
        if layer not in self.owned:
            return
        return super().touch(layer, idx)

    def rows(self, layer: int, idx) -> torch.Tensor:
        if layer not in self.owned:
            raise RuntimeError(f"tfpp-ds41: Engram rows of layer {layer} asked of a stage owning {list(self.owned)}")
        return super().rows(layer, idx)


def engram_files(engram_dir) -> list[Path]:
    return sorted(Path(engram_dir).glob("*.safetensors"))
