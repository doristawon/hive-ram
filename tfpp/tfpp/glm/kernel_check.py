"""T-K (spec 13.4): world=1's kernel shapes against the two TP=2 halves, layer by layer (one GPU, layers 0..3).

world=1 runs 64 MLA / KDA heads, 2,048-wide experts and the whole vocabulary on one rank, shapes MIA never ran
(G/latent.py:703's tile comment: "a rank's 32"). Loading the same layers three times (world=1, and world=2 ranks 0
and 1), feeding all three the same normed rows and comparing what each block writes shows whether every head of
world=1 gives its TP=2 half's bits. Prefill path: KDA outputs and states, DSA head outputs, latent rows, the
indexer's inputs, the MoE routing and expert activations must be equal bit for bit. Decode R=8: the expert
activations bit for bit, attention per head within a tolerance (q_b's split-K differs by shape: 1 vs 4 splits).
K2: TF_GLM_EXL3_DEC=1 against 0 at a TP=2 half's expert width (1,024). K4: first-call (JIT) times, BUILT paths.

  python -m tfpp.glm.kernel_check --model DIR [--layers 0:4] [--rows 2048,952] [--decode-rows 8] [--out K.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _ints(t):
    import torch

    return t.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def cmp(a, b, *, name: str, need_bits: bool, report: list, **where) -> bool:
    """Record one comparison: bitwise equality, and max relative error / cosine for the tolerance items."""

    import torch

    rec = dict(where, item=name, shape=list(a.shape))
    if a.shape != b.shape or a.dtype != b.dtype:
        rec.update(ok=False, why=f"shapes {list(a.shape)} {a.dtype} vs {list(b.shape)} {b.dtype}")
        report.append(rec)
        return False
    same = bool(torch.equal(_ints(a), _ints(b)))
    x, y = a.float().reshape(-1), b.float().reshape(-1)
    rel = float((x - y).abs().max() / y.abs().max().clamp_min(1e-30)) if x.numel() else 0.0
    cos = float(torch.nn.functional.cosine_similarity(x, y, dim=0)) if x.numel() and y.abs().max() > 0 else 1.0
    ok = same if need_bits else (rel <= 2 ** -7 and cos >= 0.9999)
    rec.update(bitwise=same, rel=rel, cos=cos, need_bits=need_bits, ok=ok)
    report.append(rec)
    return ok


def halves(t, dim: int):
    n = t.shape[dim]
    return t.narrow(dim, 0, n // 2), t.narrow(dim, n // 2, n - n // 2)


def run(args) -> dict:
    import numpy as np
    import torch

    from tensorfold.families.glm5_next.cuda import decode as D, forward as fw, glue
    from tensorfold.families.glm5_next.cuda import engine as E

    from . import startup
    from .load import load_range

    a, b = args.layers
    torch.cuda.set_device(0)
    t0 = time.perf_counter()
    w1 = load_range(args.model, a, b, is_first=True, is_last=False, mtp=False, drafter=False)
    w2 = [load_range(args.model, a, b, is_first=True, is_last=False, mtp=False, drafter=False, world=2, rank=r)
          for r in (0, 1)]
    load_s = time.perf_counter() - t0
    cap = max(E.DENSE_CAPACITY, sum(args.rows) + args.decode_rows + E.MAX_ROWS)
    t0 = time.perf_counter()
    engines = [D.Engine(w, capacity=cap, max_rows=E.MAX_ROWS, prefill_rows=max(args.rows), graphs=False,
                        long_context=True) for w in (w1, *w2)]
    build_s = time.perf_counter() - t0
    c = w1.cfg
    rng = np.random.default_rng(7)
    report: list = []
    timing: dict = {"load_s": round(load_s, 1), "build_s": round(build_s, 1)}

    def window(ids, prefill: bool, phase: str) -> None:
        R = len(ids)
        bufs = []
        for e in engines:
            bb = e.pbuf if prefill else e.buf
            fw.stage(e.w, e.st, bb, ids)
            bufs.append(bb)
        e1, b1 = engines[0], bufs[0]
        st1 = e1.st
        glue.embed(b1.ids[:R], w1.embed, c.hidden, c.streams, b1.x[:R])
        nch, host = fw.chunks_for(st1, R), st1.pos
        x = b1.x[:R]
        for li, layers in enumerate(zip(w1.layers, w2[0].layers, w2[1].layers)):
            l1 = layers[0]
            at = dict(phase=phase, layer=l1.index, kind=l1.kind)

            def share_normed():
                for bb in bufs[1:]:
                    bb.normed[:R].copy_(b1.normed[:R])
                    bb.xs[:R].copy_(b1.xs[:R])

            h = l1.attn_hc
            glue.hc_pre(x, h.fn, h.base, h.scale, l1.in_norm, b1.normed[:R], b1.xs[:R], b1.post[:R], b1.comb[:R],
                        b1.hcpart[:R], c.eps, c.hc_eps, c.hc_iters, prompt=prefill)
            share_normed()
            t = time.perf_counter()
            g = [fw._mixer(l, e.w, e.st, bb, R, nch, host, None) for l, e, bb in zip(layers, engines, bufs)]
            torch.cuda.synchronize()
            timing.setdefault(f"{phase}:mixer:{l1.index}", round((time.perf_counter() - t) * 1e3, 2))
            if l1.kind == "kda":
                k = engines[0].st.kda_index[l1.index]
                if prefill:
                    outs = [bb.kscratch.out[:R] for bb in bufs]
                else:
                    outs = [e.st.scratch[k].out[:R] for e in engines]
                lo, hi = halves(outs[0].view(R, -1, 128), 1)
                cmp(lo, outs[1].view(R, -1, 128), name="kda out heads 0-31", need_bits=prefill, report=report, **at)
                cmp(hi, outs[2].view(R, -1, 128), name="kda out heads 32-63", need_bits=prefill, report=report, **at)
                if prefill:                                     # the layer committed its state (kda_rows)
                    recs = [e.st.rec[e.st.cur[k], k] for e in engines]
                    lo, hi = halves(recs[0], 0)
                    cmp(lo, recs[1], name="kda state heads 0-31", need_bits=True, report=report, **at)
                    cmp(hi, recs[2], name="kda state heads 32-63", need_bits=True, report=report, **at)
            else:
                di = engines[0].st.dsa_index[l1.index]
                lo, hi = halves(bufs[0].vn[:R], 1)
                cmp(lo, bufs[1].vn[:R], name="dsa out heads 0-31", need_bits=prefill, report=report, **at)
                cmp(hi, bufs[2].vn[:R], name="dsa out heads 32-63", need_bits=prefill, report=report, **at)
                lat = [e.st.kc[di][host:host + R] for e in engines]
                cmp(lat[0], lat[1], name="latent rows (rank 0)", need_bits=True, report=report, **at)
                cmp(lat[0], lat[2], name="latent rows (rank 1)", need_bits=True, report=report, **at)
                if host + R - 1 >= c.dense_limit:   # b.qi is written only for sparse rows (G/forward.py:686, 708)
                    cmp(bufs[0].qi[:R], bufs[1].qi[:R], name="indexer queries", need_bits=True, report=report, **at)
                cmp(bufs[0].ikr[:R], bufs[1].ikr[:R], name="indexer keys", need_bits=True, report=report, **at)
            if g[0] is not None and g[1] is not None and g[2] is not None:
                cmp(g[0][0], g[1][0] + g[2][0], name="mixer partial (sum of halves)", need_bits=False,
                    report=report, **at)
            glue.hc_post(x, x, g[0], b1.post[:R], b1.comb[:R])
            h = l1.ffn_hc
            glue.hc_pre(x, h.fn, h.base, h.scale, l1.post_norm, b1.normed[:R], b1.xs[:R], b1.post[:R], b1.comb[:R],
                        b1.hcpart[:R], c.eps, c.hc_eps, c.hc_iters, prompt=prefill)
            share_normed()
            t = time.perf_counter()
            f = [fw._ffn(l, e.w, bb, R) for l, e, bb in zip(layers, engines, bufs)]
            torch.cuda.synchronize()
            timing.setdefault(f"{phase}:ffn:{l1.index}", round((time.perf_counter() - t) * 1e3, 2))
            if l1.moe is not None:
                cmp(bufs[0].pick[:R], bufs[1].pick[:R], name="moe routing", need_bits=True, report=report, **at)
                slots = c.top_k + 1
                if bufs[0].exl3 is not None:
                    acts = [bb.exl3.xd[:R * slots] for bb in bufs]
                    name = "moe expert activations (exl3 xd)"
                else:
                    acts = [bb.eact[:R * slots] for bb in bufs]
                    name = "moe expert activations (eact)"
                lo, hi = halves(acts[0], 1)
                cmp(lo, acts[1], name=name + " cols 0-1023", need_bits=True, report=report, **at)
                cmp(hi, acts[2], name=name + " cols 1024-2047", need_bits=True, report=report, **at)
            if f[0] is not None and f[1] is not None and f[2] is not None:
                cmp(f[0][0], f[1][0] + f[2][0], name="ffn partial (sum of halves)", need_bits=False, report=report,
                    **at)
            glue.hc_post(x, x, f[0], b1.post[:R], b1.comb[:R])
            for bb in bufs[1:]:                 # every layer starts from world=1's streams
                bb.x[:R].copy_(x)
        for e, bb in zip(engines, bufs):
            fw.commit(e.w, e.st, bb, R, R)

    vocab = c.vocab
    for i, rows in enumerate(args.rows):
        window([int(t) for t in rng.integers(0, vocab, rows)], True, f"prefill{i}")
    window([int(t) for t in rng.integers(0, vocab, args.decode_rows)], False, "decode")

    # K2: the decode kernel against the grouped one at a TP=2 half's width (1,024)
    k2 = None
    moe = next((l for l in w2[0].layers if l.moe is not None), None)
    if moe is not None:
        e = engines[1]
        bb = e.buf
        R = args.decode_rows
        ys = []
        for flag in ("1", "0"):
            os.environ["TF_GLM_EXL3_DEC"] = flag
            fw._ffn(moe, e.w, bb, R)
            torch.cuda.synchronize()
            ys.append(bb.ey[:R].clone())
        os.environ["TF_GLM_EXL3_DEC"] = "1"
        k2 = cmp(ys[0], ys[1], name="K2 exl3 dec vs grouped (NI=1024)", need_bits=True, report=report,
                 phase="K2", layer=moe.index, kind="moe")
    built = [startup.built_info(args.layers, e) for e in engines]
    required = [r for r in report if r.get("need_bits") or not r.get("bitwise", True)]
    failed = [r for r in report if not r.get("ok")]
    upper = [r for r in failed if "32-63" in r["item"] or "1024-2047" in r["item"] or "rank 1" in r["item"]]
    lower = [r for r in failed if r not in upper]
    verdict = "pass" if not failed else ("heads 32-63 only: an index error in world=1's 64-head paths" if upper and
                                         not lower else "fail")
    return {"verdict": verdict, "failed": failed, "checked": len(report), "required": len(required), "k2": k2,
            "timing": timing, "built": built, "report": report}


def main(argv: list[str] | None = None) -> int:
    from .proto import parse_layers

    ap = argparse.ArgumentParser(prog="python -m tfpp.glm.kernel_check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=parse_layers, default=(0, 4))
    ap.add_argument("--rows", default="2048,952", help="prompt chunks (their sum passes the dense limit)")
    ap.add_argument("--decode-rows", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    args.rows = [int(x) for x in args.rows.split(",") if x.strip()]
    from . import env, pins

    env.check_env("kernel-check")
    pins.check_tree()
    env.check_device()
    res = run(args)
    for r in res["failed"]:
        print(f"FAIL {r.get('phase')} layer {r.get('layer')} {r['item']}: bitwise {r.get('bitwise')} "
              f"rel {r.get('rel', 0):.3g} cos {r.get('cos', 0):.6f} {r.get('why', '')}")
    print(f"T-K: {res['verdict']} ({res['checked']} comparisons, {len(res['failed'])} failed; K2 {res['k2']})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
    return 0 if res["verdict"] == "pass" else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
