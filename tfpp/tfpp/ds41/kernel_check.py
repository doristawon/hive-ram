"""Kernel checks at world=1 (spec 15.6 T-K', T-K). One GPU, no chain.

T-K' (``--hc``): the cut's one change of kernels, bit for bit. In one process the last sublayer's post before a cut
is fused into the next layer's mixes (decode: hc_pre2(gathered=pending), F/rounds.py:506-507; prompt chunks:
hc_pre_pf(gathered), F/model.py:1376-1378); in a chain the stage before the cut posts it alone (hc_post: F/rounds.py:
549-550, F/model.py:1416-1418) and the stage after mixes un-posted streams. The fork checks the two equal at world=2
only (gathered [2, R, D]: T/small_kernels_test.py:84, T/hc_prefill2_test.py:75); here at world=1 ([1, R, D]):
decode R = 1..16 (plain, with the Sinkhorn on a side stream, with and without "hc_dots"), prompt chunks R in
{2048, 1919, 128, 33, 17, 5}. Only buffers the kernels write are compared (lesson 3). A failure means F1
(TFPP_DS41_CUT_POST=carry) on every stage, then every test again.

T-K (``--layer i``, diagnostic): world=1's full-width kernels against the two TP=2 halves summed in rank order, for
layer i's MoE and its attention output projections (wo_a + wo_b), at a decode window (6 rows) and a prompt chunk
(257 rows): cos >= 0.9999 and max|d| <= 2^-6 max|y|.

  python -m tfpp.ds41.kernel_check --hc [--model DIR] [--layers 2,5,7] [--out tk.json]
  python -m tfpp.ds41.kernel_check --model DIR --layer 5 [--experts K] [--out tk.json]
"""

from __future__ import annotations

import argparse
import json
import sys


def beq(a, b) -> bool:
    """Bitwise equality (T/small_kernels_test.py:18-26)."""

    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))
    return torch.equal(a, b)


def hc_params(args, D: int):
    """[(label, fn, scale, base, norm)]: the checkpoint's mixes (--model) or random ones (T/hc_prefill2_test.py:18-30)."""

    import torch

    out = []
    if args.model:
        from .load import AliasShards, namer_for

        sh = AliasShards(args.model, namer_for(args.model))
        try:
            for i in [int(x) for x in args.layers.split(",") if x.strip()]:
                for which, norm in (("attn", "attn_norm"), ("ffn", "ffn_norm")):
                    p = f"layers.{i}"
                    out.append((f"L{i}.{which}", sh.get(f"{p}.hc_{which}_fn").float().cuda(),
                                sh.get(f"{p}.hc_{which}_scale").float().cuda(),
                                sh.get(f"{p}.hc_{which}_base").float().cuda(), sh.get(f"{p}.{norm}.weight").cuda()))
        finally:
            sh.close()
        return out
    gen = torch.Generator().manual_seed(7)
    for k in range(2):
        out.append((f"random{k}", (torch.randn((24, 4 * D), generator=gen) * 0.02).cuda(),
                    (torch.rand((3,), generator=gen) + 0.5).cuda(), (torch.randn((24,), generator=gen) * 0.5).cuda(),
                    (torch.rand((D,), generator=gen) + 0.5).to(torch.bfloat16).cuda()))
    return out


def run_hc(args) -> dict:
    import torch

    from tensorfold.families.deepseek_v41.cuda import kernels as K

    D, eps, hc_eps, iters = 5120, 1e-6, 1e-6, 20
    if args.model:
        from tensorfold.families.deepseek_v41.config import Cfg

        cfg = Cfg.read(args.model)
        D, eps, hc_eps, iters = cfg.dim, cfg.eps, cfg.hc_eps, cfg.hc_iters
    F32, BF16 = torch.float32, torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(5)
    res: dict = {}

    def record(name: str, ok: bool) -> None:
        r = res.setdefault(name, [0, 0])
        r[0] += 1
        r[1] += int(bool(ok))

    def rnd(*shape, scale=1.0, dtype=BF16):
        return (torch.randn(shape, generator=g, device="cuda") * scale).to(dtype)

    sink_ok = K.hc_defer_on(D)
    dots_saved = K.SMALL_SWITCHES.get("hc_dots")
    for label, fn, sc, bs, norm in hc_params(args, D):
        # -- decode windows: hc_post + hc_pre2 (the chain) against hc_pre2(gathered) (one process) -------------
        variants = [("plain", None, None)]
        if sink_ok:
            variants += [("sink+dots", torch.cuda.Stream(), True), ("sink-dots", torch.cuda.Stream(), False)]
        for R in range(1, 17):
            for scale in (1e-3, 1.0, 30.0):
                h = rnd(R, 4, D, scale=scale)
                pre_in = torch.rand(R, 4, generator=g, device="cuda") + 0.1
                gathered = torch.randn(1, R, D, generator=g, device="cuda") * scale        # world=1: [1, R, D]
                x0 = torch.empty(R, D, dtype=BF16, device="cuda")
                part = torch.empty(R * K.HC_BLOCKS * 32, dtype=F32, device="cuda")
                po, post, comb = (torch.empty(R, 4, device="cuda"), torch.empty(R, 4, device="cuda"),
                                  torch.empty(R, 4, 4, device="cuda"))
                K.hc_pre(h, fn, sc, bs, pre_in, norm, eps, hc_eps, iters, x0, po, post, comb, part)  # real mixes
                for name, sink, dots in variants:
                    if dots is not None:
                        K.set_switch("hc_dots", dots)
                    # the chain: the stage before posts alone, the stage after mixes the posted streams
                    hc_, p1, c1 = h.clone(), post.clone(), comb.clone()
                    K.hc_post(gathered, hc_, p1, c1, hc_)
                    x1 = torch.empty(R, D, dtype=BF16, device="cuda")
                    pre1 = torch.empty(R, 4, device="cuda")
                    part1 = torch.empty_like(part)
                    s1 = K.hc_pre2(hc_, fn, sc, bs, pre_in, norm, eps, hc_eps, iters, x1, pre1, p1, c1, part1, sink=sink)
                    if sink is not None:
                        torch.cuda.current_stream().wait_stream(sink)
                    # one process: the post fused into the mixes
                    p2, c2 = post.clone(), comb.clone()
                    x2 = torch.empty(R, D, dtype=BF16, device="cuda")
                    pre2 = torch.empty(R, 4, device="cuda")
                    part2 = torch.empty_like(part)
                    hout = torch.empty_like(h)
                    s2 = K.hc_pre2(h, fn, sc, bs, pre_in, norm, eps, hc_eps, iters, x2, pre2, p2, c2, part2,
                                   gathered=gathered, h_out=hout, sink=sink)
                    if sink is not None:
                        torch.cuda.current_stream().wait_stream(sink)
                    torch.cuda.synchronize()
                    record(f"decode/{name}/streams", beq(s1, s2))
                    record(f"decode/{name}/x", beq(x1, x2))
                    record(f"decode/{name}/pre_post_comb", beq(pre1, pre2) and beq(p1, p2) and beq(c1, c2))
        if dots_saved is not None:
            K.set_switch("hc_dots", dots_saved)
        # -- prompt chunks: hc_post + hc_pre (the chain) against hc_pre_pf(gathered) (one process) --------------
        gen = torch.Generator().manual_seed(11)
        for R in (2048, 1919, 128, 33, 17, 5):
            h = (torch.randn((R, 4, D), generator=gen) * 0.7).to(BF16).cuda()
            pre_in = (torch.rand((R, 4), generator=gen) * 0.9 + 0.05).cuda()
            gathered = (torch.randn((1, R, D), generator=gen) * 0.3).cuda()
            post0 = (torch.rand((R, 4), generator=gen) * 2.0).cuda()
            comb0 = torch.rand((R, 4, 4), generator=gen)
            comb0 = (comb0 / comb0.sum(1, keepdim=True)).cuda()
            h1, p1, c1 = h.clone(), post0.clone(), comb0.clone()
            x1, pre1 = torch.empty((R, D), dtype=BF16, device="cuda"), torch.empty((R, 4), device="cuda")
            part1 = torch.empty((R * K.HC_BLOCKS * 32,), device="cuda")
            K.hc_post(gathered, h1, p1, c1, h1)
            K.hc_pre(h1, fn, sc, bs, pre_in, norm, eps, hc_eps, iters, x1, pre1, p1, c1, part1)
            h2, p2, c2 = h.clone(), post0.clone(), comb0.clone()
            x2, pre2 = torch.empty((R, D), dtype=BF16, device="cuda"), torch.empty((R, 4), device="cuda")
            part2 = torch.empty((R * K.HC_BLOCKS * 32,), device="cuda")
            hout = torch.empty_like(h2)
            src = K.hc_pre_pf(h2, fn, sc, bs, pre_in, norm, eps, hc_eps, iters, x2, pre2, p2, c2, part2,
                              gathered=gathered, h_out=hout)
            torch.cuda.synchronize()
            record(f"prompt/{label}/streams", beq(src, h1))
            record(f"prompt/{label}/x_pre_post_comb", beq(x2, x1) and beq(pre2, pre1) and beq(p2, p1) and beq(c2, c1))
    ok = all(r[0] == r[1] for r in res.values())
    return {"check": "T-K'", "ok": ok, "results": {k: f"{v[1]}/{v[0]}" for k, v in sorted(res.items())},
            "verdict": "bit-identical: cut by the stretch hand-off (post)" if ok else
            "NOT bit-identical: run every stage with TFPP_DS41_CUT_POST=carry (F1) and repeat every test"}


def run_layer(args) -> dict:
    import torch

    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.model import Model, mm, wo_a_out

    from .load import AliasShards, _cut_gate, namer_for

    cfg = Cfg.read(args.model)
    i = int(args.layer)
    k = int(args.experts or cfg.n_routed)
    sh = AliasShards(args.model, namer_for(args.model))
    try:
        one = W.load_block(sh, cfg, f"layers.{i}", i, 0, 1, k)
        halves = [W.load_block(sh, cfg, f"layers.{i}", i, r, 2, k) for r in (0, 1)]
    finally:
        sh.close()
    if args.experts:
        for lay in (one, *halves):
            _cut_gate(lay, k)
    shells = []
    for _ in range(2):                           # what Model.moe reads: cfg, its scratch (one a width), the flag
        shell = Model.__new__(Model)
        shell.cfg, shell.scratch, shell.graph_moe = cfg, {}, False
        shells.append(shell)
    gen = torch.Generator(device="cuda").manual_seed(3)
    report = []

    def judge(name: str, y1, y2, rows: int) -> None:
        x, y = y1.float().reshape(-1), y2.float().reshape(-1)
        cos = float(torch.nn.functional.cosine_similarity(x, y, dim=0)) if x.numel() else 1.0
        rel = float((x - y).abs().max() / y.abs().max().clamp_min(1e-30)) if x.numel() else 0.0
        report.append({"item": name, "rows": rows, "cos": cos, "rel": rel, "bitwise": beq(y1, y2),
                       "ok": cos >= 0.9999 and rel <= 2 ** -6})

    for rows in (6, 257):
        x = (torch.randn((rows, cfg.dim), generator=gen, device="cuda")).to(torch.bfloat16)
        y1 = Model.moe(shells[0], one, x)
        parts = [Model.moe(shells[1], lay, x) for lay in halves]
        judge("moe", y1, parts[0] + parts[1], rows)
        o = (torch.randn((rows, cfg.n_heads, cfg.head_dim), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        y1 = mm(one.wo_b, wo_a_out(one, o), torch.float32)
        hl = cfg.n_heads // 2
        parts = [mm(lay.wo_b, wo_a_out(lay, o[:, r * hl:(r + 1) * hl].contiguous()), torch.float32)
                 for r, lay in enumerate(halves)]
        judge("wo_ab", y1, parts[0] + parts[1], rows)
    return {"check": "T-K", "layer": i, "experts": k, "ok": all(r["ok"] for r in report), "results": report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.ds41.kernel_check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hc", action="store_true")
    g.add_argument("--layer", type=int)
    ap.add_argument("--model", default="")
    ap.add_argument("--layers", default="2,5,7", help="--hc: the layers whose mixes are used (with --model)")
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.layer is not None and not a.model:
        raise SystemExit("--layer needs --model")
    from . import env

    try:
        env.check_env("kernel_check")
        env.check_device()
    except env.EnvError as exc:
        print(f"[tfpp-ds41] {exc}", flush=True)
        return 1
    import torch

    torch.cuda.set_device(0)
    out = run_hc(a) if a.hc else run_layer(a)
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
