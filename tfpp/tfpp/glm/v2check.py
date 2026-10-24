"""T0e: is a stream's row in a batched verify window the row of that stream's window alone? (spec v2 sections 15.4, 16)

MIA claims it (G/verify.py:5-12: "every segment's rows get the bits of that stream's window alone"; G/multi_prefill.py:
27-28 for prompt chunks) on its own shapes; nothing has run at world=1 with this model's kernel shapes. Every equivalence
test of v2 rests on it, so this checks it first, on ONE process (no chain, no warm-up, no scheduler traffic):

  1. FORWARD. P slots with prompts of different lengths (one past the dense limit with a long-context engine), then,
     for window mixes of 1x1, 4x8, (1, 3, 16, 12) and (7, 7, 9, 9) rows, random tokens: the logits and DFlash2 taps of
     every segment from ``SerialVerify`` (each window its own forward: the solo kernels), from ``BatchedVerify`` on its
     CUDA graph and from ``BatchedVerify`` eagerly must be bit-equal. Repeated ``--rounds`` times. A mix wider than the
     verify window (TF_GLM_MULTI_WINDOW 16 or 24) is scaled down to it; no batched mix at all is a failure.
  2. COMMIT. The same random windows and random keeps run through ``SerialVerify`` (forward + commit per window) and
     through ``BatchedVerify``; every later window's logits are a function of what the commits left (KDA states,
     parity, conv windows, positions), so the two sequences' logits hashes must be equal.

  python -m tfpp.glm.v2check --role leader --model DIR --layers 0:L --truncate L --tap-layers a,b,c,d,e \\
         --drafter DIR --parallel 4 --context N [--pool-rows N] [--rounds 20] [--check-out FILE]

  3. PIECES. The other half of the claim (G/multi_prefill.py:27-28): a piece of a multi-prompt chunk
     (``multi_prefill.prefill_pieces``, MFILL) gets the bits of the same rows run as a chunk of its own
     (``decode.prefill_chunk``). Fresh states, prompts of different lengths cut at the prompt grid point 64 (a piece
     that stops mid-prompt, one that ends its prompt, a prompt that is a single piece), mixes of 2, 3 and P prompts:
     the head rows of the pieces that end their prompts, every piece's DFlash2 tap rows and the logits of a window run
     after the prompts (the state the pieces left) must be bit-equal to the same prompts run one chunk at a time. The
     two runs share their slots and extents, so the solo run's window is taken before the pieces run, and the
     extents' cache rows are zeroed in between.

Exit 0: everything equal (then level B of the equivalence tests is a criterion); 1: a difference, named.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SHAPES = ((1,), (8, 8, 8, 8), (1, 3, 16, 12), (7, 7, 9, 9))
EXTENT = 2048


def log(msg: str) -> None:
    print(f"[tfpp-glm] T0e: {msg}", flush=True)


def prompt_lengths(P: int, dense_limit: int, long_context: bool, headroom: int = 0) -> list[int]:
    """Lengths of the P streams' prompts: short, a few hundred, just under the dense limit, and (long context) past it.
    A dense engine (no indexer) refuses a window past ``dense_limit``: the longest prompt leaves ``headroom`` rows."""

    top = dense_limit - 20 if long_context else dense_limit - headroom - 32
    base = [37, 300, max(64, top), dense_limit + 300 if long_context else 701]
    return [base[k % len(base)] + 11 * (k // len(base)) for k in range(P)]


def fit_shape(shape, rows_max: int) -> tuple:
    """A window mix scaled down to at most ``rows_max`` rows (TF_GLM_MULTI_WINDOW may be 16 or 24, under the 32 rows of
    the wider mixes): the same number of segments, each at least one row. A mix already that small is unchanged."""

    total = sum(shape)
    if total <= rows_max:
        return tuple(shape)
    out = [max(1, n * rows_max // total) for n in shape]
    while sum(out) > rows_max and max(out) > 1:
        out[out.index(max(out))] -= 1
    return tuple(out)


def clear_extent(st) -> None:
    """Zero the cache rows a State views (its extent's latent K/V, the MTP head's, the indexer's pools and its slot's
    rings): a run on the extent then cannot read a row it failed to write as one an earlier run left there."""

    import torch

    tensors = list(getattr(st, "kc", None) or []) + list(getattr(st, "vc", None) or [])
    tensors += [getattr(st, "mtp_kc", None), getattr(st, "mtp_vc", None)]
    for trio in getattr(st, "index", None) or []:
        tensors += list(trio)
    for t in tensors:
        if t is None:
            continue
        try:
            t.zero_()
        except (RuntimeError, TypeError, NotImplementedError):     # a dtype without a fill kernel: zero its bytes
            t.view(torch.uint8).zero_()


def run(eng, rounds: int, seed: int = 20261023) -> dict:
    import numpy as np
    import torch

    from tensorfold.families.glm5_next.cuda import decode as D
    from tensorfold.families.glm5_next.cuda.forward import State
    from tensorfold.families.glm5_next.cuda.pool import align_up
    from tensorfold.families.glm5_next.cuda.verify import Segment, SerialVerify

    from . import part

    e, w, verify = eng.e, eng.w, eng.multi.verify
    P = eng.parallel
    vocab = min(int(w.cfg.vocab), 150000)
    lens = prompt_lengths(P, int(w.cfg.dense_limit), bool(e.w.meta.get("long_context")), rounds * 16)
    size = align_up(max(lens) + rounds * 16 + 64, EXTENT)
    if size * P > e.caches.rows:
        raise RuntimeError(f"the pool's {e.caches.rows} tokens cannot hold {P} extents of {size}: --pool-rows "
                           f"{size * P} or more")
    serial = SerialVerify(e, taps=bool(verify.taps))
    result = {"forward": [], "commit": [], "pieces": [], "failures": []}

    def fresh(rng):
        """P streams on their own extents and slots, each prefilled with its prompt (the same tokens every call)."""

        sts = []
        for k in range(P):
            st = State(w, size, e.rows, caches=e.caches, base=k * size, slots=e.slots, slot=k)
            st.reset()
            prompt = [int(t) for t in rng.integers(0, vocab, lens[k])]
            prev = e.use(st)
            try:
                for start in range(0, len(prompt), e.prefill_rows):
                    D.prefill_chunk(e, prompt, start, min(e.prefill_rows, len(prompt) - start), drafter=None,
                                    mtp=False, head=False)
            finally:
                e.use(prev)
            sts.append(st)
        torch.cuda.synchronize()
        return sts

    def snapshot(v):
        return [x.clone() for x in v.logits], [None if t is None else t.clone() for t in v.taps]

    def same(a, b) -> bool:
        return all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and all(
            (p is None and q is None) or torch.equal(p, q) for p, q in zip(a[1], b[1]))

    # -- 1. forward ------------------------------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    sts = fresh(rng)
    graphs = verify.graphs
    for rd in range(rounds):
        shape = fit_shape(SHAPES[rd % len(SHAPES)][:P], int(verify.rows_max))
        segs = [Segment(sts[k], [int(t) for t in rng.integers(0, vocab, n)]) for k, n in enumerate(shape)]
        R = sum(len(s.tokens) for s in segs)
        if R > verify.rows_max:
            continue
        ref = snapshot(serial.forward(segs))
        got = snapshot(verify.forward(segs))                              # its graph for this R
        verify.graphs = {}
        try:
            eager = snapshot(verify.forward(segs))
        finally:
            verify.graphs = graphs
        row = {"round": rd, "shape": list(shape), "R": R, "graph": R in graphs,
               "graph_equal": same(ref, got), "eager_equal": same(ref, eager)}
        result["forward"].append(row)
        if not (row["graph_equal"] and row["eager_equal"]):
            result["failures"].append(f"forward round {rd} shape {shape}: graph {row['graph_equal']}, eager "
                                      f"{row['eager_equal']}")
    if not any(len(r["shape"]) > 1 for r in result["forward"]):
        result["failures"].append(f"forward: no batched window of two or more segments was checked (rounds "
                                  f"{rounds}, verify.rows_max {verify.rows_max}): nothing shown")
    # -- 2. commit -------------------------------------------------------------------------------------------------
    def sequence(use_serial: bool) -> list:
        rng = np.random.default_rng(seed + 1)
        sts = fresh(rng)
        seen = []
        for rd in range(rounds):
            shape = fit_shape(SHAPES[rd % len(SHAPES)][:P], int(verify.rows_max))
            segs = [Segment(sts[k], [int(t) for t in rng.integers(0, vocab, n)]) for k, n in enumerate(shape)]
            keeps = [int(rng.integers(1, len(s.tokens) + 1)) for s in segs]
            if sum(len(s.tokens) for s in segs) > verify.rows_max:
                continue
            seam = serial if use_serial else verify
            v = seam.forward(segs)
            seen.append([part.devhash_rows(lg[:len(s.tokens)]) for lg, s in zip(v.logits, segs)])
            seam.commit(segs, keeps)
        torch.cuda.synchronize()
        return seen

    a, b = sequence(True), sequence(False)
    if not a or len(a) != len(b):
        result["failures"].append(f"commit sequence: {len(a)} serial and {len(b)} batched windows checked")
    for rd, (x, y) in enumerate(zip(a, b)):
        ok = x == y
        result["commit"].append({"round": rd, "equal": ok})
        if not ok:
            result["failures"].append(f"commit sequence: the logits of window {rd} differ after the earlier commits")
    # -- 3. pieces: a multi-prompt chunk's piece against the same rows as a chunk of their own ----------------------
    from tensorfold.families.glm5_next.cuda import multi_prefill

    class Taps:
        """Stands in for a DFlash2 context: keeps the tap rows it is handed (``prefill_chunk`` / ``prefill_pieces``)."""

        def __init__(self) -> None:
            self.rows: list = []

        def add_taps(self, t) -> None:
            self.rows.append(t.clone())

    grid = 64
    plens = [37, 100, 64, 300, 150, 77, 200, 90][:P]
    plens = [plens[k % len(plens)] for k in range(P)]
    room = min(int(e.prefill_rows), int(e.pbuf.rows))
    if sum(plens) > room:
        plens = [max(8, room // P - 1)] * P
    for mix in sorted({tuple(range(min(2, P))), tuple(range(min(3, P))), tuple(range(P))}):
        prng = np.random.default_rng(seed + 2)
        prompts = [[int(t) for t in prng.integers(0, vocab, plens[k])] for k in mix]
        cut = [min(len(p), grid) for p in prompts]
        wtoks = [[int(t) for t in prng.integers(0, vocab, 3)] for _ in mix]

        def blank():
            out = []
            for k in mix:
                st = State(w, size, e.rows, caches=e.caches, base=k * size, slots=e.slots, slot=k)
                st.reset()
                out.append(st)
            return out

        def stages():
            """[(index, start, stop, head)] per stage: every prompt's first piece, then the tails of the longer ones."""

            first = [(i, 0, cut[i], cut[i] == len(prompts[i])) for i in range(len(mix))]
            tails = [(i, cut[i], len(prompts[i]), True) for i in range(len(mix)) if cut[i] < len(prompts[i])]
            return [first, tails]

        solo_st, solo_taps, solo_heads = blank(), [Taps() for _ in mix], {}
        for stage in stages():
            for i, start, stop, head in stage:                    # one chunk each, in the order of the mix
                prev = e.use(solo_st[i])
                try:
                    out = D.prefill_chunk(e, prompts[i], start, stop - start, drafter=solo_taps[i], mtp=False,
                                          head=head)
                finally:
                    e.use(prev)
                if head:
                    solo_heads[i] = out
        # the window after the solo prompts, NOW: the pieces' states below are views of the same slots and extents
        # (State(..., base=k * size, slot=k)), so their reset and writes replace what the solo run left. A forward
        # does not commit: the solo state stays as it was while it is read.
        sa = snapshot(serial.forward([Segment(a, t) for a, t in zip(solo_st, wtoks)]))
        pc_st, pc_taps, pc_heads = blank(), [Taps() for _ in mix], {}
        for st in pc_st:
            clear_extent(st)          # rows the pieces fail to write must not read as the solo run's right ones
        for stage in stages():
            pieces = [multi_prefill.Piece(pc_st[i], prompts[i][start:stop], drafter=pc_taps[i], head=head)
                      for i, start, stop, head in stage]
            heads = multi_prefill.prefill_pieces(e, pieces)
            for (i, _, _, head), h in zip(stage, heads):
                if head:
                    pc_heads[i] = h
        torch.cuda.synchronize()
        row = {"mix": list(mix), "lengths": [len(p) for p in prompts],
               "pos_equal": [int(a.pos) for a in solo_st] == [int(a.pos) for a in pc_st],
               "heads_equal": sorted(solo_heads) == sorted(pc_heads) and all(
                   torch.equal(solo_heads[i], pc_heads[i]) for i in solo_heads),
               "taps_equal": all(len(a.rows) == len(b.rows) and all(torch.equal(x, y) for x, y in zip(a.rows, b.rows))
                                 for a, b in zip(solo_taps, pc_taps))}
        sb = snapshot(serial.forward([Segment(a, t) for a, t in zip(pc_st, wtoks)]))
        row["window_equal"] = same(sa, sb)
        result["pieces"].append(row)
        bad = [k for k in ("pos_equal", "heads_equal", "taps_equal", "window_equal") if not row[k]]
        if bad:
            result["failures"].append(f"pieces of a {len(mix)}-prompt chunk (lengths {row['lengths']}): {bad} differ "
                                      "from the same rows run as chunks of their own")
    result["ok"] = not result["failures"]
    return result


def main(argv: list[str] | None = None) -> int:
    from . import env, pins, serve

    ap = serve.build_parser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--check-out", default="")
    a = ap.parse_args(argv)
    a.role = "leader"
    serve.check_args(a)
    if a.parallel < 2 or a.mbench or a.bench:
        raise SystemExit("v2check: --parallel 2..8, one process (--layers 0:END), no --bench / --mbench")
    if a.layers[0] != 0 or a.listen:
        raise SystemExit("v2check runs the whole (shortened) model in this process: --layers 0:L, no --listen")
    try:
        env.check_env("leader")
        pins.check_tree()
        env.check_device()
    except (env.EnvError, pins.TreeMismatch) as exc:
        print(f"[tfpp-glm] {exc}", flush=True)
        return 1
    from tensorfold.families.glm5_next.cuda.engine import DEFAULT_POLICY

    from .v2tail import PipeMultiEngine

    first, end = a.layers
    eng = PipeMultiEngine(Path(a.model), layers=(first, end), truncate=a.truncate, link=None,
                          policy=a.policy or DEFAULT_POLICY, drafter=a.drafter or None, context=a.context,
                          serial_only=a.no_drafts, prefill_rows=a.prefill_rows, parallel=a.parallel,
                          graphs=not a.no_graphs, tap_layers=a.tap_layers, reserve_gib=a.reserve_gib,
                          budget_gib=a.budget_gib, pure=a.pure, warmup=False, pool_rows=a.pool_rows,
                          kept_entries=a.kept_entries)
    t0 = time.perf_counter()
    res = run(eng, a.rounds)
    log(f"{len(res['forward'])} forward mixes, {len(res['commit'])} committed windows and {len(res['pieces'])} "
        f"multi-prompt chunk mixes in "
        f"{time.perf_counter() - t0:.1f}s: " + ("PASS" if res["ok"] else "FAIL"))
    for f in res["failures"]:
        log("  " + f)
    if a.check_out:
        Path(a.check_out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    import os

    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
