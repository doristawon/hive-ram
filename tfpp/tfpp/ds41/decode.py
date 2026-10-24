"""A stage's share of the fork's decode rounds (spec section 7): RoundDecoder / RoundRunner on a layer range.

The fork already cuts a round into stretches of layers (before each Engram layer, ENGRAM_SPLIT) whose state between
them lives on the decoder (``self._state``: F/rounds.py:421-422, 447-449), each stretch posting its last sublayer's
pending output first (F/rounds.py:549-550) and the next starting from ``pending = None`` (486). A stage is a run of
stretches [lo, ..., hi): its first stretch takes the state from the stage before (``_enter``, captured with the
stretch: replica rows ``index_copy_``'d to the round's targets, h and pre cloned out of the static input views), its
last one leaves the state to the stage after it (``_exit``: into the static output views, replica rows
``index_select``'d from the planes). ``run`` (R1, R2) reads the round's ACT after the first Engram reads are submitted
and sends it once every stretch is enqueued; a graph replay reads and writes only the static views, so a captured
round moves its state without a host step inside it.

StageBuffers holds the static views of every (R, bucket) a stage may capture, made before any capture (as the fork's
SHARED_OUT pair, F/rounds.py:37-42, 626-630). N1 (spec 7.4): a capture's eager pass reads them before any round wrote
them, so they are reset to the fork's "no entry" values first, under the KeepWarm gate.
"""

from __future__ import annotations

import math

import torch

from tensorfold.families.deepseek_v41.cuda import kernels as K, model as _model, rounds as RD
from tensorfold.families.deepseek_v41.cuda.model import l2_fork
from tensorfold.families.deepseek_v41.cuda.rounds import MAX_ROWS, RoundDecoder, RoundRunner
from tensorfold.families.deepseek_v41.ops import BF16, F32

from . import env, patch, plan as PL
from .wire import TORCH_DT, fill_value

CARRY = env.cut_post() == "carry"       # F1 (spec 3.6): read once, at import (env.check_env has validated it)


class StageBuffers:
    """The static round views of a stage: "in" (from the stage before) and "out" (to the stage after), one flat
    buffer a part sized for MAX_ROWS rows and the widest bucket; ``views(side, R, b)`` its contiguous [R, ...] views."""

    def __init__(self, plan, mv, cap: int, *, f1: bool, bucket_min: int) -> None:
        self.P, self.mv, self.f1 = plan, mv, bool(f1)
        self.bs = PL.buckets(cap, bucket_min)
        self.flat: dict[str, dict[str, torch.Tensor]] = {"in": {}, "out": {}}
        for side, carry in (("in", plan.cin), ("out", plan.cout)):
            if carry is None:
                continue
            need: dict[str, tuple[str, int]] = {}
            for b in self.bs:
                for name, dt, shape in PL.round_parts(mv, carry, MAX_ROWS, b, self.form(b), f1=self.f1):
                    n = math.prod(shape)
                    if n > need.get(name, (dt, -1))[1]:
                        need[name] = (dt, n)
            for name, (dt, n) in need.items():
                self.flat[side][name] = torch.full((max(n, 1),), fill_value(name), dtype=TORCH_DT[dt], device="cuda")
        self._views: dict = {}

    def form(self, bucket: int) -> str:
        if self.mv.cand_source < 0:
            return "cand"
        return PL.cand_form(self.mv, bucket, idx_on=K.on("idx"), cand_only_on=K.on("cand_only"))

    def views(self, side: str, R: int, b: int) -> dict:
        key = (side, R, b)
        v = self._views.get(key)
        if v is None:
            carry = self.P.cin if side == "in" else self.P.cout
            v = {}
            for name, _, shape in PL.round_parts(self.mv, carry, R, b, self.form(b), f1=self.f1):
                v[name] = self.flat[side][name][:math.prod(shape)].view(shape)
            self._views[key] = v
        return v

    def reset_in(self) -> None:
        for name, t in self.flat["in"].items():
            t.fill_(fill_value(name))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for side in self.flat.values() for t in side.values())


class StageRoundDecoder(RoundDecoder):
    run = patch.patched(RoundDecoder.run, patch.group_ids("round_run", CARRY), "RoundDecoder.run")
    if CARRY:
        _layers_fused = patch.patched(RoundDecoder._layers_fused, patch.group_ids("layers_fused", CARRY),
                                      "RoundDecoder._layers_fused")

    def __init__(self, model, pool, rows: int, bucket: int, taps: bool, out=None, *, rr) -> None:
        super().__init__(model, pool, rows, bucket, taps, out)               # F/rounds.py:76-101
        P = self.plan = rr.plan
        self.rr = rr
        self.e_in = {i: t for i, t in self.e_in.items() if P.lo <= i < P.hi}
        cuts = sorted(i for i in self.e_in if P.lo < i < P.hi) if RD.ENGRAM_SPLIT else []
        self.stretches = list(zip([P.lo, *cuts], [*cuts, P.hi]))            # global layer numbers
        self.form = rr.bufs.form(bucket)
        self.vin = rr.bufs.views("in", rows, bucket) if not P.first else None
        self.vout = rr.bufs.views("out", rows, bucket) if not P.last else None
        self._tfpp_pend = None

    # -- R1 / R2 ------------------------------------------------------------------------------------------------
    def _tfpp_in(self) -> None:
        if not self.plan.first:
            self.rr.io.recv_round(self)

    def _tfpp_out(self) -> None:
        if not self.plan.last:
            self.rr.io.send_round(self)

    # -- a stage's entry and exit, inside its first and last stretch (and their graphs) ------------------------
    def _stretch(self, k: int):
        P = self.plan
        if k == 0 and not P.first:
            self._enter()
        super()._stretch(k)          # lo > 0: reads self._state (F/rounds.py:421-422); hi < L: keeps it (447-449)
        if k == len(self.stretches) - 1 and not P.last:
            self._exit()

    def _enter(self) -> None:
        m, c, P = self.m, self.m.cfg, self.plan
        v, R = self.vin, self.R
        dev = "cuda"
        for s, with_index in P.cin.reps:
            tgt = self._ix(("ctarget", c.compress_ratios[s]))                # the rows the source wrote (F/rounds.py:305)
            codes, scales = self.pool.comp[s]
            codes.index_copy_(0, tgt, v[f"c{s}"])
            scales.index_copy_(0, tgt, v[f"cs{s}"])
            if with_index:
                kc, ks = self.pool.index_k[s]
                kc.index_copy_(0, tgt, v[f"k{s}"])
                ks.index_copy_(0, tgt, v[f"ks{s}"])
        h = v["h"].clone()
        pre = v["pre"].clone()
        x = torch.empty((R, c.dim), dtype=BF16, device=dev)                  # as F/rounds.py:413-418
        part = torch.empty((R * K.HC_BLOCKS * 32,), dtype=F32, device=dev)
        pre_a = torch.empty((R, c.hc), dtype=F32, device=dev)
        pre_f = torch.empty((R, c.hc), dtype=F32, device=dev)
        if self.rr.f1:
            post, comb = v["post"].clone(), v["comb"].clone()                 # the carried post's mixes (spec 3.6)
        else:
            post = torch.empty((R, c.hc), dtype=F32, device=dev)
            comb = torch.empty((R, c.hc, c.hc), dtype=F32, device=dev)
        shared: dict = {}
        cin = P.cin
        if cin.kv_layer is not None:
            shared["kv_layer"] = cin.kv_layer
        if cin.topk is not None:
            shared["topk"] = v["topk"]
        if cin.cand:
            if self.form == "cblk":
                shared["cblk"] = v["cblk"]
            else:
                shared["cand"] = v["cand"].view(torch.bool)
        self._state = (h, pre, x, part, pre_a, pre_f, post, comb, shared, self._seed_taps())

    def _seed_taps(self) -> list:
        """[] -- or, on a leader after DSpark taps (only --no-drafts allows it), the taps buffer with the upstream
        taps' blocks counted, so _layers_fused writes its own taps at their places and _stretch's count check holds
        (F/rounds.py:524-537, 461-463). Those blocks are never read without a drafter (and stay zeros where the
        one-process run holds the real taps: trace.py records a round's taps only on a drafting engine)."""

        P, c, m = self.plan, self.m.cfg, self.m
        if not (P.last and P.up_taps):
            return []
        if K.on("hc") and K.on("glue") and c.dim % 1024 == 0:          # _layers_fused's taps buffer (524-537)
            ntap = sum(1 for i in c.dspark_taps if i < len(m.w.layers))
            out = self.out
            if out is not None and out[1].shape[1] == ntap * c.dim:
                buf = out[1][:self.R]
            else:
                buf = torch.zeros((self.R, ntap * c.dim), dtype=BF16, device="cuda")
            return [buf, P.up_taps]
        return [torch.zeros((self.R, c.dim), dtype=BF16, device="cuda") for _ in range(P.up_taps)]

    def _exit(self) -> None:
        h, pre, x, part, pre_a, pre_f, post, comb, shared, taps = self._state
        v, c, cout = self.vout, self.m.cfg, self.plan.cout

        def put(name: str, src: torch.Tensor) -> None:
            dst = v[name]
            if src.dtype != dst.dtype or list(src.shape) != list(dst.shape):
                raise RuntimeError(f"tfpp-ds41: round exit part {name} is {src.dtype}{list(src.shape)}, the cut "
                                   f"rules say {dst.dtype}{list(dst.shape)}")
            dst.copy_(src)

        put("h", h)
        put("pre", pre)
        if cout.topk is not None:
            put("topk", shared["topk"])
        if cout.cand:
            if self.form == "cblk":
                put("cblk", shared["cblk"])
            else:
                put("cand", shared["cand"].view(torch.uint8))
        for s, with_index in cout.reps:
            tgt = self._ix(("ctarget", c.compress_ratios[s]))
            codes, scales = self.pool.comp[s]
            torch.index_select(codes, 0, tgt, out=v[f"c{s}"])
            torch.index_select(scales, 0, tgt, out=v[f"cs{s}"])
            if with_index:
                kc, ks = self.pool.index_k[s]
                torch.index_select(kc, 0, tgt, out=v[f"k{s}"])
                torch.index_select(ks, 0, tgt, out=v[f"ks{s}"])
        if self.rr.f1:
            if self._tfpp_pend is None:
                raise RuntimeError("tfpp-ds41 F1: the stage's last sublayer was posted (no pending output to carry)")
            put("pend", self._tfpp_pend)
            put("post", post)
            put("comb", comb)
            self._tfpp_pend = None
        self._state = None
        self._idx = {}

    # -- F1 (spec 3.6; the anchors F1a / F1b call these) ---------------------------------------------------------
    def _tfpp_pending_in(self, first: int):
        P = self.plan
        if self.rr.f1 and not P.first and first == P.lo:
            return self.vin["pend"]
        return None

    def _tfpp_keep_pending(self, end: int, pending) -> bool:
        P = self.plan
        if self.rr.f1 and not P.last and end == P.hi:
            self._tfpp_pend = pending
            return True
        return False

    # -- the fork's methods that assume layer 0 .. L-1 here ----------------------------------------------------
    def _l2_next(self, li: int, end: int) -> None:
        """F/rounds.py:469-475 within the stage: the next layer, the head after the last layer on the leader, nothing
        after a follower's last layer (L2 prefetch changes no bit: F/model.py:145-152)."""

        P, w = self.plan, self.m.w
        if li + 1 < P.hi:
            nxt = w.layers[li + 1]
        elif P.last:
            nxt = w.head
        else:
            return
        l2_fork(self.m, w.layers[li], "next", nxt,
                mb=_model.L2_STRETCH_END_MB if li + 1 == end < len(w.layers) else None)

    def capture(self, pool=None) -> None:
        """N1: the static inputs reset (the eager pass and the capture read them before a round wrote them), and
        nothing of KeepWarm's on the GPU meanwhile (spec 7.4, 11.1)."""

        with self.rr.gate.paused():
            if self.vin is not None:
                self.rr.bufs.reset_in()
            super().capture(pool)


class StageRoundRunner(RoundRunner):
    _forward = patch.patched(RoundRunner._forward, patch.group_ids("runner_forward", CARRY), "RoundRunner._forward")

    def __init__(self, model, pool, graphs: bool = True, graph_pool=None, *, plan, mv, io, gate,
                 announce: bool) -> None:
        # F/rounds.py:620-630 without super(): the shared logits / taps pair only on the leader (w.head)
        self.m, self.pool = model, pool
        self.graphs: dict | None = {} if graphs else None
        self.graph_pool = graph_pool
        self.captures = 0
        self.out = None
        if RD.SHARED_OUT and plan.last:
            c, w = model.cfg, model.w
            ntap = sum(1 for i in c.dspark_taps if i < len(w.layers))
            self.out = (torch.empty((MAX_ROWS, w.head.n * model.comm.world), dtype=F32, device="cuda"),
                        torch.empty((MAX_ROWS, max(ntap, 1) * c.dim), dtype=BF16, device="cuda"))
        self.plan, self.mv, self.io, self.gate, self.announce = plan, mv, io, gate, bool(announce)
        self.f1 = bool(io.f1_round)
        from tensorfold.families.deepseek_v41.cuda.graph import BUCKET_MIN

        self.bufs = StageBuffers(plan, mv, pool.cap, f1=self.f1, bucket_min=BUCKET_MIN)
        self.cur_round = 0

    def _tfpp_decoder(self, *a, **k):
        return StageRoundDecoder(*a, **k, rr=self)

    def forward(self, windows: list, replay: bool = True):
        if replay and self.announce:                     # the leader: STEP up the chain first (spec 7.6)
            self.io.ops.step(self, windows)
        return super().forward(windows, replay)
