"""tfpp-glm v2 (--parallel 2..8): a follower stage (spec v2 section 8).

MIA's own ``MultiDecoder.follow`` loop applies the leader's messages: what the leader decides (ADMIT, EVICT, MOVE, GROW,
FILL, MFILL, ROUND, FINISH, IDLE) every stage applies in the same order through the same code, so the pool, the slots,
the kept prompts and the positions stay equal on all of them. This module puts a stage under that loop:

- ``FollowerGlm``: what ``MultiDecoder`` reads off its engine, and the link side (``_share(None)`` = the next MULTI frame;
  every control frame goes to the stage before before it is handled, so stage 0 starts as early as it can);
- ``FollowerMulti``: ADMIT without a grammar (pictures: stage 0 keeps the rows the frame brought), ``_fill`` /
  ``_fill_many`` without the head and the sample (their prompt ACT is moved inside ``v2part._compute_pp_multi``), and
  ``_round``: WINDOWS, the window's ACT in and out, KEEPS, the commit;
- ``FollowerBatchedVerify``: ``BatchedVerify`` with the window's ACT moved around ``forward``.

A follower never samples, drafts, compiles a grammar or tells the leader anything but ACT; a failure leaves through the
exception (``follower.run`` reports it towards the leader and exits 75).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from tensorfold.cuda.streams import Stream
from tensorfold.families.glm5_next.cuda import decode as D, engine as E, forward as fw, multi as M, multi_prefill
from tensorfold.families.glm5_next.cuda.verify import BatchedVerify, Segment

from . import capacity, part, proto, startup, v2part, v2proto, v2startup, v2wire
from .feed import LocalFeed, NullFeed
from .load import load_range
from .v2proto import Fault, RoundStats, log_rows
from .wire import ActIO, LinkError, wait_any


def log(msg: str) -> None:
    print(f"[tfpp-glm] {msg}", flush=True)


class Quit(Exception):
    pass


def empty_head():
    """The kept head row a follower stands in for (the leader keeps the real one; a replay needs only that it exists)."""

    return torch.empty(0, device="cuda")


class FollowerGlm:
    """What MultiDecoder reads off its engine (``self.g``) on a follower stage."""

    world = 1
    comm = drafter = drafter_dir = copy = vision = opener = spill_cfg = costs = None
    serial_only = False

    def __init__(self, fol, agree: dict) -> None:
        self.fol, self.e, self.w, self.stage = fol, fol.e, fol.w, fol.stage
        self.rank = fol.stage.count - 1 - fol.stage.index            # hops from the leader (>= 1)
        self.grid, self.multi_window = agree["grid"], int(agree["multi_window"])
        self.limit, self.cache_entries = int(agree["limit"]), int(agree["entries"])   # _keep's cap: G/multi.py:1091
        self.shared, self.model_dir, self.eos = E.shared_prefix(), Path(fol.args.model), tuple(fol.w.cfg.eos)
        self.agreed, self.incoming = [], {}
        self.fault = Fault(os.environ.get("TFPP_GLM_FAULT_FOLLOWER", ""), kill=True, armed=True)   # SIGKILL self (tests)

    def _ring(self) -> None:
        pass

    def _gather_ints(self, values):
        return [list(values)]

    def _await_bell(self) -> None:           # idle: v1's idle branch (T/follower.py:104-111), nothing consumed
        f = self.fol
        while True:
            if wait_any([f.down, f.up]) is f.down:
                return
            g = f.up.recv(timeout=f.args.io_timeout)
            if g.op == proto.ERROR:
                raise LinkError(f"stage {g.hdr.get('stage')} failed in {g.hdr.get('where')}: {g.hdr.get('msg')}")
            raise proto.ProtocolError(f"an unasked {g.op} from the stage before while idle")

    def next_frame(self, op: str):           # busy: KeepWarm; upstream first, then handled
        f = self.fol
        with f.warm:
            fr = f.down.recv(timeout=f.args.io_timeout, keep_raw=True)
        if f.up is not None:
            f.up.send_raw(fr.raw, fr.payload)
        if fr.op == proto.QUIT:
            raise Quit()
        if fr.op == proto.ERROR:
            raise LinkError(f"the leader stopped the chain: {fr.hdr.get('where')}: {fr.hdr.get('msg')}")
        if fr.op != op or fr.hdr.get("g") != 0:
            raise proto.ProtocolError(f"expected {op} g=0, got {fr.op} {fr.hdr}")
        return fr

    def _share(self, values):                 # MIA follow(): the next message (G/multi.py:2325)
        if values is not None:
            raise proto.ProtocolError("a follower only receives")
        fr = self.next_frame(v2proto.MULTI)
        if self.stage.first:
            for name, _, _ in fr.hdr["parts"]:
                sid = v2proto.img_sid(name)
                if sid is not None:
                    self.incoming[sid] = fr.tensor(name)         # host rows; LocalFeed uploads per chunk
        return [int(v) for v in fr.ints("ints").tolist()]


class FollowerBatchedVerify(BatchedVerify):
    """MIA's batched verify seam on a follower: the window's x (and the carried taps) in from the stage before (not on
    stage 0), the window, the rows on to the next stage."""

    io = stage = fields = stats = None

    @torch.no_grad()
    def forward(self, segments):
        t0 = time.perf_counter()
        R = sum(len(s.tokens) for s in segments)
        for s in segments:
            fw.check_room(self.w, s.st, len(s.tokens))
        w0 = self.io.recv_ms
        if not self.stage.first:
            v2wire.recv_rows(self.io, self.b, R, self.fields)
        v = super().forward(segments)                  # graph or eager; [R, 1] stand-in logits (T/part.py:73)
        v2wire.send_rows(self.io, self.b, R, self.fields)   # a follower is never last (T/follower.py:55-56)
        if self.stats is not None:
            self.stats.add(R, time.perf_counter() - t0, (self.io.recv_ms - w0) / 1e3)
        return v


REFUSED = {M.SLOT, M.FILLPART, M.MFILLPART, M.LOAD, M.LOADED, M.FLUSH, M.SPILL, M.TOUCH}


class FollowerMulti(M.MultiDecoder):
    """MIA's MultiDecoder on a follower: the leader's ops applied in order (``follow`` is MIA's)."""

    def apply(self, msg):                              # G/multi.py:2331-2409, ADMIT without a grammar
        for op, p in self.parse(msg):
            if op in REFUSED:
                raise proto.ProtocolError(f"op {M.OPS[op]}: not in a tfpp-glm pipeline")
            if op != M.ADMIT:
                super().apply([op, len(p), *p])
                continue
            a = self._parse_admit(p)                   # G/multi.py:928-949
            s = Stream(list(a["prompt"]), a["count"], a["sampling"], draft=a["draft"], stop_eos=a["stop_eos"],
                       sid=a["sid"])
            feed = None
            if a["positions"]:                         # lane.feed alike on every stage (G/multi.py:1902-1906)
                if self.g.stage.first:
                    if a["sid"] not in self.g.incoming:
                        raise proto.ProtocolError(f"ADMIT of stream {a['sid']} brings {len(a['positions'])} image "
                                                  "rows but the frame carries none")
                    feed = LocalFeed(a["positions"], self.g.incoming.pop(a["sid"]))
                else:
                    feed = NullFeed(a["positions"])
                s.vision = feed
            self._admitted(a, s, feed, None)

    @torch.no_grad()
    def _fill(self, lane, stop, *, layers=None):       # G/multi.py:1930-1976 without the head and the sample
        if layers is not None:
            raise proto.ProtocolError("FILLPART: layer-sliced fills are off")
        e, s = self.e, lane.s
        n, start, replay = len(s.prompt), lane.st.pos, lane.head is not None
        if not (start < stop <= n or replay and start == stop == n) or stop - start > e.prefill_rows:
            raise RuntimeError(f"stream {lane.sid}: a prompt chunk {start} .. {stop} of {n}")
        prev = e.use(lane.st)
        try:
            if replay:
                lane.head = None                       # the leader samples the kept head; nothing runs here
            else:
                D.prefill_chunk(e, s.prompt, start, stop - start, drafter=None, feed=lane.feed, mtp=False,
                                head=False)            # ACT chunk in/out inside _compute_pp_multi
            if stop < n:
                if stop in lane.stops:
                    self._keep(lane, stop, own=stop == lane.point)
                return None
            if lane.point == n and not replay:
                self._keep(lane, n, head=empty_head(), own=True)   # a replay's check (G/multi.py:1008-1012)
        finally:
            e.use(prev)
        self._started(lane, 0)                         # decoding; copies stay None (g.copy None: :1987)
        return None

    @torch.no_grad()
    def _fill_many(self, group, *, layers=None):       # G/multi.py:1995-2051 without heads and samples
        if layers is not None:
            raise proto.ProtocolError("MFILLPART: layer-sliced fills are off")
        pieces = []
        for lane, stop in group:
            start, n = lane.st.pos, len(lane.s.prompt)
            if lane.decoding or lane.feed is not None or lane.head is not None:
                raise RuntimeError(f"stream {lane.sid} cannot join a multi-prompt chunk")
            if not start < stop <= n or stop - start > self.e.prefill_rows:
                raise RuntimeError(f"stream {lane.sid}: a prompt chunk {start} .. {stop} of {n}")
            pieces.append(multi_prefill.Piece(lane.st, list(lane.s.prompt[start:stop]), drafter=None, head=False))
        multi_prefill.prefill_pieces(self.e, pieces)   # ACT pieces in/out inside _compute_pp_multi
        for lane, stop in group:                       # rank 0's _keep order: the same kid numbering
            n = len(lane.s.prompt)
            if stop < n:
                if stop in lane.stops:
                    self._keep(lane, stop, own=stop == lane.point)
                continue
            if lane.point == n:
                self._keep(lane, n, head=empty_head(), own=True)
        for lane, stop in group:
            if stop == len(lane.s.prompt):
                self._started(lane, 0)
        return [None] * len(group)

    @torch.no_grad()
    def _round(self, lanes, depths=None, lone=False, short=False):
        if lone or short:
            raise proto.ProtocolError("lone / short rounds are off (MULTI_LONE=0, FILL_BUDGET_MS=0)")
        # (ROUND's depths are not checked here as MIA's rank 1 does, G/multi.py:2062-2063: a follower derives no drafts and
        # keeps no ``lane.depth``; rows vs depth is no invariant -- copy drafts, trim and grammar cuts change it. The
        # window's slot/pos/rows and KEEPS' ``end`` are what cross-check the lanes.)
        g = self.g
        w = g.next_frame(v2proto.WINDOWS)
        h, ids = w.hdr, [int(t) for t in w.ints("ids").tolist()]
        try:
            segs = v2proto.check_windows(h, len(ids), received=self.received,
                                         lanes=[[l.slot, l.st.pos] for l in lanes], rows_max=self.rows_max,
                                         window=self.window)
        except ValueError as exc:
            raise proto.ProtocolError(f"WINDOWS {h} does not fit this stage's lanes: {exc}") from None
        out, at = [], 0
        for l, (_, _, k) in zip(lanes, segs):
            out.append(Segment(l.st, ids[at:at + k]))
            at += k
        self.verify.fields = v2proto.act_round(0, h["rid"], len(ids))
        g.fault.hit("windows")
        self.verify.forward(out)
        g.fault.hit("act")
        k = g.next_frame(v2proto.KEEPS).hdr
        try:
            keeps = v2proto.check_keeps(k, rid=h["rid"], rows=[x[2] for x in segs])
        except ValueError as exc:
            raise proto.ProtocolError(f"KEEPS {k} for WINDOWS rid {h['rid']}: {exc}") from None
        self.verify.commit(out, keeps)                 # G/verify.py:269-293
        if [int(l.st.pos) for l in lanes] != [int(x) for x in k["end"]]:
            raise proto.ProtocolError(f"KEEPS end {k['end']}, this stage is at {[l.st.pos for l in lanes]}")


class V2Follower:
    """What ``Follower.v2`` holds: the built stage and its loop."""

    def __init__(self, fol, glm: FollowerGlm, multi: FollowerMulti) -> None:
        self.fol, self.glm, self.multi = fol, glm, multi

    def run(self) -> int:
        """Apply the leader's messages until QUIT (0); anything else leaves as an exception (``follower.run`` reports
        it towards the leader and exits 75)."""

        f = self.fol
        try:
            self.multi.follow()
        except Quit:
            log(f"{f.where}: QUIT; sent {f.down.sent / 2**20:.1f} MiB down"
                + (f", received {f.up.received / 2**20:.1f} MiB from the stage before" if f.up else ""))
            return 0
        return 0


def start(fol, go: dict, prompt_precision, KeepWarm) -> V2Follower:
    """``Follower._start`` (T/follower.py:48-97) for P > 1, after GO: the stage, its layers, CAP, AGREE (checked), the
    engine on the pool without graphs of its own, the batched verify seam (graphs captured on scrubbed streams), the
    decoder, READY."""

    args = fol.args
    a, b = fol.layers
    timeout = args.connect_timeout
    index, count = startup.stage_of(go, (a, b))
    if index == count - 1:
        raise proto.ProtocolError("a follower cannot be the chain's last stage")
    taps = tuple(int(t) for t in go["taps"])
    ranges = [tuple(r) for r in go["stages"]]
    cin, cout = proto.tap_routes(taps, ranges)[index]
    prompt_precision.set_fp8(bool(go["fp8"]))           # before any weight loads (TF/prompt_precision.py:17)
    fol.warm = KeepWarm()
    stage = part.Stage(index=index, count=count, layers=(a, b), first=a == 0, last=False, taps=taps,
                       carry_in=cin, carry_out=cout)
    fol.io = ActIO(stage, fol.up, fol.down, warm=fol.warm, timeout=args.io_timeout, tracer=fol.tracer)
    stage.io = fol.io
    fol.stage = stage
    v2part.install(stage)
    log(f"stage {index} of {count} {fol.where}: loading (taps in {list(cin)}, out {list(cout)}; v2 --parallel "
        f"{go['P']})")
    t0 = time.perf_counter()
    fol.w = load_range(args.model, a, b, is_first=a == 0, is_last=False, mtp=False, drafter=False,
                       prefetch_next=not args.no_prefetch_next)
    log(f"stage {index} {fol.where}: layers loaded in {time.perf_counter() - t0:.1f}s "
        f"({fol.w.nbytes() / 2**30:.2f} GiB)")
    prefill_rows = int(go["prefill_rows"])
    staging = capacity.staging_bytes(prefill_rows, len(cin), len(cout), first=a == 0, last=False)
    cap = capacity.measure_budget(reserve_gib=args.reserve_gib, budget_gib=args.budget_gib, staging=staging)
    cap["layers"] = [a, b]
    agree = startup.follower_cap(fol.down, fol.up, cap, timeout=timeout)
    fol.agree = agree
    multi_window = E.multi_window()
    v2startup.check_agree(go, agree, multi_window=multi_window)
    if fol.up is not None:
        fol.up.set_staging(recv_bytes=proto.link_staging(prefill_rows, len(cin)))
    fol.down.set_staging(send_bytes=proto.link_staging(prefill_rows, len(cout)))
    part.scrubbing(True)
    try:
        fol.e = D.Engine(fol.w, capacity=agree["pool_rows"], max_rows=agree["max_rows"],
                         prefill_rows=agree["prefill_rows"], graphs=False,
                         graph_rows=tuple(agree["graph_rows"]), long_context=agree["long_context"],
                         taps=tuple(agree["taps"]), grid=agree["grid"], split=None, mtp_rows=agree["max_rows"],
                         streams=agree["parallel"], pool_rows=agree["pool_rows"], kv=agree["kv"])
        verify = FollowerBatchedVerify(fol.e, taps=tuple(agree["taps"]), rows=multi_window)   # F-8: AGREE's taps
        verify.io, verify.stage = fol.io, stage
        if agree["graphs"] and os.environ.get("TF_GLM_MULTI_GRAPHS", "1") != "0":
            verify.capture()                                                 # F-11: before any admission
    finally:
        part.scrubbing(False)
    if os.environ.get("TFPP_GLM_TIMING") == "1":
        verify.stats = RoundStats(f"stage {index}")
    if os.environ.get("TFPP_GLM_TIME_ROWS") == "1" and verify.graphs:
        if not stage.first:
            verify.b.x.normal_()                       # timings on rows that route like text, not on leftovers
        log_rows(f"stage {index}", verify.time_rows())
    glm = FollowerGlm(fol, agree)
    multi = FollowerMulti(glm, agree["parallel"], verify=verify)
    if fol.tracer is not None:
        from . import v2trace

        v2trace.install_follower(fol.tracer, glm, multi)
    startup.follower_ready(fol.down, fol.up,
                           v2startup.built_info_v2((a, b), fol.e, verify, agree["pool_rows"], multi_window,
                                                   verify.segs), timeout=timeout)
    log(f"stage {index} {fol.where}: ready (window {agree['window']}, cache slots {agree['capacity']}, pool "
        f"{agree['pool_rows']} tokens, {agree['entries']} kept entries, "
        f"{'long contexts' if agree['long_context'] else 'dense'})")
    return V2Follower(fol, glm, multi)
