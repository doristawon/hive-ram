"""A follower stage (spec 8.2): one thread, the leader's control ops in order, the fork's own MultiDecoder steps on
its layers. It never samples or drafts; it mirrors the leader's slots, extents and streams because the leader tells
it every step that changes them (ADMIT, FILL, STEP, FINISH, DROP), each op carrying a digest of the mirrored state
from before it, checked before the op is applied.

Control frames arrive from the next stage (towards the leader) and go to the previous one (towards stage 0) before
they are handled, so stage 0 starts as early as it can; ACT frames come from the stage before and go to the next.
While streams live, the waits for the next op spin KeepWarm (the GPU keeps its clocks between rounds); idle, the stage
also watches the stage before it (an ERROR or EOF there stops it at once, not at the next request).
"""

from __future__ import annotations

import os
import sys

import torch

from tensorfold.cuda.streams import Stream

from . import capacity, env, plan as PL, proto, startup
from .engine import FollowerEngine
from .multi import fault_point, log
from .warm_gate import GatedKeepWarm
from .wire import LinkError, wait_any


class Follower:
    def __init__(self, args, tracer=None) -> None:
        self.args = args
        a, b = args.layers
        self.layers = (a, b)
        self.where = f"[{a},{b})"
        self.tracer = tracer
        self.fault = fault_point(os.environ.get("TFPP_FAULT"), "follower")
        self.steps = 0
        self.down, self.up = startup.follower_connect((a, b), args.down, args.listen,
                                                      connect_timeout=args.connect_timeout,
                                                      io_timeout=args.io_timeout)
        try:
            self._start()
        except BaseException as exc:
            startup.report_error(self.down, self.where, "startup", exc)
            raise

    def _start(self) -> None:
        args = self.args
        a, b = self.layers
        timeout = args.connect_timeout
        mv = PL.ModelView.read(args.model)
        experts, carry = env.experts(), env.cut_post()
        extra = startup.stage_extra(args.model, args.engram or None, mv)
        own = startup.stage_info((a, b), "follower", args.model, experts=experts, carry=carry, extra=extra)
        go = startup.follower_exchange(self.down, self.up, proto.HELLO, "stages", own.to_json(), proto.GO,
                                       timeout=timeout)
        index, count = startup.stage_of(go, (a, b))
        if index == count - 1:
            raise proto.ProtocolError("a follower cannot be the chain's last stage")
        plans = PL.plan_chain(mv, go["stages"], drafts=int(go["drafts"]) > 0)
        if PL.chain_digest(plans) != go["plan_digest"]:
            raise proto.ProtocolError("this stage's plan of the chain differs from the leader's (config or tfpp-ds41 "
                                      "version)")
        if int(go["experts"]) != experts or go["cut_post"] != carry:
            raise proto.ProtocolError(f"GO: experts {go['experts']} cut_post {go['cut_post']}, this stage "
                                      f"{experts} / {carry}")
        P = plans[index]
        log(P.summary())
        self.gate = GatedKeepWarm()
        self.eng = FollowerEngine(args.model, plan=P, mv=mv, role="follower", up=self.up, down=self.down,
                                  drafts=int(go["drafts"]), context=int(go["context"]), parallel=int(go["parallel"]),
                                  graphs=bool(go["graphs"]), experts=experts, engram_dir=args.engram or None,
                                  gate=self.gate, io_timeout=args.io_timeout, tracer=self.tracer)
        cap_positions = int(go["context"]) + int(go["drafts"]) + 1 + 8
        f1 = startup.f1_chunk_now(carry, mv)
        need = capacity.stage_need(mv, P, cap_positions, int(go["parallel"]), drafter=False, chunk=int(go["chunk"]),
                                   f1=f1)
        rx, tx = capacity.stage_staging(mv, P, int(go["chunk"]), cap_positions, f1=f1)
        budget = capacity.measure(reserve_gib=args.reserve_gib, budget_gib=args.budget_gib, staging=need["staging"])
        own_cap = {"layers": [a, b], "pool": need["pool"], "workspace": need["workspace"], "staging": need["staging"],
                   "need": need["need"], "weights": self.eng.weights_bytes, **budget}
        agree = startup.follower_exchange(self.down, self.up, proto.CAP, "caps", own_cap, proto.AGREE, timeout=timeout)
        if self.up is not None:
            self.up.set_staging(recv_bytes=rx)
        self.down.set_staging(send_bytes=tx)
        self.eng.staging_bytes = rx + tx
        self.eng.build(agree)
        built = self.eng.built_info()
        built["declared"] = args.budget_gib is not None
        startup.follower_ready(self.down, self.up, built, timeout=timeout)
        self.mu = self.eng.multi
        log(f"stage {index} of {count} {self.where}: ready (pool {agree['cap']} positions, {agree['slots']} slots, "
            f"{len(built['graphs'])} round graphs, {built['free_after'] / 2**30:.2f} GiB free)")

    # -- the loop ------------------------------------------------------------------------------------------------
    def recv_ctl(self):
        mu = self.mu
        if mu.streams or mu.filling:
            with self.gate:                                   # streams live: keep the clocks up between rounds
                return self.down.recv(timeout=None, keep_raw=True)
        while True:                                           # idle: also notice the stage before going away
            ready = wait_any([self.down, self.up])
            if ready is self.down:
                return self.down.recv(timeout=None, keep_raw=True)
            f = self.up.recv(timeout=self.args.io_timeout)
            if f.op == proto.ERROR:
                raise LinkError(f"stage {f.hdr.get('stage')} failed in {f.hdr.get('where')}: {f.hdr.get('msg')}")
            raise proto.ProtocolError(f"an unasked {f.op} from the stage before while idle")

    def run(self) -> int:
        while True:
            f = self.recv_ctl()
            startup.forward_or_fail(f, self.up)              # upstream first; an ERROR exits 75 after forwarding
            if f.op == proto.QUIT:
                log(f"{self.where}: QUIT; sent {self.down.sent / 2**20:.1f} MiB down"
                    + (f", received {self.up.received / 2**20:.1f} MiB from the stage before" if self.up else ""))
                return 0
            handler = getattr(self, "on_" + f.op, None)
            if handler is None:
                raise proto.ProtocolError(f"control op {f.op} after startup")
            with torch.no_grad():                            # as MultiDecoder.admit / round (F/multi.py:758, 997)
                handler(f)

    def _digest(self, f) -> None:
        got = self.mu._shape_pp()
        if f.hdr.get("digest") != got:
            raise proto.ProtocolError(f"{f.op}: the leader's state digest {f.hdr.get('digest')}, this stage's {got} "
                                      "(the mirrored slots / extents / streams parted)")

    # -- handlers (spec 8.2) ---------------------------------------------------------------------------------------
    def on_ADMIT(self, f) -> None:
        h, mu = f.hdr, self.mu
        self._digest(f)
        if int(h["sid"]) != mu.next_id:
            raise proto.ProtocolError(f"ADMIT of stream {h['sid']}, this stage's next is {mu.next_id}")
        prompt = [int(x) for x in f.ints("prompt").tolist()]
        s = Stream(prompt, int(h["count"]), None, draft=False, stop_eos=bool(h["stop_eos"]))
        s.emit = lambda new: None                            # as F/multi.py:1273-1277
        s.keys = None
        mu._admit(s, int(h["index"]), [], None)
        s.pp = (len(prompt), 0)                              # (P, R) of its last round: none yet

    def on_FILL(self, f) -> None:
        h, mu = f.hdr, self.mu
        self._digest(f)
        s = next((x for x in mu.filling if x.sid == int(h["sid"])), None)
        if s is None:
            raise proto.ProtocolError(f"FILL of stream {h['sid']}, which is not filling here")
        if s.filled != int(h["at"]):
            raise proto.ProtocolError(f"FILL of stream {s.sid} at {h['at']}, this stage filled to {s.filled}")
        mu._fill_follower(s)

    def on_STEP(self, f) -> None:
        h, mu = f.hdr, self.mu
        rows = [tuple(int(x) for x in r) for r in h["rows"]]
        ids = [int(x) for x in f.ints("ids").tolist()]
        if sum(r[2] for r in rows) != len(ids):
            raise proto.ProtocolError(f"STEP of {sum(r[2] for r in rows)} rows with {len(ids)} ids")
        n = int(h["round"])
        if n != mu.runner.cur_round + 1:
            raise proto.ProtocolError(f"STEP {n}, this stage ran {mu.runner.cur_round}")
        by_slot = {s.st.index: s for s in mu.streams.values() if s.st is not None}
        windows, at = [], 0
        for slot, P, R in rows:
            s = by_slot.get(slot)
            if s is None:
                raise proto.ProtocolError(f"STEP row of slot {slot}: no stream decodes there")
            p0, r0 = s.pp
            if (r0 == 0 and P != p0) or (r0 > 0 and not p0 < P <= p0 + r0) or R < 1:
                raise proto.ProtocolError(f"stream {s.sid}: a round at {P} of {R} rows after one at {p0} of {r0}")
            s.pp = (P, R)
            toks = ids[at:at + R]
            at += R
            sc = s.st.sc
            sc.length = P
            sc.host.set(P, toks)                             # drops the rejected rows' ids (FO/ops.py:348-360)
            windows.append((s.st.index, s.base, s.size, P, toks, sc.host.view()))
        b = mu.runner.bucket(max(P + R for _, P, R in rows))
        if b != int(h["b"]):
            raise proto.ProtocolError(f"STEP bucket {h['b']}, this stage's {b}")
        self.steps += 1
        if self.fault == ("on-step", self.steps):
            raise RuntimeError(f"tfpp-ds41: injected fault at STEP {self.steps} (TFPP_FAULT)")
        mu.runner.cur_round = n
        mu.runner.forward(windows)                           # graphs or eager; R1 / R2 move the ACT

    def on_FINISH(self, f) -> None:
        self._digest(f)
        self.mu._finish([int(x) for x in f.hdr["sids"]], [])

    def on_DROP(self, f) -> None:
        self.mu._drop()


def run(args, tracer=None) -> int:
    """A follower until QUIT (0) or a broken chain (75)."""

    fo = None
    try:
        fo = Follower(args, tracer)
        return fo.run()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        where = fo.where if fo is not None else str(args.layers)
        log(f"FATAL {where}: {type(exc).__name__}: {exc}")
        if fo is not None:
            startup.report_error(fo.down, where, "run", exc)
            if fo.up is not None:
                try:
                    fo.up.close()
                except Exception:  # noqa: BLE001
                    pass
        if not isinstance(exc, (LinkError, proto.ProtocolError)):
            import traceback

            traceback.print_exc()
        sys.stdout.flush()
        return proto.FAIL_EXIT
