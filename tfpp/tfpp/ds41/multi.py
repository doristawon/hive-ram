"""The control plane (spec section 8): the fork's MultiDecoder on every stage, driven by the leader.

The leader runs MultiDecoder as the fork does (admit / _admit's placement, _fill, round / _round with the drafts and
the sampling, finish, drop: F/multi.py:758-1259) under the fork's Scheduler (C/scheduler.py:49-181); MultiDecoder
already sends each of its steps to "the followers" before running it (``self._send(op)``: F/multi.py:493-496, 782,
1007, 1013, 1204, 1247), and ``PipeOps`` is that link: ADMIT, FILL, FINISH and DROP go up the chain as frames; the
round's STEP is sent by the round runner itself (it carries the windows: decode.StageRoundRunner.forward). A follower
mirrors the state by running the same ``_admit`` / ``_fill`` steps / ``_finish`` / ``_drop`` (follower.py); a digest
of the mirrored state rides on every op and is checked before the op is applied (spec 8.3).

``PipeWatch`` is the fork's Watchdog's place (``watch.busy`` / ``watch.broken``, F/multi.py:498-512): between steps it
watches the leader's upstream link; anything readable while no ACT is owed is the chain breaking (a follower's ERROR,
EOF): fail-stop, exit 75. ``StageMulti._broken`` keeps a leader-local failure local (the Scheduler drops the live
streams and PipeOps drains the owed ACT frames and sends DROP: the chain goes on, spec 8.4).
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

from tensorfold.families.deepseek_v41.cuda.multi import MultiDecoder

from . import patch, proto
from .decode import StageRoundRunner
from .wire import LinkError

ProtocolError = proto.ProtocolError


def log(msg: str) -> None:
    print(f"[tfpp-ds41] {msg}", flush=True)


def fault_point(text: str | None, role: str):
    """TFPP_FAULT (tests, spec 8.4): leader:after-step:N, leader:after-fill:N, follower:on-step:N."""

    text = (text or "").strip()
    if not text:
        return None
    try:
        who, where, n = text.split(":")
        n = int(n)
    except ValueError:
        raise ValueError(f"TFPP_FAULT={text!r}: leader:after-step:N, leader:after-fill:N or follower:on-step:N") \
            from None
    if (who, where) not in (("leader", "after-step"), ("leader", "after-fill"), ("follower", "on-step")):
        raise ValueError(f"TFPP_FAULT={text!r}: leader:after-step:N, leader:after-fill:N or follower:on-step:N")
    return (where, n) if who == role else None


class StageMulti(MultiDecoder):
    _multi_init = patch.patched(MultiDecoder.__init__, patch.group_ids("multi_init", False), "MultiDecoder.__init__")

    def __init__(self, engine, *, slots: int, cap: int, plan, mv, io, gate, role: str) -> None:
        self.plan, self.mv, self.io, self.gate, self.role = plan, mv, io, gate, role
        self._multi_init(engine, slots=slots, cap=cap)                          # F/multi.py:377-440 (MU1)

    def _tfpp_runner(self, m, pool, graphs: bool = True, graph_pool=None):
        return StageRoundRunner(m, pool, graphs=graphs, graph_pool=graph_pool, plan=self.plan, mv=self.mv,
                                io=self.io, gate=self.gate, announce=self.role == "leader" and self.io.up is not None)

    def _shape_pp(self) -> str:
        """The mirrored state (spec 8.3): _shape (F/multi.py:535-539) without what only the leader knows or knows a
        round sooner (lengths, replies, done) and without kept prompts (KEEP=0)."""

        return proto.digest([self.next_id, list(self.free), [list(g) for g in self.extents.gaps],
                             [[s.sid, s.st.index, s.base, s.size] for s in self.streams.values()],
                             [[s.sid, s.st.index, s.base, s.size, s.filled] for s in self.filling]])

    def _broken(self, exc: BaseException) -> None:
        """F/multi.py:502-508: a broken link ends the chain; any other failure stays this request's (the Scheduler
        drops the live streams, PipeOps sends DROP after the owed frames)."""

        if self.watch is not None and isinstance(exc, (LinkError, ProtocolError)):
            self.watch.fatal(exc)

    def _drafts(self, tokens, q0, slots):
        """A drafter graph captured on first use takes the KeepWarm gate (spec 11.1)."""

        if len(tokens) not in self.drafters:
            with self.gate.paused():
                return super()._drafts(tokens, q0, slots)
        return super()._drafts(tokens, q0, slots)

    def _fill_follower(self, s) -> None:
        """A follower's FILL: the next chunk of the stream's prompt; after its last one the stream decodes (the
        leader samples its first token: F/multi.py:898-916 without the sampling)."""

        try:
            s.filled = next(s.steps)
        except StopIteration:
            s.filled = len(s.prompt)
            self.filling.remove(s)
            s.steps = None
            self.streams[s.sid] = s


class PipeOps:
    """MultiDecoder's link on the leader (``multi.link``): the fork's ops as frames up the chain (spec 8.1)."""

    def __init__(self, mu: StageMulti, link, *, timeout: float) -> None:
        self.mu, self.link = mu, link
        self.timeout = float(timeout)
        self.lock = threading.Lock()
        self.outstanding = 0                    # ACT frames asked for (FILL, STEP) and not yet read
        self.steps = self.fills = 0
        self.fault = fault_point(os.environ.get("TFPP_FAULT"), "leader")
        self.armed = False                      # the fault counts from here (serve arms it after the self-check)
        self.fault_steps = self.fault_fills = 0

    def got_act(self) -> None:
        with self.lock:
            if self.outstanding <= 0:
                raise ProtocolError("an ACT nobody asked for")
            self.outstanding -= 1

    def _ctl(self, op: str, fields: dict, parts=()) -> None:
        self.link.send(op, fields, parts)

    def send(self, op: list) -> None:
        kind = op[0]
        mu = self.mu
        try:
            if kind == "admit":
                _, prompt, count, sampling, draft, stop_eos, index, positions, reuse = op
                if positions or reuse:
                    raise ValueError("tfpp-ds41 v1: no image spans, no kept prompts")
                self._ctl(proto.ADMIT, {"sid": int(mu.next_id), "index": int(index), "count": int(count),
                                        "stop_eos": bool(stop_eos), "draft": bool(draft), "digest": mu._shape_pp()},
                          [("prompt", np.asarray(list(prompt), dtype=np.int32))])
            elif kind == "fill":
                sid = int(op[1])
                s = next(f for f in mu.filling if f.sid == sid)
                with self.lock:
                    self.outstanding += 1
                    self.fills += 1
                self._ctl(proto.FILL, {"sid": sid, "at": int(s.filled), "digest": mu._shape_pp()})
                if self.armed and self.fault is not None:
                    self.fault_fills += 1
                    if self.fault == ("after-fill", self.fault_fills):
                        raise RuntimeError(f"tfpp-ds41: injected fault after FILL {self.fault_fills} (TFPP_FAULT)")
            elif kind == "round":
                return                                     # the runner sends STEP with the windows
            elif kind == "finish":
                _, sids, drops, *_ = op
                if drops:
                    raise ValueError("tfpp-ds41 v1 keeps no prompts (TF_DS_KEEP=0): no drops")
                self._ctl(proto.FINISH, {"sids": [int(x) for x in sids], "digest": mu._shape_pp()})
            elif kind == "drop":
                self.drain()
                self._ctl(proto.DROP, {})
            else:
                raise ValueError(f"MultiDecoder op {kind!r}")
        except (LinkError, ProtocolError) as exc:
            if mu.watch is not None:
                mu.watch.fatal(exc)
            raise

    def step(self, rr, windows: list) -> None:
        """STEP up the chain before the leader's own round (spec 9.3): each window's (slot, P, R) and the ids."""

        rows = [[int(slot), int(p0), len(toks)] for slot, _, _, p0, toks, _ in windows]
        ids = [int(t) for *_, toks, _ in windows for t in toks]
        b = rr.bucket(max(p0 + len(toks) for _, _, _, p0, toks, _ in windows))
        with self.lock:
            self.steps += 1
            self.outstanding += 1
            n = self.steps
        rr.cur_round = n
        try:
            self._ctl(proto.STEP, {"round": n, "b": int(b), "rows": rows}, [("ids", np.asarray(ids, dtype=np.int32))])
        except (LinkError, ProtocolError) as exc:
            if self.mu.watch is not None:
                self.mu.watch.fatal(exc)
            raise
        if self.armed and self.fault is not None:
            self.fault_steps += 1
            if self.fault == ("after-step", self.fault_steps):
                raise RuntimeError(f"tfpp-ds41: injected fault after STEP {self.fault_steps} (TFPP_FAULT)")

    def drain(self) -> None:
        """Read and drop the ACT frames still owed (a leader-local failure left them in flight)."""

        while True:
            with self.lock:
                if self.outstanding <= 0:
                    return
            f = self.link.recv(timeout=self.timeout)
            if f.op == proto.ERROR:
                raise LinkError(f"stage {f.hdr.get('stage')} failed in {f.hdr.get('where')}: {f.hdr.get('msg')}")
            if f.op != proto.ACT:
                raise ProtocolError(f"draining: expected ACT, got {f.op}")
            with self.lock:
                self.outstanding -= 1

    def quit(self) -> None:
        self._ctl(proto.QUIT, {})


class PipeWatch:
    """The leader's watch on its upstream link between steps (spec 8.4): ``busy`` / ``broken`` as the fork's Watchdog
    (MultiDecoder._step / _alive read and write them)."""

    def __init__(self, ops: PipeOps, link) -> None:
        self.ops, self.link = ops, link
        self.busy: float | None = None
        self.broken: str | None = None
        self.closing = False
        self.thread = threading.Thread(target=self._run, name="tfpp-ds41-watch", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.broken is None and not self.closing:
            if self.ops.outstanding > 0:
                time.sleep(0.05)
                continue
            if not self.link.readable(0.5):
                continue
            with self.ops.lock:
                if self.ops.outstanding > 0 or self.closing or self.broken is not None or not self.link.readable(0.0):
                    continue
                try:
                    f = self.link.recv(timeout=self.ops.timeout)
                    if f.op == proto.ERROR:
                        exc: BaseException = LinkError(f"between steps: stage {f.hdr.get('stage')} failed in "
                                                       f"{f.hdr.get('where')}: {f.hdr.get('msg')}")
                    else:
                        exc = ProtocolError(f"between steps: an unasked {f.op} from the stage before")
                except Exception as err:  # noqa: BLE001  (a closed link: LinkError)
                    exc = err
            if not self.closing:
                self.fatal(exc)
            return

    def fatal(self, exc: BaseException) -> None:
        """Fail-stop: requests in flight fail, the process exits 75 a moment later (every stage restarts, in any
        order; no stage rejoins a chain)."""

        if self.broken is not None:
            return
        self.broken = f"{type(exc).__name__}: {exc}"
        log(f"FATAL: the chain broke: {self.broken}; exiting {proto.FAIL_EXIT}")
        threading.Timer(1.0, os._exit, (proto.FAIL_EXIT,)).start()

    def abort(self, why: str) -> None:                     # the fork's Watchdog interface
        self.fatal(RuntimeError(why))
