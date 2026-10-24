"""A follower stage (spec section 10): one thread, the leader's control ops in order, MIA's own functions on its
layers. It never samples, drafts or compiles a grammar; it keeps the same caches and kept states as the leader
because the leader tells it every step that changes them (PREFILL, STEP + KEEP, SNAP / SAVE / LOAD / DROP).

Control frames arrive from the next stage (towards the leader) and go to the previous one (towards stage 0) before
they are handled, so stage 0 starts as early as it can; ACT frames come from the previous stage and go to the next.
"""

from __future__ import annotations

import time

import torch

from tensorfold.families.glm5_next.cuda import decode as D, forward as fw

from . import capacity, part, proto, startup
from .feed import LocalFeed
from .load import load_range
from .wire import ActIO, LinkError, wait_any


def log(msg: str) -> None:
    print(f"[tfpp-glm] {msg}", flush=True)


class Follower:
    v2 = None                         # tfpp-glm v2: the stage v2follower.start built when GO says P > 1

    def __init__(self, args, tracer=None) -> None:
        from tensorfold.cuda import prompt_precision

        from tfpp.warm import KeepWarm

        self.args = args
        a, b = args.layers
        self.layers = (a, b)
        self.where = f"[{a},{b})"
        self.tracer = tracer
        torch.cuda.set_device(0)
        self.down, self.up = startup.follower_connect((a, b), args.down, args.listen,
                                                      connect_timeout=args.connect_timeout,
                                                      io_timeout=args.io_timeout)
        try:
            self._start(prompt_precision, KeepWarm)
        except BaseException as exc:
            startup.report_error(self.down, self.where, "startup", exc)
            raise

    def _start(self, prompt_precision, KeepWarm) -> None:
        args = self.args
        a, b = self.layers
        timeout = args.connect_timeout
        own = startup.stage_info((a, b), "follower", args.model)
        go = startup.follower_hello(self.down, self.up, own, timeout=timeout)
        if int(go["P"]) > 1:
            from . import v2follower
            self.v2 = v2follower.start(self, go, prompt_precision, KeepWarm)
            return
        index, count = startup.stage_of(go, (a, b))
        if index == count - 1:
            raise proto.ProtocolError("a follower cannot be the chain's last stage")
        taps = tuple(int(t) for t in go["taps"])
        ranges = [tuple(r) for r in go["stages"]]
        cin, cout = proto.tap_routes(taps, ranges)[index]
        prompt_precision.set_fp8(bool(go["fp8"]))           # before any weight loads (TF/prompt_precision.py:17)
        self.warm = KeepWarm()
        stage = part.Stage(index=index, count=count, layers=(a, b), first=a == 0, last=False, taps=taps,
                           carry_in=cin, carry_out=cout)
        self.io = ActIO(stage, self.up, self.down, warm=self.warm, timeout=args.io_timeout, tracer=self.tracer)
        stage.io = self.io
        self.stage = stage
        part.install_common(stage)
        log(f"stage {index} of {count} {self.where}: loading (taps in {list(cin)}, out {list(cout)})")
        t0 = time.perf_counter()
        self.w = load_range(args.model, a, b, is_first=a == 0, is_last=False, mtp=False, drafter=False,
                            prefetch_next=not args.no_prefetch_next)
        log(f"stage {index} {self.where}: layers loaded in {time.perf_counter() - t0:.1f}s "
            f"({self.w.nbytes() / 2**30:.2f} GiB)")
        prefill_rows = int(go["prefill_rows"])
        staging = capacity.staging_bytes(prefill_rows, len(cin), len(cout), first=a == 0, last=False)
        cap = capacity.measure_budget(reserve_gib=args.reserve_gib, budget_gib=args.budget_gib, staging=staging)
        cap["layers"] = [a, b]
        agree = startup.follower_cap(self.down, self.up, cap, timeout=timeout)
        self.agree = agree
        if self.up is not None:
            self.up.set_staging(recv_bytes=proto.link_staging(prefill_rows, len(cin)))
        self.down.set_staging(send_bytes=proto.link_staging(prefill_rows, len(cout)))
        part.scrubbing(True)
        try:
            self.e = D.Engine(self.w, capacity=agree["pool_rows"] or agree["capacity"], max_rows=agree["max_rows"],
                              prefill_rows=agree["prefill_rows"], graphs=agree["graphs"],
                              graph_rows=tuple(agree["graph_rows"]), long_context=agree["long_context"],
                              taps=tuple(agree["taps"]), grid=agree["grid"], split=None, mtp_rows=agree["max_rows"],
                              streams=agree["parallel"], pool_rows=agree["pool_rows"], kv=agree["kv"])
        finally:
            part.scrubbing(False)
        self.snaps: dict[int, D.Snapshot] = {}
        self.pending: int | None = None
        self.busy = False
        startup.follower_ready(self.down, self.up, startup.built_info((a, b), self.e), timeout=timeout)
        log(f"stage {index} {self.where}: ready (window {agree['window']}, cache slots {agree['capacity']}, "
            f"{'long contexts' if agree['long_context'] else 'dense'})")

    # -- the loop ------------------------------------------------------------------------------------------------
    def recv_ctl(self):
        if self.busy:
            with self.warm:
                return self.down.recv(timeout=self.args.io_timeout, keep_raw=True)
        while True:                       # idle: also notice the stage before this one going away
            ready = wait_any([self.down, self.up])
            if ready is self.down:
                return self.down.recv(timeout=None, keep_raw=True)
            f = self.up.recv(timeout=self.args.io_timeout)          # only an ERROR (or a closed link) is expected
            if f.op == proto.ERROR:
                raise LinkError(f"stage {f.hdr.get('stage')} failed in {f.hdr.get('where')}: {f.hdr.get('msg')}")
            raise proto.ProtocolError(f"an unasked {f.op} from the stage before while idle")

    def run(self) -> int:
        if self.v2 is not None:
            return self.v2.run()
        while True:
            f = self.recv_ctl()
            if self.up is not None:
                self.up.send_raw(f.raw, f.payload)               # upstream first: stage 0 starts as early as it can
            handler = getattr(self, "on_" + f.op, None)
            if handler is None:
                raise proto.ProtocolError(f"control op {f.op} after startup")
            if handler(f):
                return 0

    # -- handlers (spec 8.1) ---------------------------------------------------------------------------------------
    def on_PREFILL(self, f) -> None:
        h = f.hdr
        self.busy = True
        if self.pending is not None:
            raise proto.ProtocolError("PREFILL with an uncommitted window")
        e = self.e
        if h["mode"] == "fresh":
            e.reset()
        elif h["mode"] == "resume":
            D.restore(e, self._snap(h["sid"]), None)               # G/decode.py:533-548
        else:
            raise proto.ProtocolError(f"PREFILL mode {h['mode']!r}")
        begin, end, rows = int(h["begin"]), int(h["end"]), int(h["rows"])
        if e.st.pos != begin:
            raise proto.ProtocolError(f"PREFILL from {begin}, this stage is at {e.st.pos}")
        if rows != e.prefill_rows:
            raise proto.ProtocolError(f"PREFILL in chunks of {rows}, this stage's are {e.prefill_rows}")
        ids = f.ints("ids").tolist()
        if len(ids) != end - begin:
            raise proto.ProtocolError(f"PREFILL [{begin}, {end}) with {len(ids)} ids")
        feed = None
        if h.get("img_pos") and self.stage.first:
            feed = LocalFeed(h["img_pos"], f.tensor("img"))      # host rows; each chunk uploads its own (feed.py)
        prompt = proto.PromptView(begin, ids)
        for start in range(begin, end, rows):
            R = min(rows, end - start)
            # the original: stage -> compute (_compute_pp reads / sends the ACT) -> commit(R, R) (G/decode.py:616-639)
            D.prefill_chunk(e, prompt, start, R, drafter=None, feed=feed, mtp=False, head=False)

    def on_STEP(self, f) -> None:
        h = f.hdr
        if self.pending is not None:
            raise proto.ProtocolError("STEP with an uncommitted window")
        e = self.e
        if e.st.pos != h["pos"]:
            raise proto.ProtocolError(f"STEP at {h['pos']}, this stage is at {e.st.pos}")
        ids = f.ints("ids").tolist()
        R = len(ids)
        if R != h["R"]:
            raise proto.ProtocolError(f"STEP of {h['R']} rows with {R} ids")
        self.busy = True
        S = self.stage
        if not S.first:
            self.io.recv_act(e.buf, R, e.st, "step")
        e.forward(ids)                                   # decode.Engine.forward: a graph or the eager path
        self.io.send_act(e.buf, R, e.st, "step")
        self.pending = R

    def on_KEEP(self, f) -> None:
        if self.pending is None:
            raise proto.ProtocolError("KEEP without a window")
        e = self.e
        keep = int(f.hdr["pos"]) - e.st.pos
        if not 1 <= keep <= self.pending:
            raise proto.ProtocolError(f"KEEP of {keep} rows of a {self.pending}-row window")
        fw.commit(e.w, e.st, e.buf, self.pending, keep)  # G/forward.py:1131-1143
        self.pending = None

    def on_SNAP(self, f) -> None:
        h = f.hdr
        if self.pending is not None:
            raise proto.ProtocolError("SNAP with an uncommitted window")
        n = int(h["n"])
        if self.e.st.pos != n:
            raise proto.ProtocolError(f"SNAP of {n} tokens, this stage is at {self.e.st.pos}")
        self.snaps[int(h["sid"])] = D.take_snapshot(self.e, [0] * n, None, mtp=False, drafter=None)

    def on_SAVE(self, f) -> None:
        D.save_rows(self.e, self._snap(f.hdr["sid"]), None)

    def on_LOAD(self, f) -> None:
        snap = self._snap(f.hdr["sid"])
        D.load_rows(self.e, snap, None)
        snap.rows, snap.nbytes = None, 0                 # live again (G/engine.py:1366-1367)

    def on_DROP(self, f) -> None:
        self._forget(int(f.hdr["sid"]))

    def on_END(self, f) -> None:
        h = f.hdr
        if h.get("abort"):
            if self.pending is not None:
                log(f"request {h.get('req')} aborted: the uncommitted {self.pending}-row window is dropped")
            self.pending = None
        else:
            if self.pending is not None:
                raise proto.ProtocolError("END with an uncommitted window")
            if self.e.st.pos != h["pos"]:
                raise proto.ProtocolError(f"END at {h['pos']}, this stage is at {self.e.st.pos}")
        live = {int(s) for s in h.get("live", [])}
        stale = [s for s in self.snaps if s not in live]
        if stale and not h.get("abort"):
            log(f"request {h.get('req')}: {len(stale)} kept states the leader no longer holds (dropped)")
        for s in stale:
            self._forget(s)
        self.busy = False

    def on_QUIT(self, f) -> bool:
        log(f"{self.where}: QUIT; sent {self.down.sent / 2**20:.1f} MiB down"
            + (f", received {self.up.received / 2**20:.1f} MiB from the stage before" if self.up else ""))
        return True

    def on_ERROR(self, f) -> None:
        h = f.hdr
        raise LinkError(f"the leader stopped the chain: {h.get('where')}: {h.get('msg')}")

    # -- kept states -----------------------------------------------------------------------------------------------
    def _snap(self, sid):
        try:
            return self.snaps[int(sid)]
        except (KeyError, TypeError, ValueError):
            raise proto.ProtocolError(f"no kept state {sid!r} on this stage") from None

    def _forget(self, sid: int) -> None:
        snap = self.snaps.pop(int(sid), None)
        if snap is None:
            return
        freed = D.snapshot_bytes(snap)
        snap.rows = None
        del snap
        if freed > 64 << 20:
            torch.cuda.empty_cache()


def run(args, tracer=None) -> int:
    """A follower until QUIT (0) or a broken chain (75)."""

    f = None
    try:
        f = Follower(args, tracer)
        return f.run()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        where = f.where if f is not None else str(args.layers)
        log(f"FATAL {where}: {type(exc).__name__}: {exc}")
        if f is not None:
            startup.report_error(f.down, where, "run", exc)
            if f.up is not None:
                try:
                    f.up.close()
                except Exception:  # noqa: BLE001
                    pass
        if not isinstance(exc, (LinkError, proto.ProtocolError)):
            import traceback

            traceback.print_exc()
        return proto.FAIL_EXIT
