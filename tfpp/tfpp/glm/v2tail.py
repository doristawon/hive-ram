"""tfpp-glm v2 (--parallel 2..8): the leader, the last stage of the chain, MIA's rank 0 (spec v2 section 7).

The leader runs MIA's ``GlmScheduler`` / ``MultiDecoder`` unchanged on its own layer range, with ``comm = None`` (so the
samplers and the forward take their single-rank branches). What reaches the followers is the iteration's message (MULTI:
MIA's sealed op list, one frame a ``_flush``) and, for a decode round, the window's token ids (WINDOWS) and the rows it
keeps (KEEPS); the stage before sends the rows back as ACT frames (a lane's chunk, a multi-prompt chunk, a window).

- ``PipeMultiEngine._share``: ``MultiDecoder._flush`` -> MULTI up the chain (G/multi.py:625-636).
- ``PipeBatchedVerify``: ``forward`` = WINDOWS up, the window's ACT back, then the original ``BatchedVerify.forward``;
  ``commit`` = KEEPS up, then the original commit (the followers commit while the leader does).
- ``PipeMultiDecoder``: ``round`` / ``finish`` hold ``run_lock`` (the idle watch keeps off the link), any exception in an
  iteration is fatal for the whole chain (MIA latches ``broken`` and every later admit fails: G/multi.py:615-617).
- the prompt ACT: ``v2part._compute_pp_multi`` reads a chunk's rows inside ``prefill_chunk`` / ``prefill_pieces``.

One round is in flight at a time and one stage works at a time (v2.0): the gain over v1 is the experts' weights read once
for all streams' rows. Nothing in v1's modules changes; P=1 never imports this module.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.families.glm5_next.cuda import decode as D, engine as E, forward as fw, multi as M
from tensorfold.families.glm5_next.cuda.verify import BatchedVerify

from . import capacity, feed, part, proto, startup, v2capacity, v2part, v2proto, v2startup, v2wire
from .load import load_range
from .tail import GIB, LeaderLink, PipeGlmEngine, log, read_taps, vision_tower_bytes
from .v2bench import Landings, Lockstep, Req
from .v2proto import Fault, RoundStats, log_rows
from .wire import ActIO, Link


class GuardedEngine(D.Engine):
    """``decode.Engine`` for a --parallel stage: its single-stream ``forward`` (the CUDA graphs it would replay are not
    captured: spec v2 F-14) must not run inside a chain, where it would compute a non-first stage's window from x it
    never received."""

    tail = None

    def forward(self, tokens):
        T = self.tail
        if T is not None and T.link is not None:
            raise RuntimeError("tfpp-glm v2: Engine.forward is the single-stream path; a --parallel chain runs "
                               "batched windows only")
        return super().forward(tokens)


def check_parallel(parallel: int, drafter, serial_only: bool) -> int:
    """GlmEngine.__init__'s --parallel checks (G/engine.py:521-533)."""

    parallel = int(parallel)
    if parallel > 1 and drafter is None and not serial_only:
        raise ValueError(f"--parallel {parallel}: concurrent requests draft with DFlash2 only: give the leader the "
                         "DFlash2 draft model (--drafter), or pass --no-drafts to decode them together without drafts")
    if not 1 <= parallel <= E.PARALLEL_MOST:
        raise ValueError(f"--parallel: 1 to {E.PARALLEL_MOST} requests at once for GLM-5.3-Flash, not {parallel}")
    return parallel


class V2Link:
    """The leader's control frames (the worker thread only; ``Link.send`` is locked anyway)."""

    def __init__(self, link: Link) -> None:
        self.link = link

    def multi(self, values, parts) -> None:
        self.link.send(v2proto.MULTI, v2proto.multi_fields(0, values), parts)

    def windows(self, rid: int, segs, ids) -> None:
        self.link.send(v2proto.WINDOWS, v2proto.windows_fields(0, rid, segs),
                       [("ids", np.asarray(ids, dtype=np.int32))])

    def keeps(self, rid: int, keeps, end) -> None:
        self.link.send(v2proto.KEEPS, v2proto.keeps_fields(0, rid, keeps, end))


class PipeBatchedVerify(BatchedVerify):
    """MIA's batched verify seam with the chain around it (the leader's). N=1 (``eng.link`` None, or while the graphs
    are captured: ``eng`` unset) it is MIA's own."""

    eng = decoder = None
    rid = None
    segments: list

    @torch.no_grad()
    def forward(self, segments):
        g = self.eng
        if g is None or g.link is None:
            return super().forward(segments)                     # N=1: MIA's (G/verify.py:216-248)
        t0 = time.perf_counter()
        for s in segments:
            fw.check_room(self.w, s.st, len(s.tokens))           # a refusal sends nothing
        rid = (self.decoder.sent - 1) % M.SEAL_MOD               # the MULTI that carried this ROUND
        R = sum(len(s.tokens) for s in segments)
        segs = [[int(s.st.slot), int(s.st.pos), len(s.tokens)] for s in segments]
        g.v2link.windows(rid, segs, [int(t) for s in segments for t in s.tokens])
        g.v2fault.hit("round.windows")
        w0 = g.io.recv_ms
        v2wire.recv_rows(g.io, self.b, R, v2proto.act_round(0, rid, R))   # x + carried taps, inside KeepWarm
        g.v2fault.hit("round.act")
        self.rid = rid
        out = super().forward(segments)                          # stages ids, tables; graph replay reads b.x
        if g.v2stats is not None:
            g.v2stats.add(R, time.perf_counter() - t0, (g.io.recv_ms - w0) / 1e3)
        return out

    @torch.no_grad()
    def commit(self, segments, keeps):
        g = self.eng
        if g is not None and g.link is not None:
            if [id(s) for s in segments] != [id(s) for s in self.segments] or any(
                    not 1 <= int(k) <= len(s.tokens) for s, k in zip(segments, keeps)):
                raise ValueError("commit takes the last forward's segments, keeps in 1..rows")  # G/verify.py:277-281
            g.v2link.keeps(self.rid, [int(k) for k in keeps],
                           [int(s.st.pos) + int(k) for s, k in zip(segments, keeps)])
            g.v2fault.hit("round.keeps")
        super().commit(segments, keeps)                          # G/verify.py:269-293, while the followers commit


class PipeMultiDecoder(M.MultiDecoder):
    """MIA's MultiDecoder on the leader, its I/O under ``run_lock`` and its failures the chain's."""

    def __init__(self, engine, streams, *, verify, draft_graphs: bool | None = None) -> None:
        self.outgoing = {}                 # sid -> image rows of an ADMIT not yet sent (stage 0 injects them)
        self.clear_due = None              # an Event: evict every kept prompt at the next idle iteration
        self.cleared = None                # that Event once the evictions went out: set when the iteration's finish
                                           # (its IDLE message) is done too, so the warm-up returns at rest
        # MultiDrafter + its graphs (G/multi.py:520-527); --no-graphs: its block passes run eagerly (the same bits)
        super().__init__(engine, streams, verify=verify, draft_graphs=draft_graphs)

    def round(self):
        g = self.g
        with g.run_lock:                                         # F-4: tail._watch keeps off the link
            if g.closing or (g.broken and g.link is not None):
                time.sleep(0.05)
                return []
            ev = None
            if self.clear_due is not None and not self.lanes:
                ev, self.clear_due = self.clear_due, None
                for c in list(self.kept):
                    self._evict(c, spill=False)                  # EVICT on every stage (G/multi.py:1160-1164)
            try:
                return super().round()                           # G/multi.py:1661-1676
            except BaseException as exc:
                if g.link is not None:
                    g._fatal(exc)                                # F-7: the whole chain restarts
                raise
            finally:
                if ev is not None:
                    self.cleared = ev                            # set by this iteration's finish (G/multi.py:2686)

    def _iterate(self):
        """MIA reaps a filling lane whose client has gone only with a fill budget (G/multi.py:1696-1705); tfpp-glm runs
        with ``TF_GLM_FILL_BUDGET_MS=0`` (the sliced fills compute a non-first stage's embedding wrong: spec F-2), so
        without this a 100K-token prompt whose client left would still be filled chunk by chunk. The same reap, no I/O:
        ``finish`` then sends FINISH (cancelled) and every stage runs ``_finish`` (``partial`` is always None here)."""

        if not self.fill_budget:
            cancelled = []
            for lane in self.lanes.values():
                poll = getattr(lane.s.emit, "cancelled", None)
                if not lane.decoding and not lane.s.done and poll is not None and poll():
                    lane.s.done = True
                    lane.s.started = lane.s.finished = time.perf_counter()
                    cancelled.append(lane.s)
            if cancelled:
                return cancelled
        return super()._iterate()

    def finish(self, done):
        g = self.g
        try:
            with g.run_lock:
                if g.closing or (g.broken and g.link is not None):  # nothing after QUIT / a fatal error
                    return
                try:
                    super().finish(done)                             # G/multi.py:2231-2248
                except BaseException as exc:
                    if g.link is not None:
                        g._fatal(exc)
                    raise
        finally:
            ev, self.cleared = getattr(self, "cleared", None), None
            if ev is not None:
                ev.set()

    def _admitted(self, a, s, feed_, constraint, spec=""):
        super()._admitted(a, s, feed_, constraint, spec)         # G/multi.py:951-1022
        if self.rank == 0 and self.g.link is not None:
            if feed_ is not None and a["positions"]:
                self.outgoing[a["sid"]] = feed_.rows             # PipeVisionFeed's rows (G/multi.py:903, 924)
            self.g.v2fault.hit("admit.after")


def send_multi(eng, values) -> None:
    """One sealed message up the chain: ``ints`` and, for every ADMIT that brings pictures, their feature rows."""

    ops = M.MultiDecoder.parse(values[:-2])                      # G/multi.py:764-780
    parts = [("ints", np.asarray(values, dtype=np.int32))]
    for op, p in ops:
        if op == M.ADMIT and p[0] in eng.multi.outgoing:         # ADMIT's payload starts with the sid (:883)
            parts.append((v2proto.img_part(p[0]), eng.multi.outgoing.pop(p[0]).to(torch.bfloat16).contiguous()))
    eng.v2link.multi(values, parts)
    if any(op in (M.FILL, M.MFILL) for op, _ in ops):
        eng.v2fault.hit("fill.sent")


def install_stream_tags(eng) -> None:
    """Every request that enters the scheduler's queue gets a tag (``eng.request.trace_tag`` of the submitting thread, else
    ``req<n>``) in its ``glm`` dict, which a replay carries along (G/multi.py:2642, 2651); the harness' clients are told
    (``eng.v2landed``) when theirs is in the queue, and ``eng.v2cancel[tag] = K`` makes a stream end deterministically
    on the worker once it holds K tokens (the lockstep schedules' client-side stops)."""

    sch = eng.scheduler
    eng.v2landed = Landings()
    eng.v2cancel = {}
    counter = itertools.count(1)
    put = sch.waiting.put
    worker = sch.thread

    def tagged_put(item, block=True, timeout=None):
        stream = item[0]
        glm = getattr(stream, "glm", None)              # (a queued _Flush item has none)
        if glm is not None:
            if "tfpp_tag" not in glm:
                glm["tfpp_tag"] = getattr(eng.request, "trace_tag", None) or f"req{next(counter)}"
            k = eng.v2cancel.get(glm["tfpp_tag"])
            if k and not getattr(stream, "_tfpp_capped", False):
                # (also a replay: a give-back or a background yield puts a fresh Stream with the same glm,
                # G/multi.py:2642, 2651, and its ``out`` starts again from nothing)
                take = stream.take

                def capped(new, eos=(), _take=take, _stream=stream, _k=int(k)):
                    _take(new, eos)
                    if len(_stream.out) >= _k and not _stream.done:
                        _stream.done = True
                        _stream.finished = time.perf_counter()

                stream.take = capped
                stream._tfpp_capped = True
        result = put(item, block, timeout)
        if glm is not None and threading.current_thread() is not worker:
            eng.v2landed.signal(glm["tfpp_tag"])        # (a no-op unless a harness registered the tag)
        return result

    sch.waiting.put = tagged_put


def build_multi(eng, taps, graphs: bool) -> None:
    """The leader's verify seam (its graphs captured before any admission: G/verify.py:121-148), decoder and scheduler
    (its worker thread starts now and waits for a request)."""

    pv = PipeBatchedVerify(eng.e, taps=tuple(taps), rows=eng.multi_window)       # F-8: AGREE's taps
    pv.eng = eng
    if graphs and os.environ.get("TF_GLM_MULTI_GRAPHS", "1") != "0":
        part.scrubbing(True)
        try:
            pv.capture()                                                         # F-11: before any admission
        finally:
            part.scrubbing(False)
    eng.v2link = V2Link(eng.link.link) if eng.link is not None else None
    eng.multi = PipeMultiDecoder(eng, eng.parallel, verify=pv, draft_graphs=None if graphs else False)
    pv.decoder = eng.multi
    eng.multi.model_dir = eng.model_dir                                          # G/engine.py:876
    eng.scheduler = M.GlmScheduler(eng.multi, max_streams=eng.parallel)          # its worker starts now
    install_stream_tags(eng)
    eng.concurrent = True
    if os.environ.get("TFPP_GLM_TIME_ROWS") == "1" and pv.graphs:
        if not eng.stage.first:
            pv.b.x.normal_()                      # as the followers do: rows that route like text
        log_rows("leader", pv.time_rows())                                       # G/verify.py:152-179


class PipeMultiEngine(PipeGlmEngine):
    """The leader of a --parallel chain. ``__init__`` is ``PipeGlmEngine.__init__`` (T/tail.py:157-416) with the steps
    marked v2 changed: the chain's pool and kept-state count (v2capacity), a plain engine without its own graphs, the
    batched verify seam, MIA's decoder and scheduler, the lockstep warm-up. It inherits ``_watch`` / ``_fatal`` /
    ``quit`` / ``generate`` / ``_effective`` unchanged."""

    def __init__(self, model_dir, *, layers: tuple[int, int], truncate: int | None = None, link: Link | None = None,
                 policy: str = E.DEFAULT_POLICY, drafter=None, context: int | None = None,
                 context_explicit: bool | None = None, serial_only: bool = False, prefill_rows: int | None = None,
                 vision: bool = False, vision_urls: bool = False, parallel: int = 4, graphs: bool = True,
                 tap_layers=None, reserve_gib: float = 3.0, budget_gib: float | None = None,
                 io_timeout: float = 600.0, startup_timeout: float = 3600.0, pure: bool = False,
                 prefetch_next: bool = True, warmup: bool = True, pool_rows: int | None = None,
                 kept_entries: int | None = None) -> None:
        from tensorfold.cuda import prompt_precision
        from tensorfold.cuda.capacity import config as read_config
        from tensorfold.cuda.geometry import PREFILL_ROWS
        from tensorfold.families.glm5_next.cuda import LATENT, tp as tp_mod
        from tensorfold.families.glm5_next.cuda.copy_drafts import CopySettings
        from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings
        from tensorfold.families.glm5_next.cuda.kv8 import kv_kind
        from tensorfold.families.glm5_next.cuda.weights import Config

        from tfpp.warm import KeepWarm

        a, b = (int(x) for x in layers)
        model_dir = Path(model_dir)
        drafter = Path(drafter) if drafter else None
        self.link: LeaderLink | None = None
        self.io: ActIO | None = None
        self.broken: str | None = None
        # held through an iteration's I/O (round, finish) and through QUIT: the idle watch reads the link only
        # while none of them runs
        self.run_lock = threading.RLock()
        self.closing = False
        self.pure = bool(pure)
        self._sids = itertools.count(1)
        self.fault = None                                  # v1's TFPP_FAULT is the single-stream abort path
        self.steps = self.req_base = 0
        self.armed = False
        self.v2fault = Fault(os.environ.get("TFPP_GLM_FAULT_V2", ""))     # armed after the warm-up
        self.v2stats = RoundStats("leader") if os.environ.get("TFPP_GLM_TIMING") == "1" else None
        if pure and (link is not None or a != 0):
            raise ValueError("--pure is the one-process reference: --layers 0:END and no followers")
        if link is None and a != 0:
            raise ValueError(f"layers {a}:{b}: a leader past layer 0 listens for the stage before it (--listen)")

        # 1. policies, copy drafts (G/engine.py:503-510); v2: the cost-table policy is refused for every auto request
        E.encode_policy(policy)
        if not E.encode_policy(E.DFLASH_POLICY)[0] >= 10:
            raise ValueError(f"TF_GLM_DFLASH_POLICY={E.DFLASH_POLICY!r}: a DFlash2 policy (f...), e.g. fnc7:0.3")
        if E.encode_policy(E.DFLASH_POLICY)[0] == 17:
            raise ValueError(f"TF_GLM_DFLASH_POLICY={E.DFLASH_POLICY!r}: fcost needs the startup cost table, which "
                             "tfpp-glm does not measure (use fnc7:0.3 or fc5:0.3)")
        self.copy = None if serial_only else CopySettings.from_env(E.MAX_ROWS - 1, pad=E.WIDE_GRAPHS)
        # 2. the row split stays off (:515; world=1 raises in hcsplit.py:147-149)
        self.split = SplitSettings.from_env()
        if self.split.split:
            raise ValueError("TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP: 0 in a pipeline")
        # 3. (:521-543); v2: 2..8 streams
        parallel = check_parallel(parallel, drafter, serial_only)
        if parallel < 2:
            raise ValueError("PipeMultiEngine serves --parallel 2..8; --parallel 1 is PipeGlmEngine")
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank, self.world, self.parallel = 0, 1, parallel
        self.spill_cfg = None
        self.multi_window = E.multi_window()
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        # 4.* no communicator (:544), and w.comm stays None (:783)
        self.comm = None
        # 5. (:558-618)
        cfg = Config.read(model_dir)
        if b != (truncate if truncate is not None else cfg.layers):
            raise ValueError(f"the leader's layers end at {truncate if truncate is not None else cfg.layers}, not {b}"
                             + ("" if truncate is not None else " (--truncate L for a shortened test model)"))
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        self.tp = tp_mod.Layout.from_config(cfg, 0, 1)        # a Layout, as on the GB10 (TF/cuda/server.py:540)
        kv = kv_kind()
        if kv != "bf16" and not LATENT:
            raise ValueError("TF_GLM_KV=fp8 holds the latent cache: it needs TF_GLM_LATENT=1")
        self.kv = kv
        self.mtp_on = False                                   # multi: DFlash2 drafts only (G/engine.py:603)
        if prefill_rows is None:
            prefill_rows = int(os.environ.get("TF_GLM_PREFILL_ROWS", "") or PREFILL_ROWS)
        prefill_rows = int(prefill_rows)
        if not 64 <= prefill_rows <= 16384 or prefill_rows % 64:
            raise ValueError(f"prefill rows: a multiple of 64 from 64 to 16,384, not {prefill_rows}")
        self.grid = E.prompt_grid(prefill_rows)
        self.shared = E.shared_prefix()
        vision = bool(vision)
        if drafter is None and not serial_only:
            raise ValueError("no DFlash2 draft model was given: pass --drafter, or --no-drafts for the serial reference")
        # 6.* the tap layers (DFlash2's config; --tap-layers for shortened test models), then HELLO -> GO
        taps: tuple[int, ...] = ()
        if drafter is not None:
            conf = read_taps(drafter)
            taps = tuple(int(t) for t in tap_layers) if tap_layers else conf
            if len(taps) != len(conf):
                raise ValueError(f"--tap-layers: {len(conf)} layers (DFlash2 reads {len(conf)} taps), not {len(taps)}")
        elif tap_layers:
            raise ValueError("--tap-layers needs a drafter")
        own = startup.stage_info((a, b), "leader", model_dir)
        infos = startup.leader_hello(link, own, timeout=startup_timeout)
        ranges = [tuple(i.layers) for i in infos]
        try:
            proto.check_chain(ranges, b, cfg.kinds, taps)
            problems = proto.compare_stages(infos)
            if problems:
                raise RuntimeError("the stages were started with different settings:\n  " + "\n  ".join(problems))
        except Exception as exc:
            startup.leader_fail(link, "HELLO", exc)
            raise
        routes = proto.tap_routes(taps, ranges)
        fp8 = bool(prompt_precision.FP8_BY_DEFAULT)
        prompt_precision.set_fp8(fp8)                         # before any weight loads (TF/cli.py:275-278)
        go = {"P": parallel, "context": context, "explicit": explicit, "prefill_rows": prefill_rows,
              "max_rows": E.MAX_ROWS, "graph_rows": list(E.GRAPH_ROWS + E.WIDE_GRAPHS), "graphs": bool(graphs),
              "taps": list(taps), "stages": [list(r) for r in ranges], "fp8": fp8, "end": b}
        if link is not None:
            link.send(proto.GO, go)
        cin, _ = routes[-1]
        stage = part.Stage(index=len(ranges) - 1, count=len(ranges), layers=(a, b), first=a == 0, last=True,
                           taps=taps, carry_in=cin, carry_out=())
        self.stage = stage
        if link is not None:
            self.io = ActIO(stage, link, None, warm=KeepWarm(), timeout=io_timeout)
            stage.io = self.io
        if not pure:
            v2part.install(stage)                             # v2: the prompt ACT under v2 headers
        E.VisionFeed = feed.PipeVisionFeed                    # generate builds it by global name (G/engine.py:1506)
        log(f"leader: layers [{a}, {b}) of a {len(ranges)}-stage chain {ranges}; taps {list(taps)}; v2 --parallel "
            f"{parallel}" + (" (pure: the package's own _compute)" if pure else ""))

        # 7.* this stage's layers (head, norm, the embedding when it drafts; no MTP under --parallel)
        try:
            t0 = time.perf_counter()
            w = load_range(model_dir, a, b, is_first=a == 0, is_last=True, mtp=self.mtp_on,
                           drafter=drafter is not None, prefetch_next=prefetch_next)
            log(f"leader: layers loaded in {time.perf_counter() - t0:.1f}s ({w.nbytes() / GIB:.2f} GiB)")
        except Exception as exc:
            startup.leader_fail(link, "load", exc)
            raise
        # 8.* CAP -> the chain's admission -> the pool -> AGREE
        try:
            staging = capacity.staging_bytes(prefill_rows, len(cin), 0, first=a == 0, last=True)
            own_cap = capacity.measure_budget(reserve_gib=reserve_gib, budget_gib=budget_gib, staging=staging)
            own_cap["layers"] = [a, b]
            caps = startup.leader_caps(link, own_cap, timeout=startup_timeout)
            if [tuple(c["layers"]) for c in caps] != ranges:
                raise proto.ProtocolError(f"CAP from {[c['layers'] for c in caps]}, expected {ranges}")
            text = read_config(model_dir)
            flags = capacity.flags_now(cfg.quant)
            fixed, draft_geo = self._leader_extras_v2(model_dir, drafter, vision, parallel)
            last = len(ranges) - 1
            geos = [capacity.stage_geometry(text, r[0], r[1], last=i == last, mtp=False, parallel=parallel,
                                            prefill_rows=prefill_rows, flags=flags,
                                            fixed=fixed if i == last else 0, draft=draft_geo if i == last else None)
                    for i, r in enumerate(ranges)]
            wanted = int(float(os.environ.get("TF_GLM_CACHE_GIB", "3")) * GIB)
            agreed = capacity.agree(native=int(text.get("max_position_embeddings") or 0),
                                    requested=context if explicit else cfg.dense_limit, explicit=explicit,
                                    budgets=[int(c["budget"]) for c in caps], geos=geos,
                                    kinds=[capacity.layer_kinds(text, r[0], r[1]) for r in ranges],
                                    cache_bytes_wanted=wanted, dense_limit=cfg.dense_limit, max_rows=E.MAX_ROWS,
                                    dense_capacity=E.DENSE_CAPACITY)
            states = v2capacity.stage_states(text, ranges,
                                             drafter_cfg=read_config(drafter) if drafter is not None else None)
            plan = v2capacity.pool_plan(
                window=agreed["window"], cap=agreed["capacity"], geos=geos, spare=agreed["spare"], states=states,
                cache_wanted=wanted, entries_cfg=int(os.environ.get("TF_GLM_CACHE_ENTRIES", "8")),
                share_pct=float(os.environ.get("TF_GLM_CACHE_SHARE_PCT", "0") or 0), max_rows=E.MAX_ROWS,
                pool_rows=pool_rows, kept_entries=kept_entries,
                names=[f"stage {i} [{r[0]}, {r[1]})" for i, r in enumerate(ranges)])
        except Exception as exc:
            startup.leader_fail(link, "CAP", exc)
            raise
        window, cap_slots, long_context = agreed["window"], agreed["capacity"], agreed["long_context"]
        self.capacity_plan = capacity.plan_receipt(agreed, caps[-1]["budget"])
        self.capacity_plan["pool_tokens"] = plan.rows
        self.capacity_plan["kept_bytes"] = 0
        self.v2plan = plan
        self.limit = plan.limit                              # G/engine.py:759
        self.cache_bytes = 0                                 # kept prompts live in the pool (:760)
        self.entries = plan.entries
        self.agreed = []
        self.dump = None                     # TF_GLM_DRAFT_DUMP is refused by env.check_env
        self.costs = None                    # no startup cost table (fcost is refused above)
        self.spill = None
        agree_msg = v2startup.agree_message_v2(
            {"window": window, "capacity": cap_slots, "long_context": long_context,
             "cache_bytes": 0, "entries": plan.entries, "scale": agreed["scale"],
             "pool_rows": plan.rows, "parallel": parallel, "max_rows": E.MAX_ROWS, "prefill_rows": prefill_rows,
             "graph_rows": list(E.GRAPH_ROWS + E.WIDE_GRAPHS), "graphs": bool(graphs), "taps": list(taps),
             "grid": self.grid, "kv": kv}, plan, multi_window=self.multi_window)
        if link is not None:
            link.send(proto.AGREE, agree_msg)
            link.set_staging(recv_bytes=proto.link_staging(prefill_rows, len(cin)))
            self.link = LeaderLink(link, self.io, io_timeout)
        for i, (r, c) in enumerate(zip(ranges, caps)):
            log(f"stage {i} [{r[0]}, {r[1]}): budget {c['budget'] / GIB:.2f} GiB, needs "
                f"{agreed['needed'][i] / GIB:.2f} GiB at the window, spare {agreed['spare'][i] / GIB:.2f} GiB, "
                f"a kept state {plan.state[i] / 2 ** 20:.0f} MiB, a pool token {plan.per_token[i]} B")
        if plan.note:
            log(f"kept prompt states: {plan.note}")
        log(f"context window {window} tokens ({'long contexts' if long_context else 'dense'}), cache slots "
            f"{cap_slots}")
        log(f"--parallel {parallel}: {plan.describe()}")
        self.w = w
        # 9. DFlash2 (:796): its solo contexts span the pool
        self.drafter = None
        self.drafter_dir = drafter
        if drafter is not None:
            from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=plan.rows, tap_rows=E.MAX_ROWS, ring=E.DRAFT_RING)
            if tuple(self.drafter.tap_layers) != taps:
                if not tap_layers:
                    raise RuntimeError(f"DFlash2 reads taps {self.drafter.tap_layers}, its config said {taps}")
                self.drafter.tap_layers = taps            # a shortened test model: same count, other layers
        # 10.* the engine on the pool, without graphs of its own (F-14), captured on scrubbed streams where it matters
        part.scrubbing(True)
        try:
            self.e = GuardedEngine(w, capacity=plan.rows, max_rows=E.MAX_ROWS, prefill_rows=prefill_rows,
                                   graphs=False, graph_rows=E.GRAPH_ROWS + E.WIDE_GRAPHS, long_context=long_context,
                                   taps=taps, grid=self.grid, split=None, mtp_rows=E.MTP_ROWS, streams=parallel,
                                   pool_rows=plan.rows, kv=kv)
        finally:
            part.scrubbing(False)
        # 11. (:837-843)
        if self.drafter is not None and graphs:
            self.drafter.capture()
        self.opener = E.user_opener(model_dir) if self.shared else None
        # 12. vision (:852-863), the rest of the request state (:864-870)
        self.vision = None
        if vision:
            from tensorfold.vision.glm import GlmVision

            self.vision = GlmVision(model_dir, torch.device("cuda", 0), allow_urls=vision_urls)
            self.vision.warm()
            torch.cuda.empty_cache()
            log(f"vision: a {self.vision.weight_bytes / GIB:.2f} GiB tower; stage 0 injects the rows")
        self.eos = tuple(w.cfg.eos)
        self.model_dir = model_dir
        self.request = threading.local()
        self.cache: list = []
        self.live: list[int] = []
        self.cache_entries = plan.entries
        self.e.tail = self
        # 13. v2: the batched verify seam (graphs captured before any admission), MIA's decoder and scheduler
        self.multi = self.scheduler = None
        self.concurrent = False
        build_multi(self, taps, graphs)
        # 14.* READY + BUILT from every stage, the chain's warm-up
        try:
            own_built = v2startup.built_info_v2((a, b), self.e, self.multi.verify, plan.rows, self.multi_window,
                                                self.multi.verify.segs)
            built = startup.leader_ready(link, own_built, timeout=startup_timeout)
            problems = v2startup.compare_built_v2(built)
            if problems:
                raise RuntimeError("the stages built different kernel paths (a CUDA port fell back to Triton on "
                                   "some) or different verify windows:\n  " + "\n  ".join(problems))
            if [list(x["layers"]) for x in built] != [list(r) for r in ranges]:
                raise proto.ProtocolError(f"READY from {[x['layers'] for x in built]}, expected {ranges}")
            refusals = v2capacity.kept_room_v2(caps, built, plan.entries, states)
            if refusals:
                raise RuntimeError("not enough GPU memory left after the build:\n  " + "\n  ".join(refusals))
        except Exception as exc:
            startup.leader_fail(link, "READY", exc)
            raise
        for i, (c, x) in enumerate(zip(caps, built)):
            log(f"stage {i} {x['layers']}: {x['free_after'] / GIB:.2f} GiB free after its build, verify windows "
                f"{len(x.get('verify_graphs', []))} graphs"
                + (" (--budget-gib: not checked)" if c.get("declared") else ""))
        if warmup:
            warmup_multi(self)
            log(f"after the warm-up: {torch.cuda.mem_get_info()[0] / GIB:.2f} GiB free on the leader's GPU")
        self.req_base = 0
        self.armed = True
        self.v2fault.armed = True
        if self.link is not None:
            threading.Thread(target=self._watch, name="tfpp-glm-watch", daemon=True).start()

    # -- what the leader carries besides its layers ------------------------------------------------------------
    @staticmethod
    def _leader_extras_v2(model_dir, drafter, vision: bool, parallel: int):
        """(resident bytes, DFlash2 context geometry): as v1's, with the multi-stream drafter's pool of contexts beside
        the solo drafter's (G/engine.py:653-662, 382-410)."""

        fixed, draft_geo = 0, None
        if vision:
            fixed += vision_tower_bytes(model_dir) + E.vision_workspace()
        if drafter is not None:
            from tensorfold.cuda.capacity import Geometry, config as read_config
            from tensorfold.cuda.geometry import dflash2_geometry, dflash2_weights

            fixed += dflash2_weights(drafter, 1, E.draft_fraction(drafter, 0, 1)).resident
            d = read_config(drafter)
            g = dflash2_geometry(d, 1, E.MAX_ROWS, ring=E.DRAFT_RING, kv_heads=int(d["num_key_value_heads"]))
            draft_geo = Geometry(lambda slots: g.bytes_at(slots) + E.multi_draft_bytes(
                d, parallel, 1, ring=E.DRAFT_RING, capacity=slots), g.reserve)
        return fixed, draft_geo

    def _share(self, values):                       # MIA rank 0's _flush (G/multi.py:636); the return is unused
        if self.link is not None:
            send_multi(self, values)
        return list(values)

    def warmup(self) -> None:                       # v1's single-stream warm-up has no meaning here
        warmup_multi(self)


def warmup_multi(eng) -> None:
    """The first whole-chain I/O and a self-check (spec v2 7.4): P prompts of different lengths, all admitted in one
    lockstep iteration (so the first chunk is a multi-prompt one), serially and then drafted; the replies must be the
    same tokens. Both passes run under the lockstep harness, so the number of messages, the stream ids and the kept
    ids at the end of the warm-up are the same every start. Then every kept prompt is evicted (EVICT on every stage)."""

    P = eng.parallel
    rng = np.random.default_rng(20261023)
    prompts = [[int(t) for t in rng.integers(0, eng.w.cfg.vocab, 64 + 16 * i)] for i in range(P)]
    sch = eng.scheduler

    def batch(draft: bool) -> list[list[int]]:
        out: dict[str, list[int]] = {}

        def client(tag: str, prompt: list[int]):
            got: list[int] = []
            eng.request.policy, eng.request.stop_eos, eng.request.trace_tag = None, True, tag
            t = time.perf_counter()
            eng.generate(prompt, 8, None, lambda toks: (got.extend(int(x) for x in toks), False)[1], draft=draft)
            out[tag] = got
            return time.perf_counter() - t

        reqs = [Req(tag=f"warm{int(draft)}-{i}", at=0, run=(lambda tag=f"warm{int(draft)}-{i}", p=p:
                                                                      client(tag, p)))
                for i, p in enumerate(prompts)]
        lock = Lockstep(sch, reqs, landed=eng.v2landed)
        t = time.perf_counter()
        lock.run(timeout=3600.0)
        failed = {tag: err for tag, (ok, err) in lock.results.items() if not ok}
        if failed:
            raise RuntimeError(f"warm-up ({'drafted' if draft else 'serial'}): {failed}")
        log(f"warm-up ({'drafted' if draft else 'serial'}): {P} streams, {sum(len(v) for v in out.values())} tokens in "
            f"{time.perf_counter() - t:.2f}s")
        return [out[f"warm{int(draft)}-{i}"] for i in range(P)]

    serial = batch(False)
    drafted = batch(True) if not eng.serial_only else serial
    ev = threading.Event()
    eng.multi.clear_due = ev                          # the worker evicts every kept prompt at its next idle iteration
    sch.waiting.put((sch._Flush, threading.Event()))  # wakes an idle worker; changes no state (G/multi.py:2586-2587)
    if not ev.wait(120.0):
        raise RuntimeError("warm-up: the kept prompts were not evicted within 120 s")
    torch.cuda.empty_cache()
    if serial != drafted:
        bad = [i for i in range(P) if serial[i] != drafted[i]]
        raise RuntimeError(f"warm-up: the drafted replies {bad} differ from the serial ones "
                           f"({[drafted[i] for i in bad]} vs {[serial[i] for i in bad]})")
