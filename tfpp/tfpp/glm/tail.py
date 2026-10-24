"""The leader: the last stage of the chain, MIA's rank 0 (spec section 9).

It runs the MIA engine's request path unchanged (``GlmApp`` + ``serve``, ``GlmEngine.generate`` / ``_run`` /
``_run_once`` / ``_take_over`` / ``_remember`` / ``_resume`` / ``_shared``, every decode loop, DFlash2, copy drafts,
MTP at P=1, grammar, thinking, tool calls, StopVote) on its own layer range, with ``comm = None`` (so forward,
sampling and DFlash2 take their single-rank branches: G/forward.py:416-417, G/decode.py:97-99). What reaches the
followers is a handful of seams:

- ``TailEngine.forward``: STEP up the chain, the window's ACT back, then the original forward (G/decode.py:246-267).
- ``decode.prefill``: PREFILL (the whole prompt call: reset or restore, every chunk) before the original runs; the
  chunks' ACT frames are read inside ``_compute_pp``.
- ``decode.commit`` / ``drafter_choice.commit`` (bound at import: G/decode.py:16, G/drafter_choice.py:12): KEEP
  first, so the followers commit while the leader does.
- ``decode.take_snapshot`` / ``save_rows`` / ``load_rows``: SNAP / SAVE / LOAD after the original.
- ``GlmEngine._drop``: DROP; ``_run``: END (abort after an exception).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.families.glm5_next.cuda import decode as D, drafter_choice as DC, engine as E, forward as fw
from tensorfold.families.glm5_next.cuda.engine import GlmEngine

from . import capacity, feed, part, proto, startup
from .load import load_range
from .wire import ActIO, Link, LinkError

GIB = 2 ** 30


def log(msg: str) -> None:
    print(f"[tfpp-glm] {msg}", flush=True)


class TailEngine(D.Engine):
    """``decode.Engine`` whose decode window is the chain's: STEP up, ACT back, then the original forward. Every
    stage runs its window at the same ``st.pos`` and R, so all of them pick the same graph or the eager path."""

    tail = None                     # the PipeGlmEngine (None while it is being built, and for one process)

    def forward(self, tokens):
        T = self.tail
        if T is None or T.link is None:
            return super().forward(tokens)
        R = len(tokens)
        if R > self.buf.rows or self.st is not self.home:
            raise ValueError("tfpp-glm v1: one stream at home, windows up to TF_GLM_MAX_ROWS rows")
        fw.check_room(self.w, self.st, R)              # before anything is sent
        T.link.step(self.st.pos, tokens)
        if T.fault is not None and T.armed:
            T.steps += 1
            if (T.link.req - T.req_base, T.steps) == T.fault:
                raise RuntimeError(f"tfpp-glm: injected fault at request {T.fault[0]}, step {T.fault[1]} (TFPP_FAULT)")
        T.io.recv_act(self.buf, R, self.st, "step")
        return super().forward(tokens)


class LeaderLink:
    """The leader's control side of its one link, and how many ACT frames it has asked for and not read."""

    def __init__(self, link: Link, io: ActIO, timeout: float) -> None:
        self.link, self.io, self.timeout = link, io, float(timeout)
        self.req = 0

    def begin(self) -> None:
        self.req += 1
        self.io.outstanding = 0

    def ctl(self, op: str, **fields) -> None:
        self.link.send(op, fields)

    def prefill(self, *, mode: str, sid, begin: int, end: int, rows: int, ids, feed=None) -> None:
        parts = [("ids", np.asarray(list(ids), dtype=np.int32))]
        img_pos: list[int] = []
        if feed is not None and part.STAGE is not None and not part.STAGE.first and hasattr(feed, "between"):
            pos, rws = feed.between(begin, end)
            if pos:
                img_pos = [int(p) for p in pos]
                parts.append(("img", rws.to(torch.bfloat16)))
        self.link.send(proto.PREFILL, {"req": self.req, "mode": mode, "sid": sid, "begin": int(begin),
                                       "end": int(end), "rows": int(rows), "img_pos": img_pos}, parts)
        self.io.outstanding += proto.chunk_count(begin, end, rows)

    def step(self, pos: int, ids) -> None:
        self.link.send(proto.STEP, {"req": self.req, "pos": int(pos), "R": len(ids)},
                       [("ids", np.asarray(list(ids), dtype=np.int32))])
        self.io.outstanding += 1

    def keep(self, pos: int) -> None:
        self.link.send(proto.KEEP, {"req": self.req, "pos": int(pos)})

    def end(self, pos: int, abort: bool, live: list[int]) -> None:
        self.link.send(proto.END, {"req": self.req, "pos": int(pos), "abort": bool(abort), "live": list(live)})

    def drain(self) -> None:
        """Read and drop the ACT frames still owed (an exception left them in flight)."""

        while self.io.outstanding > 0:
            f = self.link.recv(timeout=self.timeout)
            if f.op == proto.ERROR:
                raise LinkError(f"stage {f.hdr.get('stage')} failed in {f.hdr.get('where')}: {f.hdr.get('msg')}")
            if f.op != proto.ACT:
                raise LinkError(f"draining: expected ACT, got {f.op}")
            self.io.outstanding -= 1

    def quit(self) -> None:
        self.link.send(proto.QUIT, {})


def fault_point(text: str):
    """TFPP_FAULT=REQ:STEP: the leader raises after sending the STEP-th window of its REQ-th request after the
    warm-up (an exception with an ACT in flight: the abort path, spec 8.4). Tests only."""

    if not text.strip():
        return None
    req, step = (int(x) for x in text.split(":"))
    return req, step


def read_taps(drafter_dir) -> tuple[int, ...]:
    """DFlash2's tap layers from its config.json (G/dflash2.py:262), without reading its weights."""

    cfg = json.loads((Path(drafter_dir) / "config.json").read_text())
    return tuple(int(i) for i in (cfg.get("dflash_config") or {})["target_layer_ids"])


def vision_tower_bytes(model_dir) -> int:
    """The tower's bf16 weights as MIA's estimate counts them (G/engine.py:620-625), from its shards' headers."""

    from tensorfold.cuda.capacity import headers

    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({model_dir / f for n, f in index.items() if n.startswith("model.visual.")})
    if not files:
        return 0
    return sum(math.prod(info["shape"]) * 2 for n, info in headers(model_dir, files=files).items()
               if n.startswith("model.visual."))


class PipeGlmEngine(GlmEngine):
    """GlmEngine on one stage's layers at world=1. ``__init__`` sets what G/engine.py:503-891 sets, in its order;
    the steps marked * differ: no communicator, the startup handshake (HELLO -> GO -> CAP -> AGREE -> READY),
    this stage's layers only, the chain's own admission, and ``TailEngine``."""

    def __init__(self, model_dir, *, layers: tuple[int, int], truncate: int | None = None, link: Link | None = None,
                 policy: str = E.DEFAULT_POLICY, drafter=None, context: int | None = None,
                 context_explicit: bool | None = None, serial_only: bool = False, prefill_rows: int | None = None,
                 vision: bool = False, vision_urls: bool = False, parallel: int = 1, graphs: bool = True,
                 tap_layers=None, reserve_gib: float = 3.0, budget_gib: float | None = None,
                 io_timeout: float = 600.0, startup_timeout: float = 3600.0, pure: bool = False,
                 prefetch_next: bool = True, warmup: bool = True) -> None:
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
        # held through a request (_run) and through QUIT: the idle watch reads the link only while neither runs
        self.run_lock = threading.RLock()
        self.closing = False
        self.pure = bool(pure)
        self._sids = itertools.count(1)
        self.fault = fault_point(os.environ.get("TFPP_FAULT", ""))   # tests (T1-fault): "REQ:STEP"
        self.steps = self.req_base = 0
        self.armed = False                   # TFPP_FAULT counts requests after the warm-up
        if pure and (link is not None or a != 0):
            raise ValueError("--pure is the one-process reference: --layers 0:END and no followers")
        if link is None and a != 0:
            raise ValueError(f"layers {a}:{b}: a leader past layer 0 listens for the stage before it (--listen)")

        # 1. policies, copy drafts (G/engine.py:503-510)
        E.encode_policy(policy)
        if not E.encode_policy(E.DFLASH_POLICY)[0] >= 10:
            raise ValueError(f"TF_GLM_DFLASH_POLICY={E.DFLASH_POLICY!r}: a DFlash2 policy (f...), e.g. fnc7:0.3")
        self.copy = None if serial_only else CopySettings.from_env(E.MAX_ROWS - 1, pad=E.WIDE_GRAPHS)
        # 2. the row split stays off (:515; world=1 raises in hcsplit.py:147-149)
        self.split = SplitSettings.from_env()
        if self.split.split:
            raise ValueError("TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP: 0 in a pipeline")
        # 3. (:521-543)
        parallel = int(parallel)
        if parallel != 1:
            raise ValueError("tfpp-glm v1 serves one stream (--parallel 1); --parallel N is v2 (spec section 14)")
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
        self.mtp_on = E.mtp_head(drafter is not None, serial_only, cfg.mtp_layers) and parallel == 1
        if prefill_rows is None:
            prefill_rows = int(os.environ.get("TF_GLM_PREFILL_ROWS", "") or PREFILL_ROWS)
        prefill_rows = int(prefill_rows)
        if not 64 <= prefill_rows <= 16384 or prefill_rows % 64:
            raise ValueError(f"prefill rows: a multiple of 64 from 64 to 16,384, not {prefill_rows}")
        self.grid = E.prompt_grid(prefill_rows)
        self.shared = E.shared_prefix()
        vision = bool(vision)
        if not self.mtp_on and drafter is None and not serial_only:
            raise ValueError(("TF_GLM_MTP=0 leaves" if cfg.mtp_layers else "this checkpoint has") + " no MTP head and "
                             "no DFlash2 draft model was given: pass --drafter, or --no-drafts for the serial reference")
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
            part.install_common(stage)
        E.VisionFeed = feed.PipeVisionFeed                    # generate builds it by global name (G/engine.py:1506)
        log(f"leader: layers [{a}, {b}) of a {len(ranges)}-stage chain {ranges}; taps {list(taps)}"
            + (" (pure: the package's own _compute)" if pure else ""))

        # 7.* this stage's layers (head, norm, the embedding when it drafts, MTP at P=1)
        try:
            t0 = time.perf_counter()
            w = load_range(model_dir, a, b, is_first=a == 0, is_last=True, mtp=self.mtp_on,
                           drafter=drafter is not None, prefetch_next=prefetch_next)
            log(f"leader: layers loaded in {time.perf_counter() - t0:.1f}s ({w.nbytes() / GIB:.2f} GiB)")
        except Exception as exc:
            startup.leader_fail(link, "load", exc)
            raise
        # 8.* CAP -> the chain's admission -> AGREE
        try:
            staging = capacity.staging_bytes(prefill_rows, len(cin), 0, first=a == 0, last=True)
            own_cap = capacity.measure_budget(reserve_gib=reserve_gib, budget_gib=budget_gib, staging=staging)
            own_cap["layers"] = [a, b]
            caps = startup.leader_caps(link, own_cap, timeout=startup_timeout)
            if [tuple(c["layers"]) for c in caps] != ranges:
                raise proto.ProtocolError(f"CAP from {[c['layers'] for c in caps]}, expected {ranges}")
            text = read_config(model_dir)
            flags = capacity.flags_now(cfg.quant)
            fixed, draft_geo = self._leader_extras(model_dir, drafter, vision)
            last = len(ranges) - 1
            geos = [capacity.stage_geometry(text, r[0], r[1], last=i == last, mtp=self.mtp_on and i == last,
                                            parallel=parallel, prefill_rows=prefill_rows, flags=flags,
                                            fixed=fixed if i == last else 0, draft=draft_geo if i == last else None)
                    for i, r in enumerate(ranges)]
            wanted = int(float(os.environ.get("TF_GLM_CACHE_GIB", "3")) * GIB)
            agreed = capacity.agree(native=int(text.get("max_position_embeddings") or 0),
                                    requested=context if explicit else cfg.dense_limit, explicit=explicit,
                                    budgets=[int(c["budget"]) for c in caps], geos=geos,
                                    kinds=[capacity.layer_kinds(text, r[0], r[1]) for r in ranges],
                                    cache_bytes_wanted=wanted, dense_limit=cfg.dense_limit, max_rows=E.MAX_ROWS,
                                    dense_capacity=E.DENSE_CAPACITY)
        except Exception as exc:
            startup.leader_fail(link, "CAP", exc)
            raise
        window, cap_slots, long_context = agreed["window"], agreed["capacity"], agreed["long_context"]
        self.capacity_plan = capacity.plan_receipt(agreed, caps[-1]["budget"])
        self.limit = window
        self.cache_bytes = agreed["cache_bytes"]
        self.entries = int(os.environ.get("TF_GLM_CACHE_ENTRIES", "8"))
        self.agreed = []
        self.dump = None                     # TF_GLM_DRAFT_DUMP is refused by env.check_env
        self.costs = None                    # no startup cost table (fcost, auto with a drafter: v2.x)
        self.spill = None
        agree_msg = {"window": window, "capacity": cap_slots, "long_context": long_context,
                     "cache_bytes": self.cache_bytes, "entries": self.entries, "scale": agreed["scale"],
                     "pool_rows": None, "parallel": parallel, "max_rows": E.MAX_ROWS, "prefill_rows": prefill_rows,
                     "graph_rows": list(E.GRAPH_ROWS + E.WIDE_GRAPHS), "graphs": bool(graphs), "taps": list(taps),
                     "grid": self.grid, "kv": kv}
        if link is not None:
            link.send(proto.AGREE, agree_msg)
            link.set_staging(recv_bytes=proto.link_staging(prefill_rows, len(cin)))
            self.link = LeaderLink(link, self.io, io_timeout)
        for i, (r, c) in enumerate(zip(ranges, caps)):
            log(f"stage {i} [{r[0]}, {r[1]}): budget {c['budget'] / GIB:.2f} GiB, needs "
                f"{agreed['needed'][i] / GIB:.2f} GiB at the window, spare {agreed['spare'][i] / GIB:.2f} GiB, "
                f"kept-prompt scale {agreed['scale'][i]:g}")
        log(f"context window {window} tokens ({'long contexts' if long_context else 'dense'}), cache slots "
            f"{cap_slots}; other conversations' prompts kept in {self.cache_bytes / GIB:.2f} GiB of the leader's "
            f"bytes (TF_GLM_CACHE_GIB asks {wanted / GIB:.1f})")
        self.w = w
        # 9. DFlash2 (:791-796)
        self.drafter = None
        self.drafter_dir = drafter
        if drafter is not None:
            from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=cap_slots, tap_rows=E.MAX_ROWS, ring=E.DRAFT_RING)
            if tuple(self.drafter.tap_layers) != taps:
                if not tap_layers:
                    raise RuntimeError(f"DFlash2 reads taps {self.drafter.tap_layers}, its config said {taps}")
                self.drafter.tap_layers = taps            # a shortened test model: same count, other layers
        # 10.* the engine, its graphs captured on scrubbed streams (spec section 6)
        part.scrubbing(True)
        try:
            self.e = TailEngine(w, capacity=cap_slots, max_rows=E.MAX_ROWS, prefill_rows=prefill_rows, graphs=graphs,
                                graph_rows=E.GRAPH_ROWS + E.WIDE_GRAPHS, long_context=long_context, taps=taps,
                                grid=self.grid, split=None, mtp_rows=E.MTP_ROWS, streams=parallel, pool_rows=None,
                                kv=kv)
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
        self.cache_entries = self.entries
        # 13. P=1: no MultiDecoder, no scheduler (the class defaults)
        self.multi = self.scheduler = None
        self.concurrent = False
        # 14.* READY + BUILT from every stage, the seams, the chain's warm-up
        try:
            own_built = startup.built_info((a, b), self.e)
            built = startup.leader_ready(link, own_built, timeout=startup_timeout)
            problems = startup.compare_built(built)
            if problems:
                raise RuntimeError("the stages built different kernel paths (a CUDA port fell back to Triton on "
                                   "some):\n  " + "\n  ".join(problems))
            if [list(x["layers"]) for x in built] != [list(r) for r in ranges]:
                raise proto.ProtocolError(f"READY from {[x['layers'] for x in built]}, expected {ranges}")
            kept, refusals = capacity.kept_room(caps, built, agreed["scale"], self.cache_bytes)
            if refusals:
                raise RuntimeError("not enough GPU memory left after the build:\n  " + "\n  ".join(refusals))
        except Exception as exc:
            startup.leader_fail(link, "READY", exc)
            raise
        for i, (c, x) in enumerate(zip(caps, built)):
            log(f"stage {i} {x['layers']}: {x['free_after'] / GIB:.2f} GiB free after its build"
                + (" (--budget-gib: not checked)" if c.get("declared") else ""))
        if kept < self.cache_bytes:
            log(f"kept prompts: {kept / GIB:.2f} GiB of the leader's bytes, not {self.cache_bytes / GIB:.2f}: what the "
                "stages have free after their builds, half of each reserve kept free")
            self.cache_bytes = kept
            self.capacity_plan["kept_bytes"] = kept
        self.e.tail = self
        if not pure:
            install_tail_seams_v1(self)
        if warmup:
            self.warmup()
            log(f"after the warm-up: {torch.cuda.mem_get_info()[0] / GIB:.2f} GiB free on the leader's GPU")
        self.req_base = self.link.req if self.link is not None else 0
        self.armed = True
        if self.link is not None:
            threading.Thread(target=self._watch, name="tfpp-glm-watch", daemon=True).start()

    # -- what the leader carries besides its layers ------------------------------------------------------------
    @staticmethod
    def _leader_extras(model_dir, drafter, vision: bool):
        """(resident bytes, DFlash2 context geometry): the tower and its workspace (G/engine.py:73-81, 619-625),
        DFlash2's weights and context (G/engine.py:653-662), counted in the leader's geometry."""

        fixed, draft_geo = 0, None
        if vision:
            fixed += vision_tower_bytes(model_dir) + E.vision_workspace()
        if drafter is not None:
            from tensorfold.cuda.capacity import config as read_config
            from tensorfold.cuda.geometry import dflash2_geometry, dflash2_weights

            fixed += dflash2_weights(drafter, 1, E.draft_fraction(drafter, 0, 1)).resident
            d = read_config(drafter)
            draft_geo = dflash2_geometry(d, 1, E.MAX_ROWS, ring=E.DRAFT_RING, kv_heads=int(d["num_key_value_heads"]))
        return fixed, draft_geo

    # -- the overrides (spec 9.3) ------------------------------------------------------------------------------
    def new_sid(self) -> int:
        return next(self._sids)

    def kept_sids(self) -> list[int]:
        return [int(c.tfpp_sid) for c in self.cache if getattr(c, "tfpp_sid", None) is not None]

    def _share(self, values):
        return list(values)                  # one rank: rank 0's list as it is (generate: G/engine.py:1531-1536)

    def _drop(self, snap) -> None:
        sid = getattr(snap, "tfpp_sid", None)
        if sid is not None and self.link is not None and not self.broken:
            self.link.ctl(proto.DROP, sid=int(sid))
        super()._drop(snap)

    def _effective(self, code):
        code = super()._effective(code)
        if code[0] % 10 == 7 or (code[0] in (4, 5) and self.drafter is not None):
            raise ValueError("this draft policy needs the startup cost table, which tfpp-glm does not measure yet "
                             "(v2.x); use fnc7:0.3 (the default), fc5:0.3, cN:P, N or 0")
        return code

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None, *, vision=None,
                 background=False):
        if self.broken:
            raise RuntimeError(f"tfpp-glm: the chain broke ({self.broken}); restart every stage")
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft, constraint, vision=vision,
                                background=background)

    def _run(self, *args, **kwargs):
        link = self.link
        if link is None:
            return super()._run(*args, **kwargs)
        with self.run_lock:                  # the idle watch keeps off the link until END
            if self.broken:
                raise RuntimeError(f"tfpp-glm: the chain broke ({self.broken}); restart every stage")
            link.begin()
            self.steps = 0
            try:
                out = super()._run(*args, **kwargs)
            except BaseException as exc:
                self._abort(exc)
                raise
            try:
                link.end(self.e.st.pos, False, self.kept_sids())
            except Exception as exc:
                self._fatal(exc)
                raise
            return out

    def _watch(self) -> None:
        """The link between requests (a daemon thread). A follower sends only what a request asks for (ACT for a
        PREFILL chunk or a STEP), so anything readable while no request runs is the chain breaking: a follower's
        ERROR (it lost the stage before it and stopped: follower.recv_ctl) or the link closing. Fail-stop now (spec
        8.7) rather than on the next user's request."""

        raw = self.link.link
        while not self.broken and not self.closing:
            if not raw.readable(1.0):
                continue
            with self.run_lock:              # a request in flight reads its own frames: wait for its END
                if self.broken or self.closing or not raw.readable(0.0):
                    continue
                try:
                    f = raw.recv(timeout=self.link.timeout)
                    if f.op == proto.ERROR:
                        exc: BaseException = LinkError(f"between requests: stage {f.hdr.get('stage')} failed in "
                                                       f"{f.hdr.get('where')}: {f.hdr.get('msg')}")
                    else:
                        exc = proto.ProtocolError(f"between requests: an unasked {f.op} from the stage before")
                except Exception as err:  # noqa: BLE001  (a closed link: LinkError)
                    exc = err
                self._fatal(exc)
                return

    def _abort(self, exc: BaseException) -> None:
        """A request failed on the leader: read the ACT frames still in flight, tell the followers to drop their
        uncommitted window (END abort), keep nothing whose rows are in doubt. A broken link is fatal instead."""

        if self.broken or isinstance(exc, (LinkError, proto.ProtocolError)):
            self._fatal(exc)
            return
        try:
            self.link.drain()
            self.live = []                   # the caches' rows past the failure are not this prompt's
            self.link.end(self.e.st.pos, True, self.kept_sids())
            log(f"request {self.link.req} aborted on the leader ({type(exc).__name__}: {exc}); the chain goes on")
        except Exception as again:  # noqa: BLE001
            self._fatal(again)

    def _fatal(self, exc: BaseException) -> None:
        """Fail-stop: the request in flight answers 500, the process exits 75 a moment later (every stage must be
        restarted in order: no stage rejoins a chain, as MIA's ranks do not: G/multi.py:125-136)."""

        if self.broken:
            return
        self.broken = f"{type(exc).__name__}: {exc}"
        log(f"FATAL: the chain broke: {self.broken}; exiting {proto.FAIL_EXIT}")
        threading.Timer(1.0, os._exit, (proto.FAIL_EXIT,)).start()

    def quit(self) -> None:
        held = self.run_lock.acquire(timeout=30.0)    # a request still running: QUIT after its END (30 s at most)
        try:
            self.closing = True                       # the followers close their links after QUIT: not a break
            if self.link is not None and not self.broken:
                try:
                    self.link.quit()
                except Exception as exc:  # noqa: BLE001
                    log(f"QUIT not sent: {exc}")
        finally:
            if held:
                self.run_lock.release()

    def warmup(self) -> None:
        """The first whole-chain I/O and a self-check (spec 9.5): a 64-token prompt serially (PREFILL, STEP, KEEP,
        END), again drafted (SNAP), then every kept state dropped (DROP); both replies must be the same tokens."""

        rng = np.random.default_rng(20261023)
        prompt = [int(t) for t in rng.integers(0, self.w.cfg.vocab, 64)]

        def run(draft: bool) -> list[int]:
            got: list[int] = []
            self.request.policy, self.request.stop_eos = None, True
            t = time.perf_counter()
            self.generate(prompt, 8, None, lambda toks: (got.extend(int(x) for x in toks), False)[1], draft=draft)
            log(f"warm-up ({'drafted' if draft else 'serial'}): {len(got)} tokens in {time.perf_counter() - t:.2f}s")
            return got

        serial = run(False)
        drafted = run(True) if not self.serial_only else serial
        for c in list(self.cache):
            self._drop(c)
        torch.cuda.empty_cache()
        if serial != drafted:
            raise RuntimeError(f"warm-up: the drafted reply {drafted} differs from the serial one {serial}")


def _validate_prefill(e, prompt, mtp: bool, drafter, resume, sample: bool) -> None:
    """``decode.prefill``'s refusals (G/decode.py:567-586), and the room check its chunks would make, before PREFILL
    goes out: after it, any failure leaves the followers a step ahead (fail-stop)."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    use_mtp = mtp and e.w.mtp is not None
    replay = sample and D.whole(resume, prompt)
    if resume is not None:
        begin = len(resume.ids)
        if e.grid and begin % e.grid:
            raise ValueError(f"a resumed prefill starts on the {e.grid}-row grid, not at {begin}")
        if (begin >= len(prompt) and not replay) or list(prompt[:begin]) != resume.ids:
            raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt (or of the whole "
                             "prompt with its head's logits row)")
        if (use_mtp and resume.mtp_len < 0) or (drafter is not None and resume.drafter_end != begin):
            raise ValueError("this snapshot's draft caches do not fit the request")
        if getattr(resume, "tfpp_sid", None) is None:
            raise ValueError("tfpp-glm: a kept state no follower holds (no tfpp_sid)")
    st = e.st
    n = len(prompt)
    if n > st.capacity:
        raise ValueError("context past the cache capacity")
    if n > e.w.cfg.dense_limit and st.index is None:
        raise ValueError(f"context {n} past {e.w.cfg.dense_limit} tokens: this engine was started without long "
                         "contexts (DSA's sparse top-k)")


def install_tail_seams_v1(eng: PipeGlmEngine) -> None:
    """The leader's v1 seams (spec 9.4). Each acts only on the leader's own engine and only with a link."""

    o = dict(prefill=D.prefill, commit=D.commit, snap=D.take_snapshot, save=D.save_rows, load=D.load_rows)

    def prefill(e, prompt, sampling, *, mtp=True, drafter=None, resume=None, feed=None, sample=True,
                keep_head=False):
        if e is not eng.e or eng.link is None:
            return o["prefill"](e, prompt, sampling, mtp=mtp, drafter=drafter, resume=resume, feed=feed,
                                sample=sample, keep_head=keep_head)
        _validate_prefill(e, prompt, mtp, drafter, resume, sample)
        begin = 0 if resume is None else len(resume.ids)
        end = begin if (sample and D.whole(resume, prompt)) else len(prompt)
        eng.link.prefill(mode="fresh" if resume is None else "resume",
                         sid=None if resume is None else int(resume.tfpp_sid), begin=begin, end=end,
                         rows=e.prefill_rows, ids=list(prompt[begin:end]), feed=feed)
        try:
            return o["prefill"](e, prompt, sampling, mtp=mtp, drafter=drafter, resume=resume, feed=feed,
                                sample=sample, keep_head=keep_head)     # the chunks' ACT: read in _compute_pp
        except BaseException as exc:
            # chunks still in flight, or the link itself: the chain has lost step. Once every chunk's ACT was read
            # (each stage has committed the prompt), an error of the first token's sampling or grammar is the
            # request's alone: _run's _abort sends END abort and the next PREFILL resets every stage, as in decode.
            if eng.link is None or eng.link.io.outstanding > 0 or isinstance(exc, (LinkError, proto.ProtocolError)):
                eng._fatal(exc)
            raise

    def commit(w, st, b, R, keep):
        if eng.link is not None and not b.prefill and b is eng.e.buf:
            eng.link.keep(st.pos + keep)       # first: the followers commit while the leader does
            try:
                return o["commit"](w, st, b, R, keep)
            except BaseException as exc:
                eng._fatal(exc)                # KEEP went out and the leader did not keep: fatal
                raise
        return o["commit"](w, st, b, R, keep)

    def take_snapshot(e, ids, pending, *, mtp, drafter=None):
        snap = o["snap"](e, ids, pending, mtp=mtp, drafter=drafter)
        if e is eng.e:
            snap.tfpp_sid = eng.new_sid()
            if eng.link is not None:
                eng.link.ctl(proto.SNAP, sid=int(snap.tfpp_sid), n=len(ids))
        return snap

    def save_rows(e, snap, drafter=None):
        o["save"](e, snap, drafter)
        if e is eng.e and eng.link is not None:
            eng.link.ctl(proto.SAVE, sid=int(snap.tfpp_sid))

    def load_rows(e, snap, drafter=None):
        o["load"](e, snap, drafter)
        if e is eng.e and eng.link is not None:
            eng.link.ctl(proto.LOAD, sid=int(snap.tfpp_sid))

    D.prefill = prefill
    D.commit = DC.commit = commit
    D.take_snapshot, D.save_rows, D.load_rows = take_snapshot, save_rows, load_rows


def exit_code(exc: BaseException) -> int:
    return proto.FAIL_EXIT if isinstance(exc, (LinkError, proto.ProtocolError)) else 1


def flush_and_exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
