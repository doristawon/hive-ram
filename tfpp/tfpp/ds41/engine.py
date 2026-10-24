"""A stage's DsEngine (spec 3.3): what F/engine.py:46-167 sets, in its order, on one stage's layers at world=1.

``StageEngine`` never calls DsEngine.__init__ (it makes NCCL at world > 1, loads every layer, warms up through a
communicator, sets the allocator ceiling from the WSL VM's meminfo): it sets the same attributes itself, in two phases
around the startup handshake -- the weights (``__init__``: between GO and CAP) and the rest (``build``: after AGREE):
no communicator, no NCCL environment, ``load_stage``'s weights, the stage's Engram tables, StageModel, the drafter on
the leader, GraphRunner (to hold the graph pool: F/graph.py:182-192), ``concurrent`` always (MultiDecoder at any
--parallel), no ``_memory_ceiling``; ``fp32_gemms()`` at the end (FO/__init__.py:68-79). Every other DsEngine method
(``prefill``, ``prefill_steps``, ``generate``, ``_sample``) is the fork's.

``PipeDsEngine`` (the leader) adds PipeOps / PipeWatch and the Scheduler; ``FollowerEngine`` has neither.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

from tensorfold.families.deepseek_v41 import fp32_gemms
from tensorfold.families.deepseek_v41.cuda import engine as E, kernels as K
from tensorfold.families.deepseek_v41.cuda.dspark import Drafter
from tensorfold.families.deepseek_v41.cuda.engine import DsEngine
from tensorfold.families.deepseek_v41.cuda.model import Comm

from . import capacity, env
from .load import load_stage, token_map
from .multi import PipeOps, PipeWatch, StageMulti, log
from .part import StageEngram, StageModel
from .proto import ProtocolError
from .wire import StageIO


class StageEngine(DsEngine):
    def __init__(self, model_dir, *, plan, mv, role: str, up=None, down=None, drafts: int, context: int,
                 parallel: int, graphs: bool = True, experts: int = 0, engram_dir: str | None = None, gate=None,
                 io_timeout: float = 600.0, tracer=None) -> None:
        torch.cuda.set_device(0)
        self.rank, self.world, self._master = 0, 1, ""
        self.model_dir = Path(model_dir)
        self.nccl = None                                     # no communicator (and no NCCL_* defaults: F/engine.py:61-62)
        self.plan, self.mv, self.role = plan, mv, role
        self.up_link, self.down_link = up, down
        self.drafts = int(drafts)
        self.parallel = int(parallel)
        self.graphs_on = bool(graphs) and bool(E.GRAPHS)
        self.experts = int(experts)
        self.engram_dir = engram_dir
        self.gate = gate
        self.io_timeout = float(io_timeout)
        self.tracer = tracer
        self.limit = int(context)
        self.max_rows = self.drafts + 1
        before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        self.w = load_stage(model_dir, plan, drafter=plan.last and self.drafts > 0, experts=self.experts, log=log)
        self.weights_bytes = int(torch.cuda.memory_allocated() - before)
        log(f"stage {plan.index} [{plan.lo},{plan.hi}): weights {self.weights_bytes / 2**30:.2f} GiB in "
            f"{time.perf_counter() - t0:.0f} s")
        self.token_map_sha = None
        self.engram_identity = None
        self.multi = self.scheduler = None

    # -- after AGREE --------------------------------------------------------------------------------------------
    def build(self, agree: dict) -> None:
        P, cfg = self.plan, self.w.cfg
        if int(agree["cap"]) != self.limit + self.max_rows + 8:
            raise ProtocolError(f"AGREE's pool of {agree['cap']} positions, this stage's context makes "
                                f"{self.limit + self.max_rows + 8}")
        eng = None
        if P.engram_owned:
            if not self.engram_dir:
                raise RuntimeError(f"stage [{P.lo},{P.hi}) owns Engram layers {list(P.engram_owned)}: --engram DIR")
            tm, self.token_map_sha = token_map(self.model_dir, cfg)
            eng = StageEngram(self.engram_dir, cfg, tm, P.engram_owned)
        carry = env.cut_post() == "carry"
        f1_round = carry and K.on("hc")
        f1_chunk = carry and K.on("hc_pf2") and K.hc_pf2_ok(cfg.dim)
        self.io = StageIO(P, self.mv, self.up_link, self.down_link, warm=self.gate, timeout=self.io_timeout,
                          tracer=self.tracer, f1_chunk=f1_chunk, f1_round=f1_round)
        self.model = StageModel(self.w, Comm(None, 1), eng, plan=P, io=self.io)      # F/engine.py:84
        self.vision = self.tower = self.vcfg = None                                  # v1: no images (88-107)
        self.drafter = Drafter(self.model) if P.last and self.drafts > 0 and self.w.dspark is not None else None
        from tensorfold.families.deepseek_v41.cuda.graph import GraphRunner

        self.runner = GraphRunner(self.model, self.drafts + 1) if self.graphs_on else None    # 109-111
        self.model.rope_cap = self.limit + self.max_rows + 8                                   # 115
        self.eos = (int(json.loads((self.model_dir / "config.json").read_text()).get("eos_token_id", 1)),)
        self.request = threading.local()
        self.quiet = False
        self.concurrent = True                               # MultiDecoder at any --parallel (spec 3.3)
        sys.setswitchinterval(float(os.environ.get("TF_DS_SWITCH_INTERVAL") or 0.0005))     # 129
        self.multi = StageMulti(self, slots=int(agree["slots"]), cap=int(agree["cap"]), plan=P, mv=self.mv,
                                io=self.io, gate=self.gate, role=self.role)
        mu = self.multi
        self.sc = self.model.pool_view(mu.pool, 0, 0, mu.extents.total)                       # 133-134
        self.dc = mu.slots[0].dc
        if E.WARM:
            from tensorfold.families.deepseek_v41.cuda.graph import BUCKET_MIN

            self.stage_warm()
            buckets = capacity.warm_buckets(mu.cap, BUCKET_MIN)
            if buckets is None:
                mu.warm()                                    # round graphs (and the leader's drafter graphs)
            else:
                mu.warm(buckets)                             # TFPP_DS41_WARM_ALL_BUCKETS=1: every bucket of the pool
            late = capacity.lazy_buckets(mu.cap, BUCKET_MIN, buckets)
            if late:
                log(f"stage {P.index}: the round graphs of buckets {late} are captured while serving, in memory "
                    f"READY's check does not see ({capacity.WARM_ALL}=1 on every stage captures them now)")
        fp32_gemms()

    def stage_warm(self) -> None:
        """F/engine.py:191-225's concurrent branch, each stage alone (spec 11.3): the prompt kernels of every warm-up
        length on slot 0 with synthetic ACT frames (the requests rewrite every row they read)."""

        t0 = time.perf_counter()
        ids = [1000 + (i * 7919) % 60000 for i in range(max(E.WARM_LENGTHS))]
        self.io.local = True
        self.quiet = True
        try:
            for n in E.WARM_LENGTHS:
                if n + 8 <= self.limit:
                    out = self.prefill(self.sc, self.dc if self.plan.last else None, ids[:n])
                    if self.plan.last and out is not None:
                        self._sample(out, [n], None)
        finally:
            self.io.local = False
            self.quiet = False
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"stage {self.plan.index}: warm-up {len(E.WARM_LENGTHS)} prompt lengths {time.perf_counter() - t0:.1f}s, "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB")

    def built_info(self) -> dict:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mu = self.multi
        graphs = sorted([int(r), int(b)] for (r, b) in (mu.runner.graphs or {}))
        P = self.plan
        cuts = sorted(i for i in P.engram_owned if P.lo < i < P.hi)
        return {"layers": [P.lo, P.hi], "graphs": graphs, "stretches": [P.lo, *cuts, P.hi],
                "switches_sha": capacity.switches_sha(), "max_allocated": int(torch.cuda.max_memory_allocated()),
                "allocated": int(torch.cuda.memory_allocated()), "free_after": int(torch.cuda.mem_get_info()[0]),
                "engram_aio": None if self.model.engram is None else bool(self.model.engram.decode_aio()),
                "staging": int(getattr(self, "staging_bytes", 0)), "bufs": int(mu.runner.bufs.nbytes())}


class FollowerEngine(StageEngine):
    pass


class PipeDsEngine(StageEngine):
    """The leader: the fork's request path (Scheduler -> MultiDecoder) with the chain behind PipeOps."""

    def start(self) -> None:
        mu = self.multi
        self.ops = None
        if self.up_link is not None:
            self.ops = PipeOps(mu, self.up_link, timeout=self.io_timeout)
            self.io.ops = self.ops
            mu.link = self.ops
            mu.watch = PipeWatch(self.ops, self.up_link)
        from tensorfold.cuda.scheduler import Scheduler

        self.scheduler = Scheduler(mu, max_streams=self.parallel)
        log(f"--parallel {self.parallel}: {self.parallel} streams share one window of {mu.extents.total} tokens")

    @property
    def broken(self) -> str | None:
        w = getattr(self.multi, "watch", None) if self.multi is not None else None
        return None if w is None else w.broken

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None, vision=None, **kw):
        if self.broken:
            raise RuntimeError(f"tfpp-ds41: the chain broke ({self.broken}); restart every stage")
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft, constraint, vision, **kw)

    def quit(self) -> None:
        """QUIT up the chain (each follower forwards it, exits 0); nothing once it broke."""

        mu = self.multi
        if self.scheduler is not None:              # requests still running: QUIT after them (30 s at most)
            closer = threading.Thread(target=self.scheduler.close, daemon=True)
            closer.start()
            closer.join(30.0)
        watch = getattr(mu, "watch", None)
        if watch is not None:
            watch.closing = True
        ops = getattr(self, "ops", None)
        if ops is not None and not self.broken:
            try:
                ops.quit()
            except Exception as exc:  # noqa: BLE001
                log(f"QUIT not sent: {exc}")

    def self_check(self) -> None:
        self_check(self)


def self_check(eng) -> None:
    """The chain's first requests (spec 11.4): a 33-token prompt, 8 tokens, serially and drafted: the same tokens
    (the fork's exactness: drafted output equals serial output, F/engine.py:1-6). ref.py runs it too, so the two runs'
    histories (stream ids, slots, cache rows) stay alike for the traces."""

    import numpy as np

    rng = np.random.default_rng(20261023)
    prompt = [int(t) for t in rng.integers(1000, 60000, 33)]

    def run(draft: bool) -> list[int]:
        got: list[int] = []
        eng.request.stop_eos = True
        t = time.perf_counter()
        eng.generate(prompt, 8, None, lambda toks: (got.extend(int(x) for x in toks), False)[1], draft=draft)
        log(f"self-check ({'drafted' if draft else 'serial'}): {got} in {time.perf_counter() - t:.2f}s")
        return got

    serial = run(False)
    drafted = run(True) if eng.drafter is not None else serial
    if serial != drafted:
        raise RuntimeError(f"self-check: the drafted reply {drafted} differs from the serial one {serial}")
