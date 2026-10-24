"""Admission for the chain (spec section 12): each stage's pool, work space and staging against what its GPU has
left once its weights are loaded. The arithmetic runs on any CPU (no torch); ``measure`` is tfpp-glm's
``measure_budget`` (G/capacity.py:104-116: one mem_get_info after the weights, lesson 7).

Pool bytes of a stage (C = cap, S = slots, packed FP4 planes; F/model.py:744-787):
  ring   = own layers * S * (window + 16) * head_dim * 2
  comp   = sum over planes of (C // r + 2) * (head_dim / 2 + head_dim / 16)
  index  = sum over indexer planes of (C // r + 2) * (idx_dim / 2 + idx_dim / 32)
  raw    = own ratio>1 sources * S * 64 * head_dim * 4 * 2
  tokens = 8 C; RoPE tables 512 B a position
  leader: the DSpark rings S * 3 * (window + 16) * head_dim * 2, the Markov rows 256 * V * 2, logits 16 * V * 4,
          taps 16 * 3 * dim * 2
Every stage runs with TF_DS_MEM_FLOOR_GIB=0 and without _memory_ceiling (it reads the WSL VM's meminfo, not the GPU:
F/engine.py:169-189).
"""

from __future__ import annotations

import hashlib
import json
import os

from . import plan as PL

GIB = 2 ** 30
RAW = 64                       # F/model.py:27
RING_EXTRA = 16                # F/model.py:28
ROWS_MAX = 16                  # kernels.ROUND_ROWS (TF_DS_DECODE_ROWS=16)


def pool_bytes(mv: PL.ModelView, P: PL.StagePlan, cap: int, slots: int, *, drafter: bool) -> dict:
    ring = mv.window + RING_EXTRA
    own = P.hi - P.lo
    out = {
        "ring": own * slots * ring * mv.head_dim * 2,
        "comp": sum((cap // mv.ratios[s] + 2) * (mv.head_dim // 2 + mv.head_dim // 16) for s in P.planes),
        "index": sum((cap // mv.ratios[s] + 2) * (mv.idx_dim // 2 + mv.idx_dim // 32) for s in P.index_planes),
        "raw": len(P.own_raw) * slots * RAW * mv.head_dim * 4 * 2,
        "tokens": 8 * cap,
        "rope": 512 * cap,
    }
    if P.last:
        out["logits"] = ROWS_MAX * mv.vocab * 4
        out["taps"] = ROWS_MAX * max(len(mv.taps), 1) * mv.dim * 2
        if drafter:
            out["draft_rings"] = slots * 3 * ring * mv.head_dim * 2
            out["markov"] = 256 * mv.vocab * 2
    out["total"] = sum(out.values())
    return out


def link_staging(mv: PL.ModelView, carry: PL.Carry | None, chunk: int, cap: int, *, f1: bool) -> int:
    """One link end's pinned staging: the largest prompt-chunk ACT through it (a full chunk, every entry of the pool
    visible), each part aligned to 256 bytes (G/wire.py:213). A part past it travels through a host copy instead
    (correct, slower: G/wire.py:333-338)."""

    if carry is None or chunk <= 0:
        return 0
    parts = PL.chunk_parts(mv, carry, max(0, cap - chunk), chunk, max(0, cap - chunk), chunk, f1=f1)
    return sum(-(-(PL.DT_BYTES[d] * _prod(s)) // 256) * 256 for _, d, s in parts)


def _prod(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def stage_staging(mv: PL.ModelView, P: PL.StagePlan, chunk: int, cap: int, *, f1: bool) -> tuple[int, int]:
    """(receive, send) staging bytes of a stage's links."""

    return (link_staging(mv, P.cin, chunk, cap, f1=f1) if not P.first else 0,
            link_staging(mv, P.cout, chunk, cap, f1=f1) if not P.last else 0)


def workspace_bytes() -> int:
    return int(float(os.environ.get("TFPP_DS41_WORKSPACE_GIB") or 4.0) * GIB)


def stage_need(mv: PL.ModelView, P: PL.StagePlan, cap: int, slots: int, *, drafter: bool, chunk: int,
               f1: bool = False) -> dict:
    pool = pool_bytes(mv, P, cap, slots, drafter=drafter)
    rx, tx = stage_staging(mv, P, chunk, cap, f1=f1)
    need = pool["total"] + workspace_bytes() + rx + tx
    return {"pool": pool["total"], "workspace": workspace_bytes(), "staging": rx + tx, "need": int(need),
            "detail": pool}


def measure(*, reserve_gib: float, budget_gib: float | None, staging: int) -> dict:
    import socket

    from tfpp.glm.capacity import measure_budget

    out = measure_budget(reserve_gib=reserve_gib, budget_gib=budget_gib, staging=staging)
    out["host"] = socket.gethostname()                 # check_caps sums the stages that share one GPU
    return out


CONTEXT_BYTES = 512 * 2 ** 20                          # a CUDA context beside its weights and need (stages on one GPU)


def effective_need(c: dict) -> int:
    """What a stage asks of its budget. ``measure_budget`` already took the pinned staging out of a measured budget
    (free - reserve - staging), so staging is not charged twice; a declared --budget-gib is outright, so it is."""

    need = int(c["need"])
    return need if c.get("declared", True) else need - int(c.get("staging", 0))


def check_caps(caps: list[dict]) -> list[str]:
    """Spec 12 step 4: a stage whose need passes its budget refuses the chain (every stage's numbers listed); and the
    stages that share one GPU (the same host: T0c, T1-3) together hold their weights, needs and a CUDA context each
    within that GPU, which no single stage's declared --budget-gib can know."""

    problems = []
    for c in caps:
        if effective_need(c) > int(c["budget"]):
            problems.append(f"stage {c.get('layers')}: needs {effective_need(c) / GIB:.2f} GiB (pool "
                            f"{int(c['pool']) / GIB:.2f}, work space {int(c['workspace']) / GIB:.2f}, staging "
                            f"{int(c['staging']) / GIB:.2f}), its budget is {int(c['budget']) / GIB:.2f} GiB")
    by_host: dict[str, list[dict]] = {}
    for c in caps:
        if c.get("host") and int(c.get("total", 0)) > 0:
            by_host.setdefault(str(c["host"]), []).append(c)
    for host, group in by_host.items():
        if len(group) < 2:
            continue
        total = min(int(c["total"]) for c in group)
        reserve = max(int(c.get("reserve", 0)) for c in group)
        want = sum(int(c.get("weights", 0)) + int(c["need"]) + CONTEXT_BYTES for c in group)
        if want > total - reserve:
            each = "; ".join(f"{c.get('layers')} {(int(c.get('weights', 0)) + int(c['need'])) / GIB:.1f}" for c in group)
            problems.append(f"{len(group)} stages share {host}'s GPU ({total / GIB:.1f} GiB, reserve {reserve / GIB:.1f}): "
                            f"weights + need + a CUDA context each add up to {want / GIB:.2f} GiB ({each}): fewer "
                            "stages on this GPU, or a smaller TFPP_DS41_WORKSPACE_GIB / --context / --parallel")
    if problems:
        problems.append("every stage: " + "; ".join(
            f"{c.get('layers')} weights {int(c.get('weights', 0)) / GIB:.2f} need {int(c['need']) / GIB:.2f} budget "
            f"{int(c['budget']) / GIB:.2f}" for c in caps))
    return problems


def check_built(built: list[dict], *, floor_gib: float = 1.0) -> list[str]:
    """READY's checks (spec 9.2, 12 step 5): one capture set, one switch set, 1 GiB free at least on each stage
    (stages given --budget-gib share a GPU: its free memory is not theirs alone, not checked)."""

    problems = []
    ref = built[-1]
    for b in built[:-1]:
        for k in ("graphs", "switches_sha"):
            if b.get(k) != ref.get(k):
                problems.append(f"stage {b.get('layers')}: {k} {str(b.get(k))[:80]}, leader {str(ref.get(k))[:80]}")
    for b in built:
        if not b.get("declared") and int(b.get("free_after", 1 << 62)) < floor_gib * GIB:
            problems.append(f"stage {b.get('layers')}: {int(b['free_after']) / GIB:.2f} GiB free after its build "
                            f"(at least {floor_gib:g}: lower --context or --parallel)")
    return problems


FORK_WARM_BUCKETS = (1024, 2048, 4096, 8192)          # MultiDecoder.warm's default (F/multi.py:442)
WARM_ALL = "TFPP_DS41_WARM_ALL_BUCKETS"


def _bucket_for(n: int, cap: int, bucket_min: int) -> int:
    b = bucket_min                                     # graph.bucket_for (F/graph.py:24-28)
    while b < n:
        b *= 2
    return min(b, max(cap, bucket_min))


def warm_buckets(cap: int, bucket_min: int, environ=None) -> tuple[int, ...] | None:
    """The buckets the build's ``mu.warm`` captures round graphs for: None (the fork's default, buckets up to 8192) or,
    with TFPP_DS41_WARM_ALL_BUCKETS=1, every bucket of the pool, so READY's memory check (check_built) sees the round
    graphs a long context would otherwise capture while serving. The variable is digested in HELLO: every stage alike,
    one capture set."""

    env = os.environ if environ is None else environ
    if str(env.get(WARM_ALL, "") or "0").strip() != "1":
        return None
    return tuple(PL.buckets(cap, bucket_min))


def lazy_buckets(cap: int, bucket_min: int, warmed=None) -> list[int]:
    """The pool's buckets whose round graphs are not captured by the warm-up (``warmed``: what warm_buckets returned):
    they are captured by the first round that reaches them, while serving, in memory CAP and READY never saw."""

    seen = set()
    for b in (FORK_WARM_BUCKETS if warmed is None else warmed):
        if b > cap:                                    # MultiDecoder.warm stops there (F/multi.py:450)
            break
        seen.add(_bucket_for(b, cap, bucket_min))
    return [b for b in PL.buckets(cap, bucket_min) if b not in seen]


def switches_sha() -> str:
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    return hashlib.sha256(json.dumps(sorted(K.SMALL_SWITCHES.items())).encode()).hexdigest()[:16]
