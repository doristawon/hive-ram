"""Admission for the whole chain (spec 11.3): one context window and one kept-prompt budget every stage can hold.

MIA's ``capacity.admit`` does not fit a pipeline stage: its staging allowance is three of the largest layers (10.47 GiB
at world=1, TF/cuda/capacity.py:137), it opens every shard's header (:89-98), and under WSL its MemAvailable figure
is the VM's (:150-182). Each stage instead measures what its GPU has left once its weights are loaded (CAP), and the
leader, which knows the text config and every stage's range, sizes each stage's caches and buffers with the
package's own MLA geometry (G/engine.py:634-649) restricted to that range, picks the window with the package's
``make_plan`` / ``choose`` (TF/cuda/capacity.py:198-252), and turns its kept-prompt budget (TF_GLM_CACHE_GIB, in
the leader's bytes) into what every stage can hold. The arithmetic below runs on any CPU (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

GIB = 2 ** 30


@dataclass(frozen=True)
class GeoFlags:
    """The module switches the MIA geometry takes (read from the imported modules on a GPU host: ``flags_now``)."""

    latent: bool = True
    onepass: bool = True
    exl3_prompt: bool = True
    prompt_split_k: bool = False     # an EXL3 checkpoint's BF16 / FP8 prompt projections (False under DENSE=q4)
    kv: str = "fp8"
    max_rows: int = 16               # engine.MAX_ROWS
    dense_capacity: int = 2560       # engine.DENSE_CAPACITY
    multi_window: int = 32           # engine.multi_window()


def flags_now(cfg_quant: str) -> GeoFlags:
    from tensorfold.families.glm5_next.cuda import LATENT, engine as E
    from tensorfold.families.glm5_next.cuda.exl3_mm import prompt_kernels
    from tensorfold.families.glm5_next.cuda.kv8 import kv_kind
    from tensorfold.families.glm5_next.cuda.latent import SPARSE_ONEPASS
    from tensorfold.families.glm5_next.cuda.qmm import dense_kind

    return GeoFlags(latent=bool(LATENT), onepass=bool(SPARSE_ONEPASS), exl3_prompt=bool(prompt_kernels()),
                    prompt_split_k=cfg_quant == "exl3" and dense_kind() != "q4", kv=kv_kind(),
                    max_rows=E.MAX_ROWS, dense_capacity=E.DENSE_CAPACITY, multi_window=E.multi_window())


def sub_config(text: dict, a: int, b: int, *, mtp: bool) -> dict:
    """The text config of layers [a, b) (and the MTP layer when ``mtp``): what the geometry counts per layer."""

    return dict(text, layer_types=list(text["layer_types"][a:b]), num_nextn_predict_layers=1 if mtp else 0)


def layer_kinds(text: dict, a: int, b: int) -> tuple[int, int]:
    """(KDA layers, DSA layers) of [a, b)."""

    kinds = list(text["layer_types"][a:b])
    kda = sum(k == "linear_attention" for k in kinds)
    return kda, len(kinds) - kda


def stage_geometry(text: dict, a: int, b: int, *, last: bool, mtp: bool, parallel: int, prefill_rows: int,
                   flags: GeoFlags, fixed: int = 0, draft=None):
    """The MIA startup geometry (G/engine.py:634-651) of one stage at world=1: its layers' caches and states, the
    decode and prompt buffers (a 1-row head stand-in short of the last stage), more stream slots under --parallel;
    ``fixed``: resident bytes besides (the leader: the vision tower and its workspace, DFlash2's weights);
    ``draft``: DFlash2's context geometry (a ``capacity.Geometry``, the leader)."""

    from tensorfold.cuda.capacity import Geometry
    from tensorfold.cuda.geometry import MLA_DECODE_ROWS, mla_geometry, mla_ring_bytes
    from tensorfold.families.glm5_next.cuda.engine import slot_bytes

    sub = sub_config(text, a, b, mtp=mtp)

    def share(family: str, total: int) -> int:
        return 1 if (family == "vocab" and not last) else int(total)

    window = flags.multi_window
    decode_rows = max(MLA_DECODE_ROWS, flags.max_rows + window) if parallel > 1 else MLA_DECODE_ROWS
    g = mla_geometry(sub, 1, flags.max_rows, share=share, minimum_slots=flags.dense_capacity, latent=flags.latent,
                     prefill_rows=prefill_rows, mtp=mtp, onepass=flags.onepass, exl3_prompt=flags.exl3_prompt,
                     prompt_split_k=flags.prompt_split_k, kv=flags.kv, decode_rows=decode_rows)
    extra = int(fixed) + (parallel - 1) * (slot_bytes(sub, 1, rows=flags.max_rows) + mla_ring_bytes(sub, prefill_rows))
    per = window if parallel > 1 else 0

    def bytes_at(slots: int) -> int:
        return g.bytes_at(slots) + extra + per * slots + (draft.bytes_at(slots) if draft is not None else 0)

    return Geometry(bytes_at, g.reserve, g.minimum_slots)


def staging_bytes(rows: int, carry_in: int, carry_out: int, *, first: bool, last: bool, hidden: int = 4096,
                  streams: int = 4) -> int:
    """Pinned ACT staging a stage allocates after AGREE (wire.staging_bytes for each link end it has)."""

    from .proto import link_staging

    n = 0
    if not first:
        n += link_staging(rows, carry_in, hidden=hidden, streams=streams)
    if not last:
        n += link_staging(rows, carry_out, hidden=hidden, streams=streams)
    return n


def measure_budget(*, reserve_gib: float, budget_gib: float | None, staging: int) -> dict:
    """CAP: what this GPU has for caches and buffers now that the weights are loaded (free - reserve - the pinned
    staging still to come), or ``--budget-gib`` outright (several stages on one GPU)."""

    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = (int(x) for x in torch.cuda.mem_get_info())
    reserve = int(float(reserve_gib) * GIB)
    budget = int(float(budget_gib) * GIB) if budget_gib is not None else free - reserve - int(staging)
    return {"budget": max(0, budget), "free": free, "total": total, "reserve": reserve, "staging": int(staging),
            "allocated": int(torch.cuda.memory_allocated()), "declared": budget_gib is not None}


def agree(*, native: int, requested: int | None, explicit: bool, budgets: list[int], geos: list,
          kinds: list[tuple[int, int]], cache_bytes_wanted: int, dense_limit: int, max_rows: int,
          dense_capacity: int, make_plan: Callable | None = None, choose: Callable | None = None) -> dict:
    """The chain's window and kept-prompt bytes (the leader last in every list).

    window: each stage's plan by the package's make_plan (weights already loaded: Weights(0, 0)), one window for all by
    its choose (an explicit --context that does not fit refuses with the package's message). Kept prompts (P=1): the
    leader keeps TF_GLM_CACHE_GIB of its own snapshot bytes at most; a stage r holds the same snapshots in at most
    scale_r = max(kda_r / kda_leader, dsa_r / dsa_leader) times those bytes (its KDA states and DSA rows), so the
    budget is the least of spare_r / scale_r (the leader's own spare included, scale 1)."""

    if make_plan is None or choose is None:
        from tensorfold.cuda.capacity import Weights, choose as _choose, make_plan as _make_plan

        make_plan, choose = make_plan or _make_plan, choose or _choose
        loaded = Weights(0, 0)                  # the weights are on the GPU already: the budget is what is left
    else:
        loaded = None                           # stand-in plans (tests)
    plans = [make_plan(native, requested, explicit, int(b), loaded, g) for b, g in zip(budgets, geos)]
    try:
        window = choose(plans[-1], [p.settings + [p.fitting] for p in plans])
    except ValueError as exc:                   # the package's message says "across the ranks": name the stage
        raise ValueError(f"{exc} {fit_table(budgets, geos, plans, requested)}") from None
    capacity = max(dense_capacity, window + max_rows)
    spare = [int(b) - int(g.needed(window)) for b, g in zip(budgets, geos)]
    kda_l, dsa_l = kinds[-1]
    scale = []
    for r, (kda, dsa) in enumerate(kinds):
        if r == len(kinds) - 1:
            scale.append(1.0)
        elif kda_l and dsa_l:
            scale.append(max(kda / kda_l, dsa / dsa_l))
        else:                                   # a leader without one of the kinds: per-token cache bytes instead
            def per_token(g) -> int:
                return max(1, int(g.bytes_at(capacity + 65536)) - int(g.bytes_at(capacity)))
            scale.append(per_token(geos[r]) / per_token(geos[-1]))
    room = min(s / c if c > 0 else float("inf") for s, c in zip(spare, scale))
    cache_bytes = max(0, min(int(cache_bytes_wanted), int(room)))
    return {"window": int(window), "capacity": int(capacity), "long_context": window > dense_limit,
            "cache_bytes": int(cache_bytes), "scale": [round(s, 4) for s in scale], "spare": spare,
            "fitting": [int(p.fitting) for p in plans], "largest": min(int(p.largest) for p in plans),
            "needed": [int(g.needed(window)) for g in geos]}


def fit_table(budgets: list[int], geos: list, plans: list, requested: int | None) -> str:
    """Each stage's budget, what it needs at the requested window and the window it fits (the leader last), and the
    stage that binds: what a refused admission is told, so the operator knows which node to free."""

    rows = []
    for i, (b, g, p) in enumerate(zip(budgets, geos, plans)):
        need = ""
        if requested:
            try:
                need = f", needs {int(g.needed(int(requested))) / GIB:.2f} GiB at {int(requested)} tokens"
            except Exception:  # noqa: BLE001  (a geometry that cannot say: leave it out)
                need = ""
        who = "the leader" if i == len(plans) - 1 else f"stage {i}"
        rows.append(f"{who}: budget {int(b) / GIB:.2f} GiB{need}, fits {int(p.fitting)} tokens")
    least = min(range(len(plans)), key=lambda i: int(plans[i].fitting)) if plans else None
    tail = "" if least is None else ("; the least: " + ("the leader" if least == len(plans) - 1 else f"stage {least}"))
    return "Per stage: " + "; ".join(rows) + tail + "."


def kept_room(caps: list[dict], built: list[dict], scale: list[float], cache_bytes: int) -> tuple[int, list[str]]:
    """Kept prompts against what each stage really has left after its build (spec 11.3 checks the budget once, at
    CAP; the reserve is then the only margin for what the geometry leaves out). A stage that measured its own budget
    (no --budget-gib) keeps half its reserve free at all times: kept prompts get at most (free_after - reserve / 2)
    / scale_r, and a stage with less than half its reserve free right after the build refuses the start. Stages
    given --budget-gib (several on one GPU: free memory is the whole GPU's) are not checked.
    Returns (cache_bytes, refusals)."""

    refusals: list[str] = []
    for cap, b, s in zip(caps, built, scale):
        if cap.get("declared") or "free_after" not in b:
            continue
        floor = int(cap.get("reserve", 0)) // 2
        room = int(b["free_after"]) - floor
        if room < 0:
            refusals.append(f"stage {b.get('layers')}: {b['free_after'] / 2**30:.2f} GiB free after its build, less "
                            f"than half its {int(cap.get('reserve', 0)) / 2**30:.1f} GiB reserve (lower --context, "
                            "or raise --reserve-gib on that stage so the window leaves more)")
        elif s > 0:
            cache_bytes = min(cache_bytes, int(room / s))
    return max(0, int(cache_bytes)), refusals


def plan_receipt(agreed: dict, budget: int) -> dict:
    """The ``capacity_plan`` dict GlmEngine keeps (TF/cuda/server.py:148 reads largest_window)."""

    return {"context_window": agreed["window"], "cache_slots": agreed["capacity"],
            "largest_window": agreed["largest"], "budget_bytes": int(budget),
            "cache_workspace_bytes_estimate": agreed["needed"][-1], "kept_bytes": agreed["cache_bytes"],
            "stage_spare_bytes": agreed["spare"], "stage_scale": agreed["scale"]}
