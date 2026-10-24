"""tfpp-glm v2 admission: the shared pool and the kept-prompt count of a --parallel chain (spec v2 section 10).

MIA's --parallel startup (G/engine.py:735-761) turns TF_GLM_CACHE_GIB (what the 262,144-token window leaves beyond the
estimate, at most the asked bytes) into ONE pool of per-token cache rows that every stream and every kept prompt shares,
plus a count of kept prompt states reserved outside the pool. A pipeline stage holds only its layers' share of each, so
the count and the pool must fit the tightest stage: this module does that arithmetic on the leader (no torch: the MIA
helpers it borrows import without it) and the result travels to the stages in AGREE.

  per stage r:   per_token_r = ceil((bytes_at(cap + 65536) - bytes_at(cap)) / 65536)       (G/engine.py:743)
                 state_r     = slot_bytes(layers of r, 1, rows=0) // 2 + 22 MiB             (G/engine.py:747)
  cache   = max(0, min(TF_GLM_CACHE_GIB, spare_leader))
  entries = kept_entries(TF_GLM_CACHE_ENTRIES, TF_GLM_CACHE_SHARE_PCT, cache, max_r state_r)
  extra   = (cache - entries * state_leader) // per_token_leader, then, for every stage,
            min(extra, (spare_r - entries * state_r) // per_token_r)    (a stage that cannot hold the entries: refused)
  pool    = (cap + extra) // 2048 * 2048      limit = min(window, pool - MAX_ROWS)               (:754-759)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

ALIGN = 2048                      # G/pool.py:24: extents (and the pool) start and end on multiples of this
STATE_FIXED = 22 << 20            # G/engine.py:747: a kept state's fixed cost beside its KDA state


class PoolRefused(ValueError):
    """The chain cannot hold the asked pool or kept states (names the stage)."""


@dataclass
class PoolPlan:
    rows: int                     # the pool's tokens on every stage
    entries: int                  # kept prompt states
    limit: int                    # contexts the engine serves: min(window, rows - max_rows)
    extra: int                    # tokens beyond the engine's own cache slots
    cap: int                      # the engine's own cache slots (capacity)
    cache: int                    # the leader's kept-prompt bytes (TF_GLM_CACHE_GIB, clamped)
    per_token: list[int] = field(default_factory=list)       # per stage
    state: list[int] = field(default_factory=list)           # per stage: one kept state's bytes
    spare: list[int] = field(default_factory=list)           # per stage
    note: str = ""

    def describe(self) -> str:
        return f"pool {self.rows} tokens, {self.entries} kept entries, limit {self.limit}"


def draft_window_extra(drafter_cfg: dict | None) -> int:
    """What a kept state of a DFlash2-drafted stream holds more on a stage that carries the WHOLE draft model (the
    leader, world=1) than MIA's 22 MiB, which is a TP2 rank's share: ``draft_window`` copies the ring window (the last
    ``sliding_window`` rows of every draft layer's keys and values, G/decode.py:450-453; G/multi.py:311-317), and a TP2
    rank holds half of the KV heads. Zero without a drafter or a ring (no sliding window)."""

    if not drafter_cfg:
        return 0
    rows = int(drafter_cfg.get("sliding_window") or 0)
    if rows <= 0:
        return 0
    layers, heads, hd = (int(drafter_cfg[k]) for k in ("num_hidden_layers", "num_key_value_heads", "head_dim"))
    full = 2 * layers * heads * rows * hd * 2                       # K and V, bf16
    rank0 = 2 * layers * -(-heads // 2) * rows * hd * 2             # the TP2 rank with the most heads
    return max(0, full - rank0)


def stage_states(text: dict, ranges: Sequence[Sequence[int]], slot_bytes: Callable | None = None,
                 drafter_cfg: dict | None = None) -> list[int]:
    """One kept state's bytes on each stage: its KDA layers' recurrent state (half of ``slot_bytes``' two parities)
    and conv window, plus the fixed 22 MiB (the indexer rings' tail and DFlash2's window copy of a TP2 rank). The last
    stage (the leader) holds the whole drafter: ``draft_window_extra`` more (``drafter_cfg``: the draft model's
    config; None without drafts)."""

    from .capacity import sub_config

    if slot_bytes is None:
        from tensorfold.families.glm5_next.cuda.engine import slot_bytes
    out = [int(slot_bytes(sub_config(text, a, b, mtp=False), 1, rows=0)) // 2 + STATE_FIXED for a, b in ranges]
    if out:
        out[-1] += draft_window_extra(drafter_cfg)
    return out


def per_token_bytes(geo, cap: int) -> int:
    """The bytes one more pool token costs a stage with geometry ``geo`` (G/engine.py:743)."""

    return max(1, -(-(int(geo.bytes_at(cap + 65536)) - int(geo.bytes_at(cap))) // 65536))


def pool_plan(*, window: int, cap: int, geos: Sequence, spare: Sequence[int], states: Sequence[int],
              cache_wanted: int, entries_cfg: int = 8, share_pct: float = 0.0, max_rows: int = 16,
              pool_rows: int | None = None, kept_entries: int | None = None,
              names: Sequence[str] | None = None, kept_entries_fn: Callable | None = None) -> PoolPlan:
    """The pool and the kept-prompt count for the chain (the leader is the last stage in every list).

    ``window``/``cap``: the agreed context window and cache slots; ``geos``: ``capacity.stage_geometry`` of each stage
    (built with parallel=P); ``spare``: each stage's budget minus what the window needs (``agree``'s "spare");
    ``states``: ``stage_states``. ``pool_rows`` / ``kept_entries`` override the computation (tests: a pool and a count
    that every cut and the one-process reference can hold alike); each stage must still hold them or the start is
    refused, naming the stage."""

    n = len(geos)
    if not (n == len(spare) == len(states)) or n == 0:
        raise ValueError("pool_plan: geos, spare and states must be lists of one entry per stage")
    names = list(names) if names is not None else [f"stage {i}" for i in range(n)]
    if kept_entries_fn is None:
        from tensorfold.families.glm5_next.cuda.engine import kept_entries as kept_entries_fn
    per = [per_token_bytes(g, cap) for g in geos]
    cache = max(0, min(int(cache_wanted), int(spare[-1])))
    note = ""
    if kept_entries is not None:
        entries = int(kept_entries)
        if entries < 1:
            raise PoolRefused(f"--kept-entries {entries}: at least 1")
    else:
        entries, note = kept_entries_fn(int(entries_cfg), float(share_pct), cache, max(states))
    for r in range(n):
        room = int(spare[r]) - entries * int(states[r])
        if room < 0:
            raise PoolRefused(f"{names[r]}: {entries} kept prompt states ({entries * states[r] / 2 ** 30:.2f} GiB) do "
                              f"not fit its {spare[r] / 2 ** 30:.2f} GiB of spare memory beside the "
                              f"{window}-token window (lower TF_GLM_CACHE_ENTRIES or --context)")
    if pool_rows is not None:
        rows = int(pool_rows)
        if rows < ALIGN or rows % ALIGN:
            raise PoolRefused(f"--pool-rows {rows}: a multiple of {ALIGN}, at least {ALIGN}")
        extra = max(0, rows - cap)
        for r in range(n):
            need = extra * per[r] + entries * int(states[r])
            if need > int(spare[r]):
                raise PoolRefused(f"{names[r]}: a pool of {rows} tokens ({extra} past the window's slots, "
                                  f"{extra * per[r] / 2 ** 30:.2f} GiB) and {entries} kept states need "
                                  f"{need / 2 ** 30:.2f} GiB, it has {spare[r] / 2 ** 30:.2f} GiB spare")
    else:
        extra = max(0, cache - entries * int(states[-1])) // per[-1]
        for r in range(n):
            extra = min(extra, (int(spare[r]) - entries * int(states[r])) // per[r])
        rows = (cap + extra) // ALIGN * ALIGN
        if rows < ALIGN:
            raise PoolRefused(f"--parallel: a pool of {rows} tokens is less than one {ALIGN}-token extent; free "
                              "memory or lower --context")
    limit = min(int(window), rows - int(max_rows))
    if limit < 1:
        raise PoolRefused(f"a pool of {rows} tokens leaves no context past the {max_rows}-row windows")
    return PoolPlan(rows=rows, entries=entries, limit=limit, extra=extra, cap=cap, cache=cache, per_token=per,
                    state=[int(s) for s in states], spare=[int(s) for s in spare], note=note)


def kept_room_v2(caps: Sequence[dict], built: Sequence[dict], entries: int, states: Sequence[int]) -> list[str]:
    """Refusals after the build (spec v2 10.1): a stage that measured its own budget (no --budget-gib) keeps half its
    reserve free at all times, so its kept states (``entries`` of ``states[r]`` bytes, outside the pool) must fit what
    it has left. The count is never lowered here: it travelled in AGREE and every stage must agree on it."""

    refusals: list[str] = []
    for cap, b, state in zip(caps, built, states):
        if cap.get("declared") or "free_after" not in b:
            continue
        floor = int(cap.get("reserve", 0)) // 2
        room = int(b["free_after"]) - floor
        need = int(entries) * int(state)
        if room < need:
            refusals.append(f"stage {b.get('layers')}: {b['free_after'] / 2 ** 30:.2f} GiB free after its build, "
                            f"half its {int(cap.get('reserve', 0)) / 2 ** 30:.1f} GiB reserve kept free, leaves "
                            f"{max(room, 0) / 2 ** 30:.2f} GiB for {entries} kept states of "
                            f"{state / 2 ** 20:.0f} MiB (lower TF_GLM_CACHE_ENTRIES / --context, or raise --reserve-gib "
                            "on that stage)")
    return refusals
