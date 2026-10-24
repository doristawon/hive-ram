"""tfpp-glm v2 startup frames (spec v2 section 5.5): AGREE's v2 values, READY's BUILT keys and their checks.

The handshake is v1's (HELLO -> GO -> CAP -> AGREE -> READY, ``startup``); for P > 1 AGREE carries the pool's rows and the
kept-state count in the keys v1 already has (``pool_rows``, ``entries``; ``cache_bytes`` 0: the kept prompts live in the
pool) and three more at the end (``multi_window``, ``limit``, ``groups``), and each stage's BUILT adds what a --parallel
build holds (the batched verify graphs, the pool, the slots). Nothing here needs torch except ``built_info_v2``'s call
of v1's ``built_info``.
"""

from __future__ import annotations

from typing import Sequence

from . import proto

AGREE_V2_KEYS = ("multi_window", "limit", "groups")
BUILT_V2_KEYS = ("verify_graphs", "pool_rows", "slots", "multi_window", "segs")


def agree_message_v2(base: dict, plan, *, multi_window: int, groups: int = 1) -> dict:
    """v1's AGREE message with the pool's values and the three v2 keys appended (the keys and their order are v1's)."""

    msg = dict(base)
    msg["pool_rows"] = int(plan.rows)
    msg["entries"] = int(plan.entries)
    msg["cache_bytes"] = 0
    msg["multi_window"] = int(multi_window)
    msg["limit"] = int(plan.limit)
    msg["groups"] = int(groups)
    return msg


def check_agree(go: dict, agree: dict, *, multi_window: int) -> None:
    """A follower's checks of the leader's AGREE for P > 1 (``multi_window``: what this process's
    ``engine.multi_window()`` read from its own env); a ProtocolError names what differs."""

    problems = []
    if int(agree.get("parallel", 0)) != int(go.get("P", 0)):
        problems.append(f"parallel {agree.get('parallel')!r}, GO said {go.get('P')!r}")
    rows = agree.get("pool_rows")
    if not isinstance(rows, int) or rows < 2048 or rows % 2048:
        problems.append(f"pool_rows {rows!r}: a multiple of 2048, at least 2048")
    if int(agree.get("multi_window", -1)) != int(multi_window):
        problems.append(f"multi_window {agree.get('multi_window')!r}, this stage's TF_GLM_MULTI_WINDOW is {multi_window}")
    if int(agree.get("entries", 0)) < 1:
        problems.append(f"entries {agree.get('entries')!r}: at least 1")
    if int(agree.get("limit", 0)) < 1 or int(agree.get("limit", 0)) > int(agree.get("window", 0)):
        problems.append(f"limit {agree.get('limit')!r} past window {agree.get('window')!r}")
    if int(agree.get("groups", -1)) != 1:
        problems.append(f"groups {agree.get('groups')!r}: tfpp-glm v2.0 runs one group")
    if problems:
        raise proto.ProtocolError("AGREE for --parallel does not fit this stage: " + "; ".join(problems))


def built_info_v2(layers, e, verify, plan_rows: int, multi_window: int, segs: int) -> dict:
    """v1's BUILT (kernel paths, memory) plus the --parallel build: the captured verify windows, the pool, the slots."""

    from .startup import built_info

    info = built_info(layers, e)
    info["verify_graphs"] = sorted(int(r) for r in getattr(verify, "graphs", {}) or {})
    info["pool_rows"] = int(plan_rows)
    info["slots"] = int(e.slots.count)
    info["multi_window"] = int(multi_window)
    info["segs"] = int(segs)
    return info


def compare_built_v2(built: Sequence[dict], v1_compare=None) -> list[str]:
    """v1's BUILT comparison (the leader last; ``v1_compare`` stands in for ``startup.compare_built``, which needs
    torch to import: tests), plus the v2 keys: every stage built the same verify windows on the same pool, slots and
    window."""

    if v1_compare is None:
        from .startup import compare_built as v1_compare
    out = list(v1_compare(list(built)))
    ref = built[-1]
    for b in built[:-1]:
        for k in BUILT_V2_KEYS:
            if b.get(k) != ref.get(k):
                out.append(f"stage {b.get('layers')}: {k} {b.get(k)}, leader {ref.get(k)}")
    return out
