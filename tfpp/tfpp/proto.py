# SPDX-License-Identifier: Apache-2.0
"""The Flash Next tfpp protocol's constants, errors and checks, without torch (CPU-testable).

Every rank checks what it receives before it allocates or launches anything: a frame's op, token ids, row counts and
residual shape (``check_frame``), a downstream reply (``check_reply``), and, once at startup, the whole chain
(``check_chain``: the leader collects every rank's record through the HELLO round trip). Checks that pass for the
legitimate leader and followers never change what they compute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROTO = "tfpp-flashnext/2"      # 1: the unversioned first wire (no link handshake, no HELLO)
OPS = ("hello", "reset", "prefill", "step", "quit", "error")
ROLES = ("leader", "middle", "last")
MSG_MAX = 4000                  # an error frame's message is cut to this many characters


class LinkError(ConnectionError):
    """The link broke: closed, timed out, or the peer could not be reached."""


class ProtocolError(LinkError):
    """The peer broke the protocol: a malformed or out-of-range frame, another protocol, or a refused link.
    Retrying the same peer does not help."""


class RemoteError(RuntimeError):
    """Another rank failed and said why (an ``error`` frame); this rank stops too."""


# -- command line -------------------------------------------------------------------------------------------------
def parse_layers(text: str) -> tuple[int, int]:
    """``"A:B"`` -> (A, B), a non-empty range of non-negative layer indices."""

    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"layers {text!r}: expected FIRST:END, for example 0:32")
    try:
        first, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"layers {text!r}: FIRST and END must be integers") from None
    if not 0 <= first < end:
        raise ValueError(f"layers {text!r}: an empty or negative range (need 0 <= FIRST < END)")
    return first, end


def parse_peer(text: str) -> tuple[str, int]:
    """``"host:port"`` -> (host, port)."""

    host, sep, port = str(text).rpartition(":")
    if not sep or not host:
        raise ValueError(f"peer {text!r}: expected HOST:PORT, for example 10.77.0.1:5201")
    try:
        p = int(port)
    except ValueError:
        raise ValueError(f"peer {text!r}: the port is not a number") from None
    if not 0 < p < 65536:
        raise ValueError(f"peer {text!r}: port {p} out of range")
    return host.strip("[]"), p


def check_range(first: int, end: int, total: int) -> None:
    """A rank's layers [first, end) must be a non-empty range inside the checkpoint's ``total`` layers."""

    if not (isinstance(first, int) and isinstance(end, int) and 0 <= first < end <= total):
        raise ValueError(f"layers {first}:{end} are not a non-empty range inside 0:{total} (the checkpoint has "
                         f"{total} layers)")


def check_role_range(role: str, first: int, end: int, total: int) -> None:
    """The leader embeds, so it starts at layer 0; the last rank applies the head, so it ends at the last layer."""

    check_range(first, end, total)
    if role == "leader" and first != 0:
        raise ValueError(f"the leader embeds the tokens, so its layers start at 0 (got {first}:{end})")
    if role == "last" and end != total:
        raise ValueError(f"the last rank applies the head, so its layers end at {total} (got {first}:{end})")


def config_fingerprint(model_dir: str | Path) -> str | None:
    """sha256 of the checkpoint's config.json, parsed and re-serialised (whitespace and key order do not count), or
    None without one. Ranks must load the same model."""

    p = Path(model_dir) / "config.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    canon = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


# -- frames -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Limits:
    """What one rank's engine can take: token ids below ``vocab``, at most ``rows`` per decode window and
    ``prefill_rows`` per prompt chunk, residual rows of ``width`` (streams x hidden) bf16 values."""

    vocab: int
    rows: int
    prefill_rows: int
    width: int


def _is_int(x: Any) -> bool:
    return type(x) is int        # noqa: E721 (bool is an int subclass and is not a token id)


def check_ids(ids: Any, name: str, vocab: int, lo: int, hi: int) -> list:
    """A list of ``lo``..``hi`` token ids, each in [0, vocab)."""

    if not isinstance(ids, list):
        raise ProtocolError(f"{name}: not a list of token ids")
    if not lo <= len(ids) <= hi:
        raise ProtocolError(f"{name}: {len(ids)} ids, expected {lo} to {hi}")
    if ids and (not all(_is_int(t) for t in ids) or min(ids) < 0 or max(ids) >= vocab):
        raise ProtocolError(f"{name}: token ids must be integers in [0, {vocab})")
    return ids


def _int_field(hdr: dict, name: str, lo: int, hi: int | None) -> int:
    v = hdr.get(name)
    if not _is_int(v) or v < lo or (hi is not None and v > hi):
        raise ProtocolError(f"{hdr.get('op')}: {name}={v!r}, expected an integer in [{lo}, {'' if hi is None else hi}]")
    return v


def check_frame(hdr: dict, shape: Sequence[int] | None, lim: Limits, *, hello_done: bool) -> str:
    """A frame from the rank before this one; returns its op. ``shape``: the received tensor's, or None."""

    op = hdr.get("op")
    if op not in OPS:
        raise ProtocolError(f"unknown op {op!r}")
    if op == "error":
        return op
    if not hello_done and op != "hello":
        raise ProtocolError(f"{op!r} before hello: the rank before this one runs another tfpp version (re-sync "
                            "every node)")
    if op == "hello":
        if hello_done:
            raise ProtocolError("a second hello")
        if hdr.get("proto") != PROTO:
            raise ProtocolError(f"hello for protocol {hdr.get('proto')!r}, this rank speaks {PROTO}")
        return op
    if op in ("reset", "quit"):
        if shape is not None:
            raise ProtocolError(f"{op} with a tensor")
        return op
    cap = lim.prefill_rows if op == "prefill" else lim.rows
    tokens = check_ids(hdr.get("tokens"), f"{op} tokens", lim.vocab, 1, cap)
    if shape is None or tuple(shape) != (len(tokens), lim.width):
        raise ProtocolError(f"{op}: residual rows {None if shape is None else list(shape)}, expected "
                            f"[{len(tokens)}, {lim.width}]")
    _int_field(hdr, "depth", 0, lim.rows - 1)
    _int_field(hdr, "remaining", 0, None)
    conf = hdr.get("conf")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        raise ProtocolError(f"{op}: conf={conf!r} is not a number")
    if op == "prefill":
        check_ids(hdr.get("next"), "prefill next", lim.vocab, 0, len(tokens))
        if not isinstance(hdr.get("final"), bool):
            raise ProtocolError("prefill: final must be true or false")
    return op


def check_reply(r: dict, kind: str, *, vocab: int, rows: int | None = None, depth: int | None = None) -> dict:
    """A reply from the rank after this one to ``kind`` (hello, reset, prefill, step); an ``error`` frame raises
    RemoteError. ``rows``: the window just sent (step); ``depth``: the drafts asked for."""

    if r.get("op") == "error":
        raise RemoteError(f"{r.get('where') or 'a later rank'}: {r.get('msg')}")
    if kind == "hello":
        ranks = r.get("ranks")
        if r.get("op") != "hello" or not isinstance(ranks, list) or not all(isinstance(x, dict) for x in ranks):
            raise ProtocolError("the reply to hello carries no rank records")
        return r
    if kind == "reset":
        if r.get("ok") != 1:
            raise ProtocolError(f"the reply to reset is {r!r}")
        return r
    if kind == "prefill":
        check_ids([r.get("first")], "first token", vocab, 1, 1)
    elif kind == "step":
        keep = r.get("keep")
        if not _is_int(keep) or not 1 <= keep <= (rows or 1):
            raise ProtocolError(f"step reply: keep={keep!r} for a window of {rows} rows")
        check_ids(r.get("sampled"), "step sampled", vocab, keep, keep)
    else:
        raise ValueError(f"unknown reply kind {kind!r}")
    check_ids(r.get("drafts"), f"{kind} drafts", vocab, 0, depth if depth is not None else 1 << 30)
    if not isinstance(r.get("done", False), bool):
        raise ProtocolError(f"{kind} reply: done must be true or false")
    return r


def done_after_prefill(remaining: int, first: int, eos) -> bool:
    """Whether the leader stops after the prompt's first token (it generates while len(out) < count and the last
    token is not an end token; ``remaining`` = count - 1)."""

    return remaining <= 0 or first in eos


def done_after_step(remaining: int, kept: Sequence[int], eos) -> bool:
    """Whether the leader stops after this step's kept tokens (``remaining`` = count - len(out) before them)."""

    return len(kept) >= remaining or (bool(kept) and kept[-1] in eos)


def check_request(prompt: Any, count: Any, depth: Any, confidence: Any, *, vocab: int, capacity: int,
                  rows: int) -> None:
    """Leader.generate's arguments, before anything is sent: an empty prompt would leave every rank waiting."""

    if not isinstance(prompt, (list, tuple)) or not prompt:
        raise ValueError("prefill requires at least one token")
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in prompt) or min(prompt) < 0 or \
            max(prompt) >= vocab:
        raise ValueError(f"prompt token ids must be integers in [0, {vocab})")
    if len(prompt) > capacity:
        raise ValueError(f"a {len(prompt)}-token prompt is past the cache capacity {capacity}")
    if not _is_int(count) or count < 1:
        raise ValueError(f"count {count!r}: generate at least one token")
    if not _is_int(depth) or depth < 0:
        raise ValueError(f"depth {depth!r}: expected an integer >= 0")
    if depth + 1 > rows:
        raise ValueError(f"depth {depth} needs windows of {depth + 1} rows; the smallest rank holds {rows} "
                         "(start every rank with --depth >= this one)")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence {confidence!r} is not a number")


# -- the chain (HELLO) --------------------------------------------------------------------------------------------
SAME = ("proto", "total", "prefill_rows", "config", "around_one", "tensorfold")


def rank_name(i: int, r: dict) -> str:
    return f"rank {i} ({r.get('role', '?')}, layers {r.get('first', '?')}:{r.get('end', '?')})"


def check_chain(ranks: Sequence[dict], depth: int) -> list[str]:
    """Every rank's record, leader first -> the problems that stop the pipeline (empty: it is one model, split once).

    The layer ranges must be contiguous from 0 to the last layer (a gap, an overlap or two ranks on the whole model
    give plausible wrong tokens); every rank must load the same checkpoint with the same norm convention, prompt
    chunk and TensorFold; every follower's cache must hold at least the leader's; every rank's decode buffers must
    hold the leader's windows (``depth`` + 1 rows); and when the leader drafts, the last rank needs the MTP head."""

    if not ranks:
        return ["no ranks"]
    problems: list[str] = []
    lead = ranks[0]
    if lead.get("role") != "leader":
        problems.append(f"{rank_name(0, lead)}: the first rank must be the leader")
    if ranks[-1].get("role") != "last" or len(ranks) < 2:
        problems.append(f"the chain ends at {rank_name(len(ranks) - 1, ranks[-1])}, not at a last rank")
    for i, r in enumerate(ranks[1:-1], start=1):
        if r.get("role") != "middle":
            problems.append(f"{rank_name(i, r)}: a rank between the leader and the last rank must be a middle rank")
    at = 0
    for i, r in enumerate(ranks):
        first, end, total = r.get("first"), r.get("end"), r.get("total")
        if not (_is_int(first) and _is_int(end) and _is_int(total) and 0 <= first < end <= total):
            problems.append(f"{rank_name(i, r)}: not a non-empty range inside 0:{total}")
            at = end if _is_int(end) else at
            continue
        if first != at:
            what = "a gap" if first > at else "an overlap"
            problems.append(f"{rank_name(i, r)}: starts at layer {first}, but the ranks before it end at {at} "
                            f"({what}; the ranges must be contiguous from 0)")
        at = end
    total = lead.get("total")
    if _is_int(total) and at != total:
        problems.append(f"the chain ends at layer {at}; the checkpoint has {total} layers")
    for i, r in enumerate(ranks[1:], start=1):
        for key in SAME:
            if r.get(key) != lead.get(key):
                problems.append(f"{rank_name(i, r)}: {key} {r.get(key)!r}, the leader has {lead.get(key)!r}")
        cap, lcap = r.get("capacity"), lead.get("capacity")
        if not _is_int(cap) or not _is_int(lcap) or cap < lcap:
            problems.append(f"{rank_name(i, r)}: cache capacity {cap!r}, smaller than the leader's {lcap!r} (start "
                            "every rank with the same --capacity)")
    for i, r in enumerate(ranks):
        rows = r.get("rows")
        if not _is_int(rows) or rows < depth + 1:
            problems.append(f"{rank_name(i, r)}: decode buffers of {rows!r} rows; the leader's --depth {depth} "
                            f"needs {depth + 1} (start every rank with the same --depth)")
    if depth > 0 and len(ranks) >= 2 and ranks[-1].get("role") == "last" and not ranks[-1].get("mtp"):
        problems.append(f"{rank_name(len(ranks) - 1, ranks[-1])}: no MTP head, but the leader drafts (--depth "
                        f"{depth}); start the last rank with --depth {depth} too")
    return problems
