"""tfpp-glm v2 (--parallel 2..8): the wire ops of the multi-stream chain and their fields, without torch.

v2 keeps v1's frames (``proto.make_header``, a sequence number per link and direction, LINK, QUIT, ERROR) and adds
three control ops, leader -> stage 0, every middle stage forwarding a frame upstream before it handles it (spec v2
section 5):

- ``MULTI``: one sealed MultiDecoder message (G/multi.py:112-121, 625-636) as ``ints:i32[n]``, plus the image rows an
  ADMIT of it brings (``img:<sid>``: bf16 [rows, hidden]; stage 0 keeps them);
- ``WINDOWS``: a round's segments ``[[slot, pos, rows], ...]`` and their token ids (``ids:i32[sum rows]``);
- ``KEEPS``: the rows each segment keeps and where its position ends.

and three ACT kinds (stage 0 -> the leader): ``chunk`` (a FILL's prompt chunk), ``pieces`` (an MFILL's multi-prompt
chunk) and ``round`` (a batched verify window). Every v2 frame carries ``g``, the group of the op (always 0 in v2.0,
so v2.1's grouped pipeline needs no new op or field). ``proto.py`` is not changed: nothing reads its CONTROL / DATA
sets and ``Link.send`` takes any op name.
"""

from __future__ import annotations

from typing import Any, Sequence

V2_VERSION = "2.0"

MULTI, WINDOWS, KEEPS = "MULTI", "WINDOWS", "KEEPS"
CONTROL_V2 = frozenset({MULTI, WINDOWS, KEEPS})

# v1's per-request control ops: a v2 follower refuses them (spec v2 5.3)
V1_ONLY = frozenset({"PREFILL", "STEP", "KEEP", "SNAP", "SAVE", "LOAD", "DROP", "END"})

SEAL_MOD = 2_147_483_647      # multi.SEAL_MOD (G/multi.py:109); the follower checks against the module's own


# -- ACT fields (a header's fields besides op / seq / parts) ----------------------------------------------------------
def act_chunk(g: int, slot: int, pos: int, R: int) -> dict:
    """A FILL's prompt chunk: the lane's slot and its position before the chunk's commit (G/decode.py:616-639)."""

    return {"kind": "chunk", "g": int(g), "slot": int(slot), "pos": int(pos), "R": int(R)}


def act_pieces(g: int, segs: Sequence[Sequence[int]], R: int) -> dict:
    """An MFILL's multi-prompt chunk: ``[[slot, pos, rows], ...]`` in ``PieceMixer.pieces`` order
    (G/multi_prefill.py:110-111), positions before the commit (G/multi_prefill.py:183-184)."""

    return {"kind": "pieces", "g": int(g), "R": int(R), "segs": [[int(a), int(b), int(c)] for a, b, c in segs]}


def act_round(g: int, rid: int, R: int) -> dict:
    """A batched verify window: ``rid`` is the seal number of the MULTI that carried its ROUND."""

    return {"kind": "round", "g": int(g), "rid": int(rid), "R": int(R)}


# -- control fields ---------------------------------------------------------------------------------------------------
def multi_fields(g: int, values: Sequence[int]) -> dict:
    """MULTI's header: ``mseq`` is the seal's sequence number (``values[-2]``), ``n`` the ints' count."""

    if len(values) < 2:
        raise ValueError("a MULTI carries a sealed message (at least its sequence number and checksum)")
    return {"g": int(g), "mseq": int(values[-2]), "n": len(values)}


def windows_fields(g: int, rid: int, segs: Sequence[Sequence[int]]) -> dict:
    return {"g": int(g), "rid": int(rid), "segs": [[int(a), int(b), int(c)] for a, b, c in segs]}


def keeps_fields(g: int, rid: int, keeps: Sequence[int], end: Sequence[int]) -> dict:
    return {"g": int(g), "rid": int(rid), "keeps": [int(k) for k in keeps], "end": [int(e) for e in end]}


def img_part(sid: int) -> str:
    return f"img:{int(sid)}"


def img_sid(name: str) -> int | None:
    """The stream id of an ``img:<sid>`` part name, else None."""

    if not name.startswith("img:"):
        return None
    try:
        return int(name[4:])
    except ValueError:
        return None


# -- checks the follower makes (pure: tests) ---------------------------------------------------------------------------
def check_windows(hdr: dict, n_ids: int, *, received: int, lanes: Sequence[Sequence[int]], rows_max: int,
                  window: int) -> list[list[int]]:
    """WINDOWS against this stage's view (spec v2 5.4 item 3): ``g`` 0, ``rid`` the message being applied, one segment
    per lane of the ROUND with the lane's [slot, pos], 1 .. rows_max rows each, the ids' count their sum, at most the
    window. Returns the segments; raises ValueError naming what differs."""

    segs = hdr.get("segs")
    if hdr.get("g") != 0:
        raise ValueError(f"WINDOWS of group {hdr.get('g')!r} (v2.0: 0)")
    if hdr.get("rid") != received:
        raise ValueError(f"WINDOWS for message {hdr.get('rid')!r}, this stage applies {received}")
    if not isinstance(segs, list) or len(segs) != len(lanes):
        raise ValueError(f"WINDOWS with {len(segs) if isinstance(segs, list) else segs!r} segments for "
                         f"{len(lanes)} lanes")
    if [list(x[:2]) for x in segs] != [list(map(int, l)) for l in lanes]:
        raise ValueError(f"WINDOWS segments {[x[:2] for x in segs]}, this stage's lanes {[list(l) for l in lanes]}")
    if any(not 1 <= int(x[2]) <= rows_max for x in segs):
        raise ValueError(f"WINDOWS rows {[x[2] for x in segs]}: 1 .. {rows_max} each")
    total = sum(int(x[2]) for x in segs)
    if total != n_ids or n_ids > window:
        raise ValueError(f"WINDOWS of {total} rows with {n_ids} ids (the window holds {window})")
    return [[int(a), int(b), int(c)] for a, b, c in segs]


def check_keeps(hdr: dict, *, rid: int, rows: Sequence[int]) -> list[int]:
    """KEEPS for the round ``rid``: one keep per segment, 1 .. its rows."""

    keeps = hdr.get("keeps")
    if hdr.get("g") != 0 or hdr.get("rid") != rid:
        raise ValueError(f"KEEPS g={hdr.get('g')!r} rid={hdr.get('rid')!r} for WINDOWS rid {rid}")
    if not isinstance(keeps, list) or len(keeps) != len(rows):
        raise ValueError(f"KEEPS {keeps!r} for {len(rows)} segments")
    if any(not 1 <= int(k) <= int(r) for k, r in zip(keeps, rows)):
        raise ValueError(f"KEEPS {keeps} of windows of {list(rows)} rows")
    end = hdr.get("end")
    if not isinstance(end, list) or len(end) != len(rows):
        raise ValueError(f"KEEPS end {end!r} for {len(rows)} segments")
    return [int(k) for k in keeps]


def check_act(hdr: dict, fields: dict, parts: set) -> str | None:
    """An ACT header against the fields this stage computed itself and the parts it expects (None: it fits)."""

    if hdr.get("op") not in (None, "ACT"):
        return f"expected ACT, got {hdr.get('op')}"
    bad = [k for k, v in fields.items() if hdr.get(k) != v]
    if bad:
        return "ACT " + ", ".join(f"{k}={hdr.get(k)!r} (expected {fields[k]!r})" for k in bad)
    got = {p[0] for p in hdr.get("parts", [])}
    if got != set(parts):
        return f"ACT parts {sorted(got)}, expected {sorted(parts)}"
    return None


# -- faults (tests) ---------------------------------------------------------------------------------------------------
class Fault:
    """``POINT:N`` (TFPP_GLM_FAULT_V2 on the leader, TFPP_GLM_FAULT_FOLLOWER on a follower): the N-th time the code
    passes POINT it raises (``kill``: SIGKILLs its own process instead; ``POINT:N:stop``: SIGSTOPs it, the test then
    watches the chain's timeouts and lets it go with SIGCONT). Counts only while ``armed`` (the leader arms it after
    its warm-up; a follower's counts from its start, the warm-up's rounds included)."""

    def __init__(self, text: str = "", *, kill: bool = False, armed: bool = False) -> None:
        text = (text or "").strip()
        self.stop = text.endswith(":stop")
        if self.stop:
            text = text[:-len(":stop")]
        self.point, self.at = None, 0
        if text:
            point, _, n = text.rpartition(":")
            if not point or not n.isdecimal() or int(n) < 1:
                raise ValueError(f"fault {text!r}: POINT:N (N >= 1)")
            self.point, self.at = point, int(n)
        self.kill = bool(kill)
        self.armed = bool(armed)
        self.count = 0

    def hit(self, point: str) -> None:
        if self.point is None or point != self.point or not self.armed:
            return
        self.count += 1
        if self.count != self.at:          # the N-th time only: a stopped stage let go with SIGCONT carries on
            return
        if self.stop:
            import os
            import signal

            print(f"[tfpp-glm] injected fault {point}:{self.at}: SIGSTOP", flush=True)
            os.kill(os.getpid(), getattr(signal, "SIGSTOP", signal.SIGTERM))
            return
        if self.kill:
            import os
            import signal

            print(f"[tfpp-glm] injected fault {point}:{self.at}: SIGKILL", flush=True)
            os.kill(os.getpid(), getattr(signal, "SIGKILL", signal.SIGTERM))
        raise RuntimeError(f"tfpp-glm: injected fault at {point}:{self.at}")


# -- measurements (TFPP_GLM_TIMING / TFPP_GLM_TIME_ROWS) ---------------------------------------------------------------
class RoundStats:
    """TFPP_GLM_TIMING=1: how long a stage's rounds take and how much of it they wait for the stage before (a log line
    every ``every`` rounds)."""

    def __init__(self, who: str, every: int = 100) -> None:
        self.who, self.every = who, every
        self.n = self.rows = 0
        self.total = self.wait = 0.0

    def add(self, rows: int, total_s: float, wait_s: float) -> None:
        self.n += 1
        self.rows += rows
        self.total += total_s
        self.wait += wait_s
        if self.n % self.every == 0:
            print(f"[tfpp-glm] timing {self.who}: {self.every} rounds, {self.rows / self.every:.1f} rows, "
                  f"{self.total / self.every * 1e3:.1f} ms a round, {self.wait / self.every * 1e3:.1f} ms of it "
                  "waiting for the stage before", flush=True)
            self.rows = 0
            self.total = self.wait = 0.0


def log_rows(who: str, ms: Sequence[float]) -> None:
    """TFPP_GLM_TIME_ROWS=1: the captured verify window's ms by rows (1 .. its rows)."""

    print(f"[tfpp-glm] {who}: verify window ms by rows: " + " ".join(f"{i + 1}:{v:.2f}" for i, v in enumerate(ms)),
          flush=True)


def describe_ops(values: Sequence[int], names: dict[int, str]) -> list[list[Any]]:
    """[[op name, payload length], ...] of a sealed message's body (trace records; a malformed one: [["?", n]])."""

    out: list[list[Any]] = []
    body = list(values[:-2])
    i = 0
    while i + 2 <= len(body):
        op, n = int(body[i]), int(body[i + 1])
        if n < 0 or i + 2 + n > len(body):
            out.append(["?", len(body) - i])
            return out
        out.append([names.get(op, str(op)), n])
        i += 2 + n
    if i != len(body):
        out.append(["?", len(body) - i])
    return out
