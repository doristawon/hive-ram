"""tfpp-glm's wire protocol without torch: frame headers, the ops, stage descriptions and DFlash2 tap routing.

A frame is ``[u32 header length][header: UTF-8 JSON][payload]``; the header names the op, the sender's sequence number
on that link and direction, the op's fields, and the payload's parts in order (``[name, dtype, shape]``), each
``prod(shape) * itemsize`` bytes. Control ops travel from the leader (the last stage) towards stage 0, every follower
forwarding a frame upstream before it handles it; data ops travel from stage 0 towards the leader.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

U32 = struct.Struct("<I")
MAX_HEADER = 64 << 20

# -- ops ---------------------------------------------------------------------------------------------------------
LINK = "LINK"             # the first frame each end of a link sends (seq -1, outside the numbering)
# control: leader -> stage 0
GO, AGREE = "GO", "AGREE"
PREFILL, STEP, KEEP = "PREFILL", "STEP", "KEEP"
SNAP, SAVE, LOAD, DROP = "SNAP", "SAVE", "LOAD", "DROP"
END, QUIT = "END", "QUIT"
# data: stage 0 -> leader
HELLO, CAP, READY, ACT = "HELLO", "CAP", "READY", "ACT"
ERROR = "ERROR"           # data direction (a follower's failure); the leader also sends it upstream during startup

CONTROL = frozenset({GO, AGREE, PREFILL, STEP, KEEP, SNAP, SAVE, LOAD, DROP, END, QUIT, ERROR})
DATA = frozenset({HELLO, CAP, READY, ACT, ERROR})

DTYPES = {"bf16": 2, "i32": 4, "f32": 4, "u8": 1, "i64": 8}

FAIL_EXIT = 75            # exit code of a stage that stops because the chain broke (restart the whole chain)
ALIGN = 256               # staging offsets of a frame's device parts


def align(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


def link_staging(rows: int, carry: int, *, hidden: int = 4096, streams: int = 4) -> int:
    """One link end's pinned ACT staging for chunks of ``rows`` rows: x [rows, streams * hidden] and ``carry`` taps
    [rows, hidden], bf16, each part aligned (S0->S1 at 2,048 rows and one tap: 80 MiB; S1->tail with three: 112)."""

    if rows <= 0:
        return 0
    return align(rows * streams * hidden * 2) + int(carry) * align(rows * hidden * 2)


class ProtocolError(RuntimeError):
    """A frame that does not fit the protocol's order or shape: the chain has lost step (fail-stop)."""


# -- frames ------------------------------------------------------------------------------------------------------
def part_nbytes(dtype: str, shape: Sequence[int]) -> int:
    if dtype not in DTYPES:
        raise ProtocolError(f"part dtype {dtype!r}: one of {', '.join(DTYPES)}")
    if any(int(n) < 0 for n in shape):
        raise ProtocolError(f"negative part shape {list(shape)}")
    return math.prod(int(n) for n in shape) * DTYPES[dtype]


def encode_header(hdr: dict) -> bytes:
    raw = json.dumps(hdr, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_HEADER:
        raise ProtocolError(f"a {len(raw)}-byte header (at most {MAX_HEADER})")
    return raw


def decode_header(raw: bytes) -> dict:
    try:
        hdr = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"unreadable frame header: {exc}") from None
    if not isinstance(hdr, dict) or not isinstance(hdr.get("op"), str) or not isinstance(hdr.get("seq"), int):
        raise ProtocolError("a frame header without op and seq")
    parts = hdr.setdefault("parts", [])
    if not isinstance(parts, list) or any(not isinstance(p, list) or len(p) != 3 for p in parts):
        raise ProtocolError("frame parts: a list of [name, dtype, shape]")
    for name, dtype, shape in parts:
        if not isinstance(name, str) or not isinstance(shape, list):
            raise ProtocolError(f"frame part {name!r}: [name, dtype, shape]")
        part_nbytes(dtype, shape)
    return hdr


def make_header(op: str, seq: int, fields: dict | None, parts: Sequence[tuple[str, str, Sequence[int]]]) -> dict:
    hdr: dict[str, Any] = {"op": op, "seq": int(seq)}
    for key, value in (fields or {}).items():
        if key in ("op", "seq", "parts"):
            raise ProtocolError(f"field {key!r} is the frame's own")
        hdr[key] = value
    hdr["parts"] = [[str(n), str(d), [int(x) for x in s]] for n, d, s in parts]
    return hdr


def frame_bytes(hdr: dict, payload: Sequence[bytes] = ()) -> bytes:
    """A whole frame as bytes (small frames, tests)."""

    raw = encode_header(hdr)
    return U32.pack(len(raw)) + raw + b"".join(bytes(p) for p in payload)


def split_frame(data: bytes) -> tuple[dict, dict[str, bytes], int]:
    """(header, parts, bytes used) of the frame at the start of ``data`` (tests and tools)."""

    if len(data) < 4:
        raise ProtocolError("short frame")
    (n,) = U32.unpack_from(data, 0)
    hdr = decode_header(data[4:4 + n])
    off = 4 + n
    parts = {}
    for name, dtype, shape in hdr["parts"]:
        size = part_nbytes(dtype, shape)
        if off + size > len(data):
            raise ProtocolError("short frame payload")
        parts[name] = data[off:off + size]
        off += size
    return hdr, parts, off


# -- stages ------------------------------------------------------------------------------------------------------
@dataclass
class StageInfo:
    """What a stage says about itself in HELLO (compared by the leader before any weight loads)."""

    layers: tuple[int, int]
    role: str                                      # "follower" or "leader"
    host: str = ""
    pid: int = 0
    settings: dict = field(default_factory=dict)   # env.settings_vector()
    versions: dict = field(default_factory=dict)   # env.versions()
    eos: list = field(default_factory=list)
    gpu: str = ""
    mem_total: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "StageInfo":
        d = dict(d)
        d["layers"] = tuple(int(x) for x in d["layers"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def parse_layers(text: str) -> tuple[int, int]:
    try:
        a, b = (int(x) for x in str(text).split(":"))
    except ValueError:
        raise ValueError(f"layers {text!r}: A:B") from None
    if not 0 <= a < b:
        raise ValueError(f"layers {text!r}: 0 <= A < B")
    return a, b


def check_chain(ranges: Sequence[Sequence[int]], end: int, kinds: Sequence[str], taps: Sequence[int]) -> None:
    """HELLO's hard checks (spec 2.3, items 1-3): contiguous ranges from 0 to ``end``, a DSA layer in every stage
    (graph capture and the sparse graphs read ``index[0]``: G/graphs.py:57, G/decode.py:260) and a KDA layer too (a
    stage without KDA state captures one parity of graphs, G/graphs.py:31, the others two, so READY's BUILT check
    would refuse it only after every weight is loaded), every tap layer inside the chain."""

    if not ranges:
        raise ValueError("no stages")
    at = 0
    for i, (a, b) in enumerate(ranges):
        if a != at:
            raise ValueError(f"stage {i} starts at layer {a}, expected {at} (the ranges must be contiguous from 0)")
        if b <= a:
            raise ValueError(f"stage {i}: an empty range [{a}, {b})")
        if b > len(kinds):
            raise ValueError(f"stage {i}: layers up to {b}, the checkpoint has {len(kinds)}")
        if "dsa" not in list(kinds[a:b]):
            raise ValueError(f"stage {i} [{a}, {b}) holds no DSA layer; every stage needs one (any 4 or more "
                             "consecutive layers do)")
        if "kda" not in list(kinds[a:b]):
            raise ValueError(f"stage {i} [{a}, {b}) holds no KDA layer; every stage needs one (any 2 or more "
                             "consecutive layers do)")
        at = b
    if at != end:
        raise ValueError(f"the chain ends at layer {at}, expected {end}")
    for t in taps:
        if not 0 <= int(t) < end:
            raise ValueError(f"DFlash2 tap after layer {t} lies outside the chain [0, {end})")


def tap_routes(taps: Sequence[int], ranges: Sequence[Sequence[int]]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Per stage (carry_in, carry_out): the tap slots it receives with the streams (computed upstream: tap layer <
    its first layer) and the ones it sends on (computed up to its last layer); the last stage sends none."""

    out = []
    for i, (a, b) in enumerate(ranges):
        cin = tuple(s for s, t in enumerate(taps) if int(t) < a)
        cout = () if i == len(ranges) - 1 else tuple(s for s, t in enumerate(taps) if int(t) < b)
        out.append((cin, cout))
    return out


SETTINGS_SKIP = frozenset({"model_files"})   # compared separately (named in the error)


def compare_stages(infos: Sequence[StageInfo]) -> list[str]:
    """HELLO's check 4: every stage's settings vector, versions and package digests equal the leader's."""

    if not infos:
        return []
    ref = infos[-1]
    problems = []
    for i, s in enumerate(infos[:-1]):
        name = f"stage {i} [{s.layers[0]}, {s.layers[1]}) on {s.host or '?'}"
        for key in sorted(set(ref.settings) | set(s.settings)):
            if s.settings.get(key) != ref.settings.get(key):
                problems.append(f"{name}: setting {key} {s.settings.get(key)!r}, leader {ref.settings.get(key)!r}")
        for key in sorted(set(ref.versions) | set(s.versions)):
            if s.versions.get(key) != ref.versions.get(key):
                problems.append(f"{name}: {key} {s.versions.get(key)!r}, leader {ref.versions.get(key)!r}")
        if list(s.eos) != list(ref.eos):
            problems.append(f"{name}: eos {s.eos}, leader {ref.eos} (TF_GLM_ASSISTANT_ENDS, tokenizer.json)")
    return problems


def chunk_count(begin: int, end: int, rows: int) -> int:
    """Prompt chunks of a PREFILL (decode.prefill's loop: range(begin, end, rows))."""

    return len(range(int(begin), int(end), int(rows)))


class PromptView:
    """A prompt's tokens [begin, begin + len(ids)) under absolute indices, as ``decode.prefill_chunk`` slices it."""

    def __init__(self, begin: int, ids: Sequence[int]) -> None:
        self.begin, self.ids = int(begin), list(int(t) for t in ids)

    def __len__(self) -> int:
        return self.begin + len(self.ids)

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("PromptView takes slices of absolute positions")
        start = self.begin if key.start is None else int(key.start)
        stop = len(self) if key.stop is None else min(int(key.stop), len(self))
        if start < self.begin:
            raise IndexError(f"positions from {start}: this view holds {self.begin} on")
        return self.ids[start - self.begin:max(start, stop) - self.begin]
