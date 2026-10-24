"""tfpp-ds41's wire protocol without torch (spec 9): the ops, the stage description HELLO carries, its comparison.

Frames are tfpp-glm's codec (G/proto.py:36-122): ``[u32 header length][JSON header][parts]``, each part
``[name, dtype, shape]`` of ``prod(shape) * itemsize`` bytes; dtypes bf16, f32, i32, i64, u8 (bool travels as u8).
Control ops go from the leader (the last stage) towards stage 0, every follower forwarding a frame upstream before it
handles it; ACT frames go from stage 0 towards the leader. A LINK frame names the protocol ``tfpp-ds41/1``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Sequence

from tfpp.glm.proto import (DTYPES, FAIL_EXIT, U32, ProtocolError, align, decode_header, encode_header,  # noqa: F401
                            frame_bytes, make_header, part_nbytes, split_frame)

from . import PROTO  # noqa: F401

LINK = "LINK"
# control: leader -> stage 0
GO, AGREE = "GO", "AGREE"
ADMIT, FILL, STEP, FINISH, DROP, QUIT = "ADMIT", "FILL", "STEP", "FINISH", "DROP", "QUIT"
# data: stage 0 -> leader
HELLO, CAP, READY, ACT = "HELLO", "CAP", "READY", "ACT"
ERROR = "ERROR"            # both ways: a follower's failure (data), the leader stopping the chain (control)

CONTROL = frozenset({GO, AGREE, ADMIT, FILL, STEP, FINISH, DROP, QUIT, ERROR})
DATA = frozenset({HELLO, CAP, READY, ACT, ERROR})
RUNTIME_OPS = (ADMIT, FILL, STEP, FINISH, DROP, QUIT, ERROR)


@dataclass
class DsStageInfo:
    """A stage's HELLO entry: ``settings`` and ``versions`` must equal the leader's; ``extra`` is its own (its Engram
    tables, its token map)."""

    layers: tuple
    role: str                                      # "follower" or "leader"
    host: str = ""
    pid: int = 0
    settings: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    gpu: str = ""
    mem_total: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "DsStageInfo":
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


def compare_stages(infos: Sequence[DsStageInfo]) -> list[str]:
    """HELLO's check (spec 5.2 item 4, 13.2): every stage's settings and versions equal the leader's (the last)."""

    if not infos:
        return []
    ref = infos[-1]
    problems = []
    for i, s in enumerate(infos[:-1]):
        name = f"stage {i} [{s.layers[0]}, {s.layers[1]}) on {s.host or '?'}"
        for key in sorted(set(ref.settings) | set(s.settings)):
            if s.settings.get(key) != ref.settings.get(key):
                problems.append(f"{name}: setting {key} {_short(s.settings.get(key))}, leader "
                                f"{_short(ref.settings.get(key))}")
        for key in sorted(set(ref.versions) | set(s.versions)):
            if s.versions.get(key) != ref.versions.get(key):
                problems.append(f"{name}: {key} {s.versions.get(key)!r}, leader {ref.versions.get(key)!r}")
    return problems


def _short(v) -> str:
    t = json.dumps(v, sort_keys=True, default=str)
    return t if len(t) <= 160 else t[:157] + "..."


def digest(obj) -> str:
    """sha256 of the canonical JSON, 16 hex digits (the mirrored state's digest, spec 8.3)."""

    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def check_engram(ranges: Sequence[Sequence[int]], engram_layers: Sequence[int], extras: Sequence[dict]) -> list[str]:
    """Spec 5.2 item 3: every Engram layer has exactly one owner (the stage whose range holds it), the owner was given
    tables holding exactly the layers it owns (StageEngram's rule, checked here before any stage loads its weights),
    and every owner's tables are of one kind (the real tables or a test table) and its token map the same."""

    problems = []
    owners = []
    for e in engram_layers:
        own = [i for i, (a, b) in enumerate(ranges) if a <= e < b]
        if len(own) != 1:
            problems.append(f"Engram layer {e}: {len(own)} owners")
            continue
        i = own[0]
        owners.append(i)
        x = extras[i] if i < len(extras) else {}
        eng = x.get("engram") or {}
        if not eng:
            problems.append(f"stage {i} {list(ranges[i])} owns Engram layer {e} but was given no tables (--engram DIR)")
        elif e not in eng.get("layers", []):
            problems.append(f"stage {i}: its Engram tables hold layers {eng.get('layers')}, not {e}")
    for i in sorted(set(owners)):
        eng = (extras[i] if i < len(extras) else {}).get("engram") or {}
        a, b = ranges[i]
        mine = {int(e) for e in engram_layers if a <= e < b}
        others = sorted(int(e) for e in eng.get("layers", []) if int(e) not in mine)
        if eng and others:
            problems.append(f"stage {i} {list(ranges[i])}: its Engram tables also hold layers {others}, it owns "
                            f"{sorted(mine)} (the fork reads every table in the folder: give each owner a folder of "
                            "exactly its own layers' tables)")
    modes = {(extras[i].get("engram") or {}).get("mode") for i in set(owners)
             if i < len(extras) and extras[i].get("engram")}
    if len(modes) > 1:
        problems.append(f"the Engram owners' tables are of different kinds: {sorted(str(m) for m in modes)} (all real "
                        "or all the same test mode)")
    maps = {extras[i].get("token_map_sha") for i in set(owners) if i < len(extras)}
    if len(maps) > 1:
        problems.append(f"the Engram owners' token maps differ: {sorted(str(m) for m in maps)}")
    for i, x in enumerate(extras):
        if x.get("engram") and i not in owners:
            problems.append(f"stage {i} {list(ranges[i])} was given Engram tables but owns no Engram layer")
    return problems
