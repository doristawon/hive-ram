"""T-S for tfpp-glm v2 (--parallel 2..8): what can be checked on any CPU Python, without torch or a GPU (spec v2 16.1).

  cd tfpp && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tfpp/glm/tests/test_static.py tfpp/glm/tests/test_v2_static.py

Beside the pure modules (v2proto, v2capacity, v2startup, v2compare, v2bench's lockstep, v2pins) some tests import the
modules that need torch and MIA's package (v2part, v2wire, v2follower, v2tail, v2trace, v2check) under stand-ins
(``stubbed``): fake ``torch`` and ``tensorfold`` modules with MIA's real pure functions (seal, parse, ...) extracted from
the vendored source by AST. That catches import-time and wiring mistakes and exercises the link-side logic; every CUDA
path stays untested here.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import json
import os
import queue
import random
import sys
import threading
import time
import types
import typing
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
GLM = HERE.parent
T = HERE.parents[2]                                     # the tfpp checkout
if str(T) not in sys.path:
    sys.path.insert(0, str(T))

from tfpp.glm import (capacity, env, pins, proto, v2bench, v2capacity, v2compare, v2pins, v2proto,  # noqa: E402
                      v2startup)

FRO = HERE / "data" / "v1_frozen"
MANIFEST = json.loads((FRO / "MANIFEST.json").read_text(encoding="utf-8"))
WHITELIST_SHA = "03826e160f8645518ad1e3ea31cdc7e402b4ef3616ca15f3a0a093aa37ba78b6"
CONFIG = json.loads((HERE / "data" / "glm53_config.json").read_text(encoding="utf-8"))
TEXT = dict(CONFIG["text_config"])
TEXT["_quantization"] = CONFIG["quantization_config"]


def vendor_root() -> Path | None:
    for c in (os.environ.get("TFPP_GLM_VENDOR", ""), str(T.parent / "vendor" / "tensorfold-mia-v111pre" / "tensorfold")):
        if c and (Path(c) / "__init__.py").is_file():
            return Path(c)
    try:
        return pins.installed_root()
    except RuntimeError:
        return None


ROOT = vendor_root()
needs_tree = pytest.mark.skipif(ROOT is None, reason="no MIA tensorfold tree found")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# -- 1. v1 is what ran on hardware, plus documented post-v1 fixes ------------------------------------------------------------
POST = MANIFEST.get("post_v1", {})


def pinned(kind: str, name: str, hardware: str) -> str:
    """The sha256 a v1 file must have now: the hardware run's, or its documented post-v1 pin (MANIFEST post_v1)."""

    return POST.get(kind, {}).get(name, {}).get("sha256", hardware)


def test_v1_files_are_byte_identical():
    """Every v1 file is byte for byte what ran on hardware, or exactly what its documented post-v1 fix made of it (pinned
    too: a further change must update MANIFEST.json's post_v1 entry and say why)."""

    for name, want in MANIFEST["unchanged"].items():
        assert sha(GLM / name) == pinned("glm", name, want), \
            f"{name} changed: v1 stays what ran on hardware unless a documented fix (MANIFEST post_v1) says otherwise"
    for name, want in MANIFEST["scripts_unchanged"].items():
        assert sha(T / "scripts" / "glm" / name) == pinned("scripts", name, want), name
    assert sha(T / "tfpp" / "warm.py") == MANIFEST["warm_py"]
    for name, info in MANIFEST["whitelisted"].items():
        assert sha(FRO / name) == pinned("frozen", name, info["v1_sha256"]), f"the frozen copy of {name} changed"


def test_post_v1_changes_are_documented():
    """Each post-v1 pin names a v1 file, differs from its hardware hash (else it is stale) and says what changed."""

    hardware = {"glm": MANIFEST["unchanged"], "scripts": MANIFEST["scripts_unchanged"],
                "frozen": {**{n: i["v1_sha256"] for n, i in MANIFEST["whitelisted"].items()},
                           **MANIFEST["scripts_additive"]}}
    assert set(POST) <= {"what", *hardware}
    for kind, entries in POST.items():
        if kind == "what":
            continue
        for name, entry in entries.items():
            assert name in hardware[kind], (kind, name)
            assert entry["sha256"] != hardware[kind][name], f"{kind}/{name} is the hardware file again: drop the entry"
            assert entry["changes"] and all(isinstance(c, str) and c.strip() for c in entry["changes"]), (kind, name)


def test_v1_tfpp_sha_baseline():
    """env.tfpp_sha's algorithm over the v1 file set's recorded hashes (the frozen whitelisted files' v1 hashes) is the
    digest the hardware run's HELLO printed: the manifest's hardware hashes are the ones that ran."""

    names = sorted([n for n in MANIFEST["unchanged"] if n.endswith(".py") and "/" not in n] + list(MANIFEST["whitelisted"]))
    h = hashlib.sha256()
    for name in names:
        hw = MANIFEST["whitelisted"][name]["v1_sha256"] if name in MANIFEST["whitelisted"] else MANIFEST["unchanged"][name]
        h.update(f"{name}\0{hw}\n".encode())
    h.update(f"warm.py\0{MANIFEST['warm_py']}\n".encode())
    assert h.hexdigest() == MANIFEST["tfpp_sha_v1"] == "7a8e0a2e7ef98d9d942f8280015f0e1b4d241160dedc1d0b4af019d0b5ebd3a0"
    # v2 files are in the digest now (every stage must run the same code): it differs from the baseline
    assert env.tfpp_sha() != MANIFEST["tfpp_sha_v1"]


def _strip(tree: ast.AST, dumps: dict[str, str], removed: dict[str, int]) -> None:
    for node in list(ast.walk(tree)):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list) or not stmts or not isinstance(stmts[0], ast.stmt):
                continue
            keep = []
            for s in stmts:
                d = ast.dump(s)
                hit = next((k for k, v in dumps.items() if v == d), None)
                if hit is None:
                    keep.append(s)
                else:
                    removed[hit] += 1
            setattr(node, field, keep)


def test_serve_and_follower_differ_from_v1_only_by_the_whitelist():
    wl = json.loads((FRO / "WHITELIST.json").read_text(encoding="utf-8"))
    canon = hashlib.sha256(json.dumps(wl, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert canon == WHITELIST_SHA == MANIFEST["whitelist_sha256"], "the whitelist itself changed"
    assert len(wl) == 10
    for name in ("serve.py", "follower.py"):
        dumps = {e["id"]: ast.dump(ast.parse(e["source"]).body[0]) for e in wl if e["file"] == name}
        removed = dict.fromkeys(dumps, 0)
        new = ast.parse((GLM / name).read_text(encoding="utf-8"))
        old = ast.parse((FRO / name).read_text(encoding="utf-8"))
        _strip(new, dumps, removed)
        assert all(n == 1 for n in removed.values()), f"{name}: each whitelisted statement exactly once: {removed}"
        assert ast.dump(new) == ast.dump(old), f"{name} differs from v1 beyond the whitelisted statements"


def test_the_whitelist_guards_p1():
    """Every whitelisted insertion is behind --parallel > 1, a v2-only flag, or GO's P > 1 / the v2 attribute."""

    wl = json.loads((FRO / "WHITELIST.json").read_text(encoding="utf-8"))
    for e in wl:
        src = e["source"]
        assert any(k in src for k in ("a.parallel > 1", "pool_rows", "--pool-rows", "--kept-entries", "--mbench", 'go["P"]', "self.v2", "v2 = None")), e["id"]
    # P=1 and every v1 argument vector keep their v1 outcome (test_static.test_cli's rejections included)
    from tfpp.glm import serve

    p = serve.build_parser()
    serve.check_args(p.parse_args(["--role", "leader", "--model", "/m", "--layers", "0:8", "--truncate", "8",
                                   "--no-drafts"]))
    for bad in (["--role", "leader", "--model", "/m", "--layers", "0:45", "--parallel", "4"],
                ["--role", "leader", "--model", "/m", "--layers", "0:45", "--parallel", "0"],
                ["--role", "leader", "--model", "/m", "--layers", "0:45", "--pool-rows", "4096"],
                ["--role", "leader", "--model", "/m", "--layers", "0:45", "--kept-entries", "4"],
                ["--role", "leader", "--model", "/m", "--layers", "0:45", "--mbench", "AS1"]):
        with pytest.raises(SystemExit):
            serve.check_args(p.parse_args(bad))


def test_start_stage_is_additive():
    """start_stage.sh gained lines for --parallel > 1 only: with the new node.env keys unset, the leader's command is v1's."""

    text = (T / "scripts" / "glm" / "start_stage.sh").read_text(encoding="utf-8")
    assert '--parallel "$PARALLEL"' in text
    block = text[text.index("if [[ ${PARALLEL:-1} -gt 1 ]]; then       # tfpp-glm v2 only"):]
    block = block[:block.index("\nfi\n")]
    for arg in ('--pool-rows "$POOL_ROWS"', '--kept-entries "$KEPT_ENTRIES"', '--mbench "$MBENCH"'):
        assert f'args+=({arg}' in block, arg
        assert arg not in text.replace(block, ""), f"{arg} outside the PARALLEL > 1 block"


def v1_start_stage(text: str) -> str:
    """start_stage.sh with the v2 additions taken out again (the keys exported for the test hooks, the MBENCH checks, the
    PARALLEL > 1 block): what is left must be v1's file, byte for byte."""

    a = text.replace(" TFPP_GLM_TIMING TFPP_GLM_TIME_ROWS TFPP_GLM_FAULT_V2 TFPP_GLM_FAULT_FOLLOWER; do", "; do")
    i = a.index('if [[ $ROLE == leader && -n "${MBENCH:-}" ]]; then')
    a = a[:i] + a[a.index("fi" + chr(10), i) + 3:]
    i = a.index("  if [[ ${PARALLEL:-1} -gt 1 ]]; then       # tfpp-glm v2 only")
    a = a[:i] + a[a.index("  fi" + chr(10), i) + 5:]
    return a.replace('[[ $ROLE == leader && -z "${BENCH:-}" && -z "${MBENCH:-}" ]]', '[[ $ROLE == leader && -z "${BENCH:-}" ]]')


def test_start_stage_minus_v2_is_v1_byte_for_byte():
    """The P=1 command line and environment of start_stage.sh are v1's: the file less its documented v2 lines is the
    frozen v1 copy, whose sha256 is the one the manifest recorded from the hardware runs (or its post-v1 pin: the same
    fixes made in both files)."""

    now = (T / "scripts" / "glm" / "start_stage.sh").read_bytes().decode("utf-8")
    frozen = (FRO / "start_stage.sh").read_bytes()
    want = pinned("frozen", "start_stage.sh", MANIFEST["scripts_additive"]["start_stage.sh"])
    assert hashlib.sha256(frozen).hexdigest() == want
    assert v1_start_stage(now).encode("utf-8") == frozen


# -- 2. the CLI -----------------------------------------------------------------------------------------------------------
def test_cli_v2():
    from tfpp.glm import serve

    p = serve.build_parser()
    base = ["--role", "leader", "--model", "/m", "--layers", "0:45"]
    serve.check_args(p.parse_args(base + ["--parallel", "4", "--drafter", "/d"]))
    serve.check_args(p.parse_args(base + ["--parallel", "2", "--no-drafts", "--pool-rows", "6144", "--kept-entries", "4",
                                          "--mbench", "AS1", "--bench-out", "/o.json"]))
    for bad in (base + ["--parallel", "4"],                                               # no drafter
                base + ["--parallel", "9", "--drafter", "/d"],
                base + ["--parallel", "4", "--drafter", "/d", "--pool-rows", "1000"],
                base + ["--parallel", "4", "--drafter", "/d", "--pool-rows", "0"],
                base + ["--parallel", "4", "--drafter", "/d", "--kept-entries", "0"],
                base + ["--parallel", "4", "--drafter", "/d", "--mbench", "AS1", "--bench", "/p.jsonl"],
                base + ["--parallel", "4", "--drafter", "/d", "--bench-out", "/o.json"],
                base + ["--parallel", "4", "--drafter", "/d", "--mbench", "no-such-schedule"],
                ["--role", "follower", "--model", "/m", "--layers", "0:7", "--down", "h:1", "--mbench", "AS1"],
                ["--role", "follower", "--model", "/m", "--layers", "0:7", "--down", "h:1", "--pool-rows", "4096"],
                ["--role", "follower", "--model", "/m", "--layers", "0:7", "--down", "h:1", "--kept-entries", "2"]):
        with pytest.raises(SystemExit):
            serve.check_args(p.parse_args(bad))


# -- 3. the v2 wire --------------------------------------------------------------------------------------------------------
def test_v2_frames_round_trip_and_sizes():
    ints = np.arange(11, dtype=np.int32)
    cases = [
        (v2proto.MULTI, v2proto.multi_fields(0, ints.tolist()),
         [("ints", "i32", [11]), (v2proto.img_part(7), "bf16", [2, 2])], True),
        (v2proto.WINDOWS, v2proto.windows_fields(0, 41, [[0, 100, 8], [1, 200, 24]]), [("ids", "i32", [32])], True),
        (v2proto.KEEPS, v2proto.keeps_fields(0, 41, [3, 8], [103, 208]), [], True),
        (proto.ACT, v2proto.act_round(0, 41, 32), [("x", "bf16", [32, 16384]), ("tap:0", "bf16", [32, 4096])], False),
        (proto.ACT, v2proto.act_chunk(0, 3, 4096, 2048), [("x", "bf16", [2048, 16384])], False),
        (proto.ACT, v2proto.act_pieces(0, [[0, 0, 1024], [1, 0, 1024]], 2048), [("x", "bf16", [2048, 16384])], False),
    ]
    for i, (op, fields, parts, small) in enumerate(cases):
        hdr = proto.make_header(op, i, fields, parts)
        raw = proto.encode_header(hdr)
        got = proto.decode_header(raw)
        assert got["op"] == op and got["seq"] == i and got["g"] == 0
        for k, v in fields.items():
            assert got[k] == v
        assert len(raw) < 200, f"{op} header {len(raw)} bytes"         # headers are small (spec 5.6)
        if small:                                                       # control frames are whole in a test
            payload = [bytes(1) * proto.part_nbytes(d, s) for _, d, s in parts]
            data = proto.frame_bytes(hdr, payload)
            _, p2, used = proto.split_frame(data)
            assert used == len(data) and set(p2) == {p[0] for p in parts}
    assert v2proto.img_sid("img:12") == 12 and v2proto.img_sid("ints") is None and v2proto.img_sid("img:x") is None
    assert v2proto.CONTROL_V2 == {"MULTI", "WINDOWS", "KEEPS"} and not (v2proto.CONTROL_V2 & proto.CONTROL)
    with pytest.raises(ValueError):
        v2proto.multi_fields(0, [1])


def test_v2_checks():
    hdr = v2proto.windows_fields(0, 5, [[0, 10, 3], [2, 40, 8]])
    lanes = [[0, 10], [2, 40]]
    ok = dict(received=5, lanes=lanes, rows_max=16, window=32)
    assert v2proto.check_windows(hdr, 11, **ok) == [[0, 10, 3], [2, 40, 8]]
    for bad, kw in ((hdr, dict(ok, received=6)), (dict(hdr, g=1), ok), (hdr, dict(ok, lanes=[[0, 10]])),
                    (hdr, dict(ok, lanes=[[0, 10], [2, 41]])), (hdr, dict(ok, rows_max=4)), (hdr, dict(ok, window=8))):
        with pytest.raises(ValueError):
            v2proto.check_windows(bad, 11, **kw)
    with pytest.raises(ValueError):
        v2proto.check_windows(hdr, 10, **ok)                                   # ids count != sum of rows
    kh = v2proto.keeps_fields(0, 5, [3, 8], [13, 48])
    assert v2proto.check_keeps(kh, rid=5, rows=[3, 8]) == [3, 8]
    for bad, rows, rid in ((kh, [3, 7], 5), (kh, [3, 8], 6), (dict(kh, keeps=[0, 8]), [3, 8], 5), (kh, [3], 5)):
        with pytest.raises(ValueError):
            v2proto.check_keeps(bad, rid=rid, rows=rows)
    fields = v2proto.act_round(0, 5, 11)
    good = {"op": "ACT", **fields, "parts": [["x", "bf16", [11, 4]]]}
    assert v2proto.check_act(good, fields, {"x"}) is None
    assert "rid" in v2proto.check_act(dict(good, rid=6), fields, {"x"})
    assert "parts" in v2proto.check_act(good, fields, {"x", "tap:0"})


def test_fault_points():
    f = v2proto.Fault("round.act:3")
    f.hit("round.act")
    f.armed = True
    f.hit("round.windows")
    f.hit("round.act")
    f.hit("round.act")
    with pytest.raises(RuntimeError, match="round.act:3"):
        f.hit("round.act")
    for bad in ("round.act", "round.act:0", ":3", "x:y"):
        with pytest.raises(ValueError):
            v2proto.Fault(bad)
    assert v2proto.Fault("").point is None
    z = v2proto.Fault("windows:2:stop", kill=True, armed=True)
    assert (z.point, z.at, z.stop) == ("windows", 2, True)
    with mock.patch("os.kill") as kill:
        z.hit("windows")
        assert not kill.called
        z.hit("windows")                                       # SIGSTOP itself (once let go it carries on)
        assert kill.call_count == 1
    k = v2proto.Fault("windows:2", kill=True, armed=True)
    with mock.patch("os.kill") as kill:
        k.hit("windows")
        assert not kill.called
        with pytest.raises(RuntimeError):                      # (a real SIGKILL never gets to raise)
            k.hit("windows")
        assert kill.called


def test_describe_ops_and_stats(capsys):
    names = {1: "ADMIT", 5: "FILL", 8: "IDLE"}
    assert v2proto.describe_ops([1, 2, 7, 9, 5, 2, 3, 4, 8, 0, 17, 99], names) == [["ADMIT", 2], ["FILL", 2], ["IDLE", 0]]
    assert v2proto.describe_ops([1, 5, 7, 17, 99], names) == [["?", 3]]
    st = v2proto.RoundStats("t", every=2)
    st.add(8, 0.01, 0.002)
    st.add(4, 0.03, 0.004)
    assert "t: 2 rounds, 6.0 rows, 20.0 ms a round, 3.0 ms of it waiting" in capsys.readouterr().out


# -- 4. the pool plan -----------------------------------------------------------------------------------------------------
class FakeGeo:
    def __init__(self, per, fixed=1000):
        self.per, self.fixed = per, fixed

    def bytes_at(self, n):
        return self.fixed + self.per * n


def fake_kept_entries(configured, share_pct, budget, state):
    return max(1, min(configured, int(budget * 0.75) // max(1, state))), ""


def plan(**kw):
    args = dict(window=8192, cap=8208, geos=[FakeGeo(1000), FakeGeo(2000), FakeGeo(3000)],
                spare=[10 ** 9, 10 ** 9, 10 ** 9], states=[10 ** 6, 10 ** 6, 10 ** 6], cache_wanted=3 * 10 ** 8,
                entries_cfg=8, max_rows=16, kept_entries_fn=fake_kept_entries)
    args.update(kw)
    return v2capacity.pool_plan(**args)


def test_pool_plan_arithmetic_and_refusals():
    p = plan()
    # leader: (cache - entries * state) // per_token; every stage must hold it: the tightest rules
    assert p.entries == 8 and p.per_token == [1000, 2000, 3000]
    assert p.extra == (3 * 10 ** 8 - 8 * 10 ** 6) // 3000
    assert p.rows == (8208 + p.extra) // 2048 * 2048 and p.limit == min(8192, p.rows - 16)
    tight = plan(spare=[10 ** 9, 5 * 10 ** 7, 10 ** 9])
    assert tight.extra == (5 * 10 ** 7 - 8 * 10 ** 6) // 2000 < p.extra
    with pytest.raises(v2capacity.PoolRefused, match="stage 1"):
        plan(spare=[10 ** 9, 7 * 10 ** 6, 10 ** 9])                              # cannot even hold the kept states
    with pytest.raises(v2capacity.PoolRefused, match="less than one"):
        plan(window=500, cap=1000, cache_wanted=0)                                               # slots short of one extent
    # the tests' overrides
    o = plan(pool_rows=6144, kept_entries=4)
    assert (o.rows, o.entries, o.limit) == (6144, 4, min(8192, 6144 - 16))
    with pytest.raises(v2capacity.PoolRefused, match="multiple"):
        plan(pool_rows=6000)
    with pytest.raises(v2capacity.PoolRefused, match="stage 2"):
        plan(pool_rows=2048 * 40, spare=[10 ** 9, 10 ** 9, 3000 * (2048 * 40 - 8208)])      # the 8 kept states do not fit too
    with pytest.raises(v2capacity.PoolRefused):
        plan(kept_entries=0)


def importable_tensorfold() -> bool:
    if ROOT is None:
        return False
    parent = str(ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        import tensorfold.cuda.geometry  # noqa: F401
        import tensorfold.families.glm5_next.cuda.engine  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not importable_tensorfold(), reason="tensorfold's geometry not importable")
def test_pool_plan_on_the_package_geometry_19_7_19():
    """The full chain's numbers (as the design's memory model computed them): per-token bytes, a kept state, the pool
    and the limit at TF_GLM_CACHE_GIB=6.2, 32 entries, a 262,144-token window, P=4."""

    gib = 2 ** 30
    ranges = [(0, 19), (19, 26), (26, 45)]
    flags = capacity.GeoFlags()
    geos = [capacity.stage_geometry(TEXT, a, b, last=i == 2, mtp=False, parallel=4, prefill_rows=2048, flags=flags)
            for i, (a, b) in enumerate(ranges)]
    assert [v2capacity.per_token_bytes(g, 262144 + 16) for g in geos] == [2800, 1672, 3364]
    states = v2capacity.stage_states(TEXT, ranges)
    assert [s >> 20 for s in states] == [83, 42, 78]
    spare = [60 * gib, 30 * gib, 70 * gib]
    p = v2capacity.pool_plan(window=262144, cap=262144 + 16, geos=geos, spare=spare, states=states,
                             cache_wanted=int(6.2 * gib), entries_cfg=32, share_pct=0.0, max_rows=16)
    assert p.entries == 32 and p.rows == 1452032 and p.limit == 262144
    assert p.extra == 1191118
    assert "pool 1452032 tokens, 32 kept entries, limit 262144" == p.describe()
    # a chain whose middle stage has only 3 GiB spare is refused or shrinks the pool, naming no one else
    small = v2capacity.pool_plan(window=262144, cap=262144 + 16, geos=geos, spare=[60 * gib, 2 * gib, 70 * gib],
                                 states=states, cache_wanted=int(6.2 * gib), entries_cfg=32, max_rows=16)
    assert small.rows < p.rows


def test_kept_room_v2():
    gib = 2 ** 30
    caps = [{"reserve": 3 * gib, "declared": False}, {"reserve": 3 * gib, "declared": True},
            {"reserve": 3 * gib, "declared": False}]
    built = [{"layers": [0, 19], "free_after": 20 * gib}, {"layers": [19, 26], "free_after": 0},
             {"layers": [26, 45], "free_after": int(1.6 * gib)}]
    states = [83 << 20, 42 << 20, 78 << 20]
    out = v2capacity.kept_room_v2(caps, built, 32, states)
    assert len(out) == 1 and "[26, 45]" in out[0]                                   # 32 x 78 MiB = 2.4 GiB > 0.1
    assert v2capacity.kept_room_v2(caps, built, 1, states) == []


def test_startup_frames():
    class Plan:
        rows, entries, limit = 1452032, 32, 262144

    base = {"window": 262144, "capacity": 262160, "long_context": True, "cache_bytes": 12345, "entries": 8,
            "scale": [1.0], "pool_rows": None, "parallel": 4, "max_rows": 16, "prefill_rows": 2048,
            "graph_rows": [1], "graphs": True, "taps": [1], "grid": 0, "kv": "fp8"}
    msg = v2startup.agree_message_v2(base, Plan, multi_window=32)
    assert list(msg)[:15] == list(base) and list(msg)[15:] == ["multi_window", "limit", "groups"]
    assert (msg["pool_rows"], msg["entries"], msg["cache_bytes"], msg["limit"], msg["groups"]) == (1452032, 32, 0, 262144, 1)
    go = {"P": 4}
    v2startup.check_agree(go, msg, multi_window=32)
    for key, value in (("parallel", 3), ("pool_rows", 1000), ("pool_rows", 0), ("multi_window", 24), ("entries", 0),
                       ("limit", 262145), ("groups", 2)):
        with pytest.raises(proto.ProtocolError, match=key):
            v2startup.check_agree(go, dict(msg, **{key: value}), multi_window=32)
    built = [{"layers": [0, 19], "verify_graphs": [1, 2], "pool_rows": 8, "slots": 4, "multi_window": 32, "segs": 4},
             {"layers": [19, 45], "verify_graphs": [1, 2], "pool_rows": 8, "slots": 4, "multi_window": 32, "segs": 4}]
    assert v2startup.compare_built_v2(built, lambda b: []) == []
    built[0]["verify_graphs"] = [1]
    built[0]["slots"] = 3
    out = v2startup.compare_built_v2(built, lambda b: ["v1 problem"])
    assert out[0] == "v1 problem" and any("verify_graphs" in o for o in out) and any("slots" in o for o in out)


# -- 5. v2pins and the schedules --------------------------------------------------------------------------------------------
@needs_tree
def test_v2_seams_unchanged():
    assert v2pins.diff(ROOT) == []
    assert all(v != "missing" for v in v2pins.pin_hashes(ROOT).values())
    assert len(v2pins.SEAMS) == len(v2pins.PINS) == 55
    mutated = dict(v2pins.PINS)
    mutated[next(iter(mutated))] = "0" * 64
    assert v2pins.diff(ROOT, mutated) == [next(iter(mutated))]


def test_schedules():
    names = sorted(p.stem for p in (GLM / "schedules").glob("*.json"))
    assert names == ["AS1", "AS2", "AS3", "AS4", "AS5", "FREE1", "FREE2", "FREE4", "HTTP8"]
    items = v2bench.v1bench.load_prompts(GLM / "bench_prompts.jsonl")
    assert len(items) == 19
    for name in names:
        s = v2bench.load_schedule(name)
        assert s["name"] == name
        if s["mode"] == "lockstep":
            reqs = v2bench.expand(items, s)
            assert reqs and len({r["tag"] for r in reqs}) == len(reqs)
    as2 = v2bench.expand(items, v2bench.load_schedule("AS2"))
    assert [r["at"] for r in as2] == [0, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135, 152, 170]
    assert [r["id"] for r in v2bench.expand(items, v2bench.load_schedule("AS5"))] == ["code"]
    assert len(v2bench.expand(items, v2bench.load_schedule("AS1"))) == 19
    f4 = v2bench.load_schedule("FREE4")
    assert (f4["mode"], f4["concurrency"], f4["stagger_ms"], f4["rounds"]) == ("free", 4, 50.0, 2)
    with pytest.raises(ValueError):
        v2bench.expand(items, {"name": "x", "arrivals": [0, 1]})
    with pytest.raises(ValueError):
        v2bench.expand(items, {"name": "x", "only": ["nope"], "arrivals": 0})


# -- 6. the lockstep harness ---------------------------------------------------------------------------------------------
class FakeStream:
    background = False

    def __init__(self, tag):
        self.tag = tag
        self.glm = {}


class FakeSched:
    """GlmScheduler's loop in miniature: an iteration blocks on the queue when idle, admits what is queued up to
    ``max_streams``, runs every lane one tick, finishes the lanes whose ticks are over; _Flush items pass through."""

    class _Flush:
        background = False
        emit = None

    def __init__(self, ticks, max_streams=3):
        self.ticks, self.max_streams = ticks, max_streams
        self.waiting = queue.Queue()
        self.held, self.loading, self.ready = None, [], []
        self.lanes: dict[str, list] = {}
        self.decoder = types.SimpleNamespace(live=lambda: len(self.lanes))
        self.log: list = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            self._iteration()

    def _iteration(self):
        idle = not self.lanes and self.held is None
        items = [self.waiting.get()] if idle else []
        while len(self.lanes) + len(items) < self.max_streams:
            try:
                items.append(self.waiting.get_nowait())
            except queue.Empty:
                break
        admitted = []
        for stream, box in items:
            if stream is self._Flush:
                continue
            self.lanes[stream.tag] = [self.ticks[stream.tag], box]
            admitted.append(stream.tag)
        done = []
        for tag, lane in list(self.lanes.items()):
            lane[0] -= 1
            if lane[0] <= 0:
                done.append(tag)
        self.log.append((admitted, sorted(self.lanes), sorted(done)))
        for tag in done:
            self.lanes.pop(tag)[1].put(tag)


def run_lockstep_once(arrivals, ticks, seed):
    rnd = random.Random(seed)
    sch = FakeSched(ticks)
    landed = v2bench.Landings()
    iters: list = []
    orig_put = sch.waiting.put
    worker = sch.thread

    def put(item):
        orig_put(item)
        if item[0] is not FakeSched._Flush and threading.current_thread() is not worker:
            landed.event(item[0].tag).set()

    sch.waiting.put = put

    def client(tag, delay):
        def run():
            time.sleep(delay)                                  # a request thread that is a little late
            box = queue.Queue()
            sch.waiting.put((FakeStream(tag), box))
            return box.get(timeout=30)
        return run

    reqs = [v2bench.Req(tag, at, client(tag, rnd.random() * 0.003)) for tag, at in arrivals]
    lock = v2bench.Lockstep(sch, reqs, landed=landed, write=iters.append)
    results = lock.run(timeout=60)
    time.sleep(0.1)                                            # the worker's last hook call (it then blocks, idle)
    return sch, iters, results


def test_lockstep_is_deterministic():
    rnd = random.Random(7)
    arrivals = [(f"r{i}", rnd.choice([0, 0, 1, 3, 3, 8, 20, 21])) for i in range(50)]
    ticks = {tag: 1 + rnd.randrange(5) for tag, _ in arrivals}
    runs = [run_lockstep_once(arrivals, ticks, seed) for seed in range(4)]
    first_log, first_iters = runs[0][0].log, runs[0][1]
    for sch, iters, results in runs:
        assert all(ok for ok, _ in results.values()) and sorted(results) == sorted(t for t, _ in arrivals)
        assert iters == first_iters, "the released sequence must not depend on the threads' timing"
        assert sch.log == first_log, "nor the scheduler's iterations"
    assert first_iters[0]["n"] == 0
    assert [r["released"] for r in first_iters if r["n"] == 0][0] == [t for t, a in arrivals if a == 0]
    ns = [r["n"] for r in first_iters]
    assert ns == sorted(ns) and len(ns) == len(set(ns))


def test_lockstep_fast_forwards_over_idle_gaps():
    sch, iters, results = run_lockstep_once([("a", 0), ("b", 100), ("c", 100), ("d", 250)],
                                            {"a": 1, "b": 2, "c": 1, "d": 1}, seed=1)
    assert all(ok for ok, _ in results.values())
    assert [(r["n"], r["released"]) for r in iters if r["released"]] == [(0, ["a"]), (100, ["b", "c"]), (250, ["d"])]
    # the wake-up iteration (the _Flush item), then a; b and c together (c ends, b has one tick left); b's last; d
    assert sch.log == [([], [], []), (["a"], ["a"], ["a"]), (["b", "c"], ["b", "c"], ["c"]), ([], ["b"], ["b"]),
                       (["d"], ["d"], ["d"])]


def test_lockstep_failures_and_timeouts():
    sch = FakeSched({"a": 1})
    landed = v2bench.Landings()

    def boom():
        raise ValueError("generate refused it")

    lock = v2bench.Lockstep(sch, [v2bench.Req("a", 0, boom)], landed=landed, wait_s=5)
    res = lock.run(timeout=30)
    assert res["a"][0] is False and "generate refused it" in res["a"][1]            # a failed client lands anyway
    with pytest.raises(ValueError):
        v2bench.Lockstep(sch, [v2bench.Req("a", 0, boom), v2bench.Req("a", 1, boom)], landed=landed)
    assert "_iteration" not in sch.__dict__                                          # the hook is removed after a run


# -- 7. the comparison tools ----------------------------------------------------------------------------------------------
def leader_trace(tag="r0", mutate=None):
    pool = {"free": 4096, "extents": [[0, 0, 2048, None]], "kept": [[0, 64, 0]], "lanes": []}
    recs = [{"t": "meta", "role": "leader"}, {"t": "meta2", "v2": "2.0", "parallel": 4, "pool_rows": 6144, "entries": 4},
            {"t": "iter", "n": 0, "released": [tag]},
            {"t": "req", "tag": tag, "prompt_len": 64},
            {"t": "msg", "mseq": 5, "digest": 111, "ops": [["ADMIT", 40]], "sha": "aa"},
            {"t": "admit", "tag": tag, "sid": 3, "slot": 0, "base": 0, "size": 2048, "kid": -1},
            {"t": "msg", "mseq": 6, "digest": 222, "ops": [["FILL", 2]], "sha": "bb"},
            {"t": "fill", "slot": 0, "pos": 0, "start": 0, "R": 64, "head": [1, 2], "taps": [[0, [3, 4]]]},
            {"t": "pool", **pool},
            {"t": "msg", "mseq": 7, "digest": 333, "ops": [["ROUND", 5]], "sha": "cc"},
            {"t": "round", "rid": 7, "segs": [[0, 64, 2]], "sids": [3], "tokens": [9, 8], "logits": [[1, 1], [2, 2]],
             "taps": [[0, [5, 6]]], "wall_ms": 3.0},
            {"t": "keeps", "rid": 7, "keeps": [2], "end": [66]},
            {"t": "pool", **pool},
            {"t": "msg", "mseq": 8, "digest": 444, "ops": [["FINISH", 2], ["IDLE", 0]], "sha": "dd"},
            {"t": "idle", "mseq": 8, "pool": pool},
            {"t": "done", "tag": tag, "tokens": [7, 9, 8], "stats": {"rounds": 1, "decode_s": 0.5}}]
    if mutate:
        mutate(recs)
    return recs


def test_v2compare_strict():
    ref = leader_trace()
    code, lines = v2compare.strict(ref, leader_trace())
    assert code == 0 and lines[-1] == "IDENTICAL"
    # timings are not compared
    code, _ = v2compare.strict(ref, leader_trace(mutate=lambda r: r[10].update(wall_ms=99.0)))
    assert code == 0
    cases = [(lambda r: r[4].update(digest=999), "msg", "mseq 5"),
             (lambda r: r[10]["logits"][1].__setitem__(1, 3), "round", "rid 7"),
             (lambda r: r[8]["extents"][0].__setitem__(1, 2048), "pool", None),
             (lambda r: r[11].update(keeps=[1], end=[65]), "keeps", "rid 7"),
             (lambda r: r.pop(9), "record kinds", None)]
    for mutate, kind, where in cases:
        code, lines = v2compare.strict(ref, leader_trace(mutate=mutate))
        assert code == 1, kind
        text = "\n".join(lines)
        assert "worker records: DIFFER" in text and kind in text, text
        if where:
            assert where in text, text
    code, lines = v2compare.strict(ref, leader_trace(tag="other"))
    assert code == 1 and any("request" in l and "only in" in l for l in lines)
    code, lines = v2compare.strict(ref, leader_trace(mutate=lambda r: r[-1].update(tokens=[7, 9, 9])))
    assert code == 1 and any("DIFFERS in done" in l and "tokens[2]" in l for l in lines)
    # other pinned parameters: refused, not "different"
    code, lines = v2compare.strict(ref, leader_trace(mutate=lambda r: r[1].update(entries=8)))
    assert code == 2 and "pinned differently" in lines[0]
    assert v2compare.strict([r for r in ref if r["t"] != "meta2"], ref)[0] == 2


def test_v2compare_stages_and_streams():
    leader = [r for r in leader_trace() if r["t"] in ("msg", "idle")]
    fol = [{"t": "msg", "mseq": 3, "digest": 1, "ops": [["ADMIT", 1]], "sha": "zz"}] + [dict(r) for r in leader]
    code, lines = v2compare.stages(leader, [("s0", fol)])
    assert code == 0, lines
    bad = [dict(r) for r in fol]
    bad[3] = dict(bad[3], digest=5)
    code, lines = v2compare.stages(leader, [("s0", bad)])
    assert code == 1 and "DIFFERS at 2" in "\n".join(lines)
    bad = [dict(r) for r in fol]
    bad[-1] = dict(bad[-1], pool=dict(bad[-1]["pool"], free=1))
    code, lines = v2compare.stages(leader, [("s0", bad)])
    assert code == 1 and "idle pools DIFFER" in "\n".join(lines)
    ref, got = leader_trace(), leader_trace()
    assert v2compare.streams(ref, got, rows=True)[0] == 0
    assert v2compare.committed_rows(ref) == {"r0": {64: (9, (1, 1)), 65: (8, (2, 2))}}
    got[10] = dict(got[10], logits=[[1, 1], [2, 9]])
    code, lines = v2compare.streams(ref, got, rows=True)
    assert code == 1 and any("position 65" in l for l in lines)
    assert v2compare.streams(ref, got, rows=False)[0] == 0                       # replies alone are equal
    a = [{"id": "x", "rep": 0, "tokens": [1, 2, 3]}, {"id": "stop-after-5", "rep": 0, "tokens": [1, 2, 3, 4, 5]}]
    b = [{"id": "x", "rep": 0, "tokens": [1, 2, 3]}, {"id": "stop-after-5", "rep": 0, "tokens": [1, 2, 3, 4, 5, 6, 7]}]
    assert v2compare.bench(a, b)[0] == 0
    assert v2compare.bench(a, [dict(b[0], tokens=[1, 2, 4]), b[1]])[0] == 1
    assert v2compare.bench(a, b, prefix_ids=())[0] == 1
    # refused alike on both sides (a prompt past the context: the texts carry each engine's own limit) is equal
    assert v2compare.bench([{"id": "long", "rep": 0, "error": "a"}], [{"id": "long", "rep": 0, "error": "b"}])[0] == 0
    assert v2compare.bench([{"id": "long", "rep": 0, "error": "a"}], [{"id": "long", "rep": 0, "tokens": [1]}])[0] == 1
    # a free run's passes (rep + 1000 * pass) are each compared with the reference's rep; a missing one is named
    ref = [{"id": "x", "rep": 0, "tokens": [1, 2]}, {"id": "y", "rep": 0, "tokens": [3]}]
    free = [{"id": "x", "rep": 0, "tokens": [1, 2]}, {"id": "x", "rep": 1000, "tokens": [1, 2]},
            {"id": "y", "rep": 0, "tokens": [3]}, {"id": "y", "rep": 1000, "tokens": [3]}]
    assert v2compare.bench(ref, free)[0] == 0
    code, lines = v2compare.bench(ref, free[:3] + [dict(free[3], tokens=[4])])
    assert code == 1 and any("y#1000" in l for l in lines)
    assert v2compare.bench(ref, free[:2])[0] == 1


def test_v1_compare_fixture_unchanged(capsys):
    from tfpp.glm import compare

    rows = [{"t": "meta"}, {"t": "req", "n": 1, "prompt_len": 3}, {"t": "chunk", "start": 0, "R": 3, "head": [1, 2]},
            {"t": "step", "pos": 3, "R": 2, "window": [5, 6], "logits": [[1, 1], [2, 2]], "wall_ms": 3.0},
            {"t": "keep", "pos": 3, "R": 2, "keep": 1}, {"t": "done", "n": 1, "tokens": [5, 9]}]
    assert compare.compare(rows, [dict(r) for r in rows], show=0) == 0
    assert capsys.readouterr().out.splitlines() == [
        "request 1: IDENTICAL (5 records, 2 tokens, prompt 3 draft None)", "1/1 requests identical"]


# -- 8. static shape of the torch modules ---------------------------------------------------------------------------------
def _fn(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


class _Neutral(ast.NodeTransformer):
    IO = ("recv_act", "send_act", "recv_rows", "send_rows")

    def visit_Expr(self, node):
        v = node.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr in self.IO:
            return ast.Pass()
        return node


def test_compute_pp_multi_ast_differs_from_v1_only_in_the_prompt_io():
    v1 = _fn(ast.parse((GLM / "part.py").read_text(encoding="utf-8")), "_compute_pp")
    v2 = _fn(ast.parse((GLM / "v2part.py").read_text(encoding="utf-8")), "_compute_pp_multi")

    def body(fn):
        stmts = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        stmts = stmts[1:]                                    # the first assignment binds S, c (and fw, glue)
        return [ast.dump(_Neutral().visit(copy.deepcopy(s))) for s in stmts]

    assert ast.dump(v1.args) == ast.dump(v2.args)
    assert body(v1) == body(v2)
    # and the two I/O calls are exactly where v1 has its two
    assert sum(1 for n in ast.walk(v1) if isinstance(n, ast.Attribute) and n.attr in ("recv_act", "send_act")) == 2
    assert sum(1 for n in ast.walk(v2) if isinstance(n, ast.Attribute) and n.attr in ("recv_rows", "send_rows")) == 2


def test_v2_sources_keep_the_p1_seams_out():
    """F-9: v1's seams (SNAP / SAVE / LOAD / KEEP frames) are never installed under --parallel; v2 never touches v1's
    modules' attributes except through its own install (``part.STAGE`` / ``fw._compute``)."""

    tail = (GLM / "v2tail.py").read_text(encoding="utf-8")
    assert "install_tail_seams_v1" not in tail and "install_common" not in tail
    for p in sorted(GLM.glob("v2*.py")):
        src = p.read_text(encoding="utf-8")
        assert "settimeout" not in src or p.name == "v2bench.py", p.name
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in (
                            "part", "wire", "proto", "capacity", "startup", "tail", "trace", "proto"):
                        assert (p.name, t.attr) == ("v2part.py", "STAGE"), f"{p.name} assigns {t.value.id}.{t.attr}"


# -- 9. the torch modules under stand-ins ---------------------------------------------------------------------------------
def mia_pure() -> types.SimpleNamespace:
    """MIA's real pure helpers (seal, parse, ...) from the vendored multi.py, extracted by AST and run without torch."""

    import __future__

    src = (ROOT / "families" / "glm5_next" / "cuda" / "multi.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns: dict = {"np": np, "Sequence": typing.Sequence, "Any": typing.Any, "__builtins__": __builtins__}
    consts = {"OPS", "SEAL_MOD", "SHARED_SLOTS", "MAX_WINDOW", "DONE"}
    funcs = {"seal", "digest", "unseal", "trim", "multi_code", "_f64", "_unf64", "pack_sampling", "keyed_ids", "pack_keys",
             "unpack_keys"}

    def run(nodes):
        mod = ast.Module(body=nodes, type_ignores=[])
        exec(compile(mod, "multi_pure", "exec", flags=__future__.annotations.compiler_flag), ns)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)} | {
                e.id for t in node.targets if isinstance(t, ast.Tuple) for e in t.elts if isinstance(e, ast.Name)}
            if names & (consts | {"ADMIT"}):
                run([node])
        elif isinstance(node, ast.FunctionDef) and node.name in funcs:
            run([node])
        elif isinstance(node, ast.ClassDef) and node.name == "MultiDecoder":
            run([next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "parse")])
    return types.SimpleNamespace(**ns)


class _Auto(types.ModuleType):
    """A stand-in module: capitalised names are (subclassable) classes, the rest mocks; dunder names are not made up."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (), {}) if name[:1].isupper() and not name.isupper() else mock.MagicMock(name=name)
        setattr(self, name, value)
        return value


def _fake_torch():
    t = _Auto("torch")
    t.no_grad = lambda: (lambda f: f)
    t.cuda = mock.MagicMock()
    return t


STUB_PREFIXES = ("torch", "tensorfold", "tfpp.glm.part", "tfpp.glm.wire", "tfpp.glm.startup", "tfpp.glm.feed",
                 "tfpp.glm.tail", "tfpp.glm.follower", "tfpp.glm.trace", "tfpp.glm.load", "tfpp.glm.v2")
STUB_KEEP = ("tfpp.glm.v2proto", "tfpp.glm.v2bench", "tfpp.glm.v2capacity", "tfpp.glm.v2startup", "tfpp.glm.v2compare",
             "tfpp.glm.v2pins")


@pytest.fixture
def stubbed():
    """Fake ``torch`` and ``tensorfold`` in sys.modules, MIA's pure functions real, every tfpp.glm module that imports
    them imported fresh; all removed again afterwards."""

    import importlib

    mia = mia_pure()
    glm_pkg = sys.modules["tfpp.glm"]
    saved_attrs = dict(vars(glm_pkg))
    with mock.patch.dict(sys.modules):
        before = {k for k in sys.modules}
        for k in [k for k in sys.modules if k.startswith(STUB_PREFIXES) and not k.startswith(STUB_KEEP)]:
            del sys.modules[k]
        sys.modules["torch"] = _fake_torch()
        sys.modules["triton"] = _Auto("triton")
        root = _Auto("tensorfold")
        root.__path__ = []
        sys.modules["tensorfold"] = root
        names = ["tensorfold.cuda", "tensorfold.cuda.streams", "tensorfold.families", "tensorfold.families.glm5_next",
                 "tensorfold.families.glm5_next.cuda"] + [
            f"tensorfold.families.glm5_next.cuda.{m}" for m in
            ("decode", "drafter_choice", "engine", "forward", "multi", "multi_prefill", "verify", "weights", "dflash2")]
        for n in names:
            m = _Auto(n)
            m.__path__ = []
            sys.modules[n] = m
            parent, _, leaf = n.rpartition(".")
            setattr(sys.modules[parent], leaf, m)
        M = sys.modules["tensorfold.families.glm5_next.cuda.multi"]
        for k in ("ADMIT", "EVICT", "MOVE", "GROW", "FILL", "ROUND", "FINISH", "IDLE", "SLOT", "MFILL", "FILLPART",
                  "MFILLPART", "LOAD", "FLUSH", "SPILL", "LOADED", "TOUCH", "OPS", "SEAL_MOD", "SHARED_SLOTS"):
            setattr(M, k, getattr(mia, k))
        M.MultiDecoder = type("MultiDecoder", (), {"parse": staticmethod(mia.parse.__func__)})
        M.seal, M.unseal, M.digest = mia.seal, mia.unseal, mia.digest
        sys.modules["tensorfold.families.glm5_next.cuda.engine"].PARALLEL_MOST = 8
        streams = sys.modules["tensorfold.cuda.streams"]
        streams.Stream = type("Stream", (), {"__init__": lambda self, prompt, count, sampling=None, **kw: self.__dict__.update(
            prompt=prompt, count=count, sampling=sampling, out=[], done=False, **kw)})
        try:
            yield types.SimpleNamespace(mia=mia, M=M, import_module=lambda n: importlib.import_module(n))
        finally:
            for k in list(vars(glm_pkg)):                     # the package's attributes for the modules imported here
                if k not in saved_attrs:
                    delattr(glm_pkg, k)
            vars(glm_pkg).update(saved_attrs)


@needs_tree
def test_v2_modules_import_under_stubs(stubbed):
    for name in ("v2part", "v2wire", "v2follower", "v2tail", "v2trace", "v2check", "v2serve"):
        mod = stubbed.import_module(f"tfpp.glm.{name}")
        assert mod.__name__ == f"tfpp.glm.{name}"
    tail = sys.modules["tfpp.glm.v2tail"]
    assert issubclass(tail.PipeMultiEngine, sys.modules["tfpp.glm.tail"].PipeGlmEngine)
    assert tail.check_parallel(4, "d", False) == 4
    for bad in ((9, "d", False), (0, "d", False), (4, None, False)):
        with pytest.raises(ValueError):
            tail.check_parallel(*bad)
    assert tail.check_parallel(4, None, True) == 4


class FakeLink:
    def __init__(self, frames=()):
        self.frames, self.sent, self.raw_sent = list(frames), [], []
        self.name = "fake"
        self.sock = object()

    def recv(self, timeout=None, into=None, keep_raw=False, link_frame=False):
        return self.frames.pop(0)

    def send_raw(self, raw, payload):
        self.raw_sent.append((raw, list(payload)))

    def send(self, op, fields=None, parts=()):
        self.sent.append((op, fields, list(parts)))


def make_frame(wire, op, seq, fields, parts_spec, host):
    hdr = proto.make_header(op, seq, fields, parts_spec)
    return wire.Frame(hdr=hdr, raw=proto.encode_header(hdr), host=dict(host), payload=list(host.values()))


@needs_tree
def test_follower_link_side(stubbed):
    f = stubbed.import_module("tfpp.glm.v2follower")
    wire = sys.modules["tfpp.glm.wire"]
    sealed = stubbed.mia.seal([1, 2, 3, 4], 0)
    arr = np.asarray(sealed, dtype=np.int32)
    multi = make_frame(wire, "MULTI", 0, v2proto.multi_fields(0, sealed), [("ints", "i32", [len(sealed)])],
                       {"ints": arr.tobytes()})
    img = np.zeros(6, dtype=np.uint8).tobytes()
    multi_img = make_frame(wire, "MULTI", 1, v2proto.multi_fields(0, sealed),
                           [("ints", "i32", [len(sealed)]), ("img:7", "u8", [6])], {"ints": arr.tobytes(), "img:7": img})
    quit_ = make_frame(wire, "QUIT", 2, {}, [], {})
    err = make_frame(wire, "ERROR", 3, {"stage": "leader", "where": "x", "msg": "boom"}, [], {})
    wrong = make_frame(wire, "WINDOWS", 4, v2proto.windows_fields(0, 1, [[0, 1, 1]]), [("ids", "i32", [1])],
                       {"ids": np.zeros(1, dtype=np.int32).tobytes()})
    other_group = make_frame(wire, "MULTI", 5, v2proto.multi_fields(1, sealed), [("ints", "i32", [len(sealed)])],
                             {"ints": arr.tobytes()})

    def glm(first, frames):
        down, up = FakeLink(frames), None if first else FakeLink()
        fol = types.SimpleNamespace(
            e=object(), w=types.SimpleNamespace(cfg=types.SimpleNamespace(eos=[1, 2])),
            stage=types.SimpleNamespace(count=3, index=0 if first else 1, first=first),
            args=types.SimpleNamespace(model="/m", io_timeout=5.0), down=down, up=up, warm=contextlib.nullcontext())
        agree = {"grid": 0, "multi_window": 32, "limit": 100, "entries": 4}
        return f.FollowerGlm(fol, agree), down, up

    g, down, up = glm(False, [multi, quit_, err, wrong, other_group])
    assert g.rank == 1 and g.limit == 100 and g.cache_entries == 4
    assert g._share(None) == sealed
    assert up.raw_sent and up.raw_sent[0][0] == multi.raw, "a control frame goes to the stage before it is handled"
    with pytest.raises(f.Quit):
        g._share(None)
    assert len(up.raw_sent) == 2, "QUIT is forwarded before the stage stops"
    with pytest.raises(sys.modules["tfpp.glm.wire"].LinkError, match="boom"):
        g._share(None)
    with pytest.raises(proto.ProtocolError, match="expected MULTI"):
        g._share(None)
    with pytest.raises(proto.ProtocolError, match="g=0"):
        g._share(None)
    with pytest.raises(proto.ProtocolError):
        g._share([1, 2])                                         # a follower only receives
    # stage 0 keeps the rows an ADMIT brought
    g0, down0, _ = glm(True, [multi_img])
    assert g0._share(None) == sealed and 7 in g0.incoming
    g1, down1, _ = glm(False, [multi_img])
    g1._share(None)
    assert g1.incoming == {}, "only stage 0 keeps picture rows"
    # idle: the bell is down being readable; the stage before speaking is an error
    f.wait_any = lambda links, timeout=None: links[0]
    g._await_bell()
    f.wait_any = lambda links, timeout=None: links[1]
    up.frames = [make_frame(wire, "ERROR", 0, {"stage": 1, "where": "w", "msg": "gone"}, [], {})]
    with pytest.raises(sys.modules["tfpp.glm.wire"].LinkError, match="gone"):
        g._await_bell()
    up.frames = [make_frame(wire, "ACT", 0, v2proto.act_round(0, 0, 1), [], {})]
    with pytest.raises(proto.ProtocolError, match="unasked"):
        g._await_bell()
    assert f.REFUSED and all(isinstance(x, type) or isinstance(x, int) for x in f.REFUSED)


@needs_tree
def test_leader_wire_helpers(stubbed):
    t = stubbed.import_module("tfpp.glm.v2tail")
    link = FakeLink()
    v = t.V2Link(link)
    sealed = stubbed.mia.seal([5, 1, 9], 41)
    v.multi(sealed, [("ints", np.asarray(sealed, dtype=np.int32))])
    v.windows(41, [[0, 10, 2]], [4, 5])
    v.keeps(41, [2], [12])
    (op1, f1, _), (op2, f2, p2), (op3, f3, _) = link.sent
    assert (op1, f1) == ("MULTI", {"g": 0, "mseq": 41, "n": len(sealed)})
    assert (op2, f2["rid"], f2["segs"]) == ("WINDOWS", 41, [[0, 10, 2]]) and p2[0][1].tolist() == [4, 5]
    assert (op3, f3) == ("KEEPS", {"g": 0, "rid": 41, "keeps": [2], "end": [12]})
    # send_multi: the sealed message, and the picture rows of an ADMIT that brings them (popped, once)
    admit = [3, 1, 0, 2048, -1, 0, 8, 1, 1] + [0] * 10 + [0, 0] + [16, 7, 300000, 0] + [0]       # starts with sid 3
    msg = stubbed.mia.seal([stubbed.M.ADMIT, len(admit), *admit], 0)

    class Rows:
        def to(self, dtype):
            return self

        def contiguous(self):
            return "ROWS"

    eng = types.SimpleNamespace(multi=types.SimpleNamespace(outgoing={3: Rows()}), v2link=v,
                                v2fault=v2proto.Fault(""))
    link.sent.clear()
    t.send_multi(eng, msg)
    op, fields, parts = link.sent[0]
    assert op == "MULTI" and [p[0] for p in parts] == ["ints", "img:3"] and eng.multi.outgoing == {}
    link.sent.clear()
    t.send_multi(eng, msg)
    assert [p[0] for p in link.sent[0][2]] == ["ints"]


@needs_tree
def test_stream_tags_cancel_and_landing(stubbed):
    t = stubbed.import_module("tfpp.glm.v2tail")
    worker = threading.Thread(target=lambda: None)                   # the scheduler's worker thread (not started yet)

    class Flush:
        background = False
        emit = None

    sch = types.SimpleNamespace(waiting=queue.Queue(), thread=worker, _Flush=Flush)
    eng = types.SimpleNamespace(scheduler=sch, request=threading.local())
    t.install_stream_tags(eng)

    class S:
        background = False

        def __init__(self):
            self.glm, self.out, self.done, self.finished = {}, [], False, 0.0

        def take(self, new, eos=()):
            self.out.extend(new)

    eng.request.trace_tag = "alpha"
    eng.v2cancel["alpha"] = 3
    eng.v2landed.register("alpha")                                    # (a harness' tag: Lockstep registers its own)
    s = S()
    sch.waiting.put((s, queue.Queue()))
    assert s.glm["tfpp_tag"] == "alpha" and eng.v2landed.event("alpha").is_set()
    s.take([1, 2])
    assert not s.done
    s.take([3, 4])
    assert s.done and s.out == [1, 2, 3, 4], "the cap ends the stream on the worker once it holds K tokens"
    # no tag set: req<n>, numbered
    eng.request.trace_tag = None
    s2 = S()
    sch.waiting.put((s2, queue.Queue()))
    assert s2.glm["tfpp_tag"] == "req1"
    assert "req1" not in eng.v2landed._events, "an unregistered tag (every HTTP request) leaves nothing behind"
    # a replay (give-back, background yield): a fresh Stream with the same glm is capped again, once
    replay = S()
    replay.glm = s.glm
    sch.waiting.put((replay, queue.Queue()))
    assert replay._tfpp_capped and not replay.done
    replay.take([1, 2])
    assert not replay.done
    replay.take([3])
    assert replay.done and replay.out == [1, 2, 3]
    sch.waiting.put((replay, queue.Queue()))                          # put again (a held stream): not wrapped twice
    again_take = replay.take
    sch.waiting.put((replay, queue.Queue()))
    assert replay.take is again_take
    for _ in range(3):
        sch.waiting.get_nowait()
    # the worker thread's own put (a replay: the same glm) signals nobody and keeps the tag it had
    again, fresh = S(), S()
    again.glm = s2.glm
    fresh.glm = {"tfpp_tag": "w1"}
    done = threading.Event()

    def worker_puts():
        sch.waiting.put((again, queue.Queue()))
        sch.waiting.put((fresh, queue.Queue()))
        done.set()

    worker.run = worker_puts
    worker.start()
    assert done.wait(5)
    assert again.glm["tfpp_tag"] == "req1" and not eng.v2landed.event("w1").is_set()
    sch.waiting.put((Flush, threading.Event()))                       # a wake-up item passes through untouched
    assert sch.waiting.qsize() == 5


def test_landings_register_signal_forget():
    ln = v2bench.Landings()
    ln.signal("http1")                                                # nobody registered it: nothing is kept
    assert ln._events == {}
    ev = ln.register("a")
    ln.signal("a")
    assert ev.is_set()
    ln.forget("a")
    ln.forget("a")
    assert ln._events == {}


@needs_tree
def test_cancelled_filling_lane_is_reaped_without_a_fill_budget(stubbed):
    t = stubbed.import_module("tfpp.glm.v2tail")
    base = stubbed.M.MultiDecoder
    base._iterate = lambda self: ["mia's iteration"]
    ext = types.SimpleNamespace
    gone = ext(done=False, emit=ext(cancelled=lambda: True), started=0.0, finished=0.0)
    alive = ext(done=False, emit=ext(cancelled=lambda: False), started=0.0, finished=0.0)
    deco = ext(done=False, emit=ext(cancelled=lambda: True), started=0.0, finished=0.0)
    d = object.__new__(t.PipeMultiDecoder)
    d.fill_budget = 0.0
    d.lanes = {1: ext(s=gone, decoding=False), 2: ext(s=alive, decoding=False), 3: ext(s=deco, decoding=True)}
    assert d._iterate() == [gone] and gone.done and gone.finished > 0
    assert not alive.done and not deco.done, "a decoding lane ends at its next round, as MIA's"
    d.lanes = {2: ext(s=alive, decoding=False)}
    assert d._iterate() == ["mia's iteration"]
    d.fill_budget = 10.0                                              # with a budget MIA's own iteration reaps
    d.lanes = {1: ext(s=ext(done=False, emit=ext(cancelled=lambda: True)), decoding=False)}
    assert d._iterate() == ["mia's iteration"]


@needs_tree
def test_warmup_eviction_event_waits_for_the_iterations_finish(stubbed):
    """warmup_multi returns when the eviction event is set: only after the same iteration's finish, which sends its
    IDLE, so the traces of two starts begin alike (the event used to be set at the end of round)."""

    t = stubbed.import_module("tfpp.glm.v2tail")
    base = stubbed.M.MultiDecoder
    calls = []
    base.round = lambda self: calls.append("round") or []
    base.finish = lambda self, done: calls.append("finish")
    ext = types.SimpleNamespace
    d = object.__new__(t.PipeMultiDecoder)
    d.g = ext(run_lock=threading.Lock(), closing=False, broken=None, link=None)
    d.lanes, d.kept, d.cleared = {}, [], None
    ev = threading.Event()
    d.clear_due = ev
    assert d.round() == []
    assert not ev.is_set(), "the iteration's IDLE goes out in finish: not at rest yet"
    d.finish([])
    assert ev.is_set() and calls == ["round", "finish"] and d.cleared is None and d.clear_due is None


def test_kept_state_of_the_leader_holds_the_whole_draft_window():
    cfg = json.loads((Path(__file__).parent / "data" / "dflash2_config.json").read_text(encoding="utf-8"))
    mib = 2 ** 20
    # 5 layers x (K, V) x 8 kv heads x 2048 rows x 128 x 2 B = 40 MiB whole; a TP2 rank (4 heads) holds 20 of them
    assert v2capacity.draft_window_extra(cfg) == 20 * mib
    assert v2capacity.draft_window_extra(None) == 0
    assert v2capacity.draft_window_extra({**cfg, "sliding_window": 0}) == 0

    def slot(_text, _world, rows=0, rank=0):
        return 64 * mib

    ranges = [(0, 19), (19, 26), (26, 45)]
    from tfpp.glm import capacity as cap

    cap_sub = cap.sub_config
    try:
        cap.sub_config = lambda text, a, b, mtp=False: text
        plain = v2capacity.stage_states({}, ranges, slot_bytes=slot)
        drafted = v2capacity.stage_states({}, ranges, slot_bytes=slot, drafter_cfg=cfg)
    finally:
        cap.sub_config = cap_sub
    assert plain == [(32 + 22) * mib] * 3
    assert drafted == [(32 + 22) * mib, (32 + 22) * mib, (32 + 22 + 20) * mib], "only the last stage holds the drafter"


@needs_tree
def test_trace_helpers(stubbed):
    tr = stubbed.import_module("tfpp.glm.v2trace")
    ext = types.SimpleNamespace
    pool = ext(free_rows=lambda: 100, extents=[ext(eid=1, base=0, size=2048, owner=5), ext(eid=2, base=2048, size=2048, owner=None)])
    multi = ext(pool=pool, kept=[ext(kid=0, ids=[1, 2, 3], extent=pool.extents[1])],
                lanes={5: ext(sid=5, slot=1, st=ext(pos=77), decoding=True)})
    assert tr.pool_summary(multi) == {"free": 100, "extents": [[1, 0, 2048, 5], [2, 2048, 2048, None]],
                                      "kept": [[0, 3, 2]], "lanes": [[5, 1, 77, True]]}
    assert tr.WORKER_KINDS == v2compare.WORKER_KINDS and tr.CLIENT_KINDS == v2compare.CLIENT_KINDS
    assert tr.make("", "leader", (0, 8)) is None


@needs_tree
def test_control_frames_travel_the_chain_over_real_links(stubbed):
    """Real ``wire.Link`` ends over socket pairs: the leader's V2Link frames reach stage 1, which forwards every one
    upstream byte for byte before it handles it, and stage 0 (which keeps the picture rows) receives the same."""

    import socket

    t = stubbed.import_module("tfpp.glm.v2tail")
    f = stubbed.import_module("tfpp.glm.v2follower")
    wire = sys.modules["tfpp.glm.wire"]
    pairs = [socket.socketpair(), socket.socketpair()]
    try:
        leader = wire.Link(pairs[0][0], "leader")
        s1_down, s1_up = wire.Link(pairs[0][1], "s1.down"), wire.Link(pairs[1][0], "s1.up")
        s0_down = wire.Link(pairs[1][1], "s0.down")

        def glm(index, down, up):
            fol = types.SimpleNamespace(
                e=object(), w=types.SimpleNamespace(cfg=types.SimpleNamespace(eos=[1])),
                stage=types.SimpleNamespace(count=3, index=index, first=index == 0),
                args=types.SimpleNamespace(model="/m", io_timeout=10.0), down=down, up=up, warm=contextlib.nullcontext())
            return f.FollowerGlm(fol, {"grid": 0, "multi_window": 32, "limit": 10, "entries": 1})

        g1, g0 = glm(1, s1_down, s1_up), glm(0, s0_down, None)
        v = t.V2Link(leader)
        sealed = stubbed.mia.seal([5, 1, 9], 0)
        v.multi(sealed, [("ints", np.asarray(sealed, dtype=np.int32)), ("img:7", np.arange(6, dtype=np.uint8).reshape(2, 3))])
        v.windows(0, [[0, 10, 2], [1, 40, 3]], [4, 5, 6, 7, 8])
        v.keeps(0, [2, 3], [12, 43])
        leader.send("QUIT", {})
        assert g1._share(None) == sealed and g0._share(None) == sealed
        assert 7 in g0.incoming, "stage 0 keeps the picture rows of an ADMIT"
        w1, w0 = g1.next_frame("WINDOWS"), g0.next_frame("WINDOWS")
        assert w1.hdr["segs"] == w0.hdr["segs"] == [[0, 10, 2], [1, 40, 3]] and w0.ints("ids").tolist() == [4, 5, 6, 7, 8]
        k1, k0 = g1.next_frame("KEEPS"), g0.next_frame("KEEPS")
        assert k1.hdr["keeps"] == k0.hdr["keeps"] == [2, 3] and k0.hdr["end"] == [12, 43]
        assert (k1.hdr["op"], k1.hdr["g"], k1.hdr["rid"]) == ("KEEPS", 0, 0)
        with pytest.raises(f.Quit):
            g1.next_frame("MULTI")
        with pytest.raises(f.Quit):
            g0.next_frame("MULTI")
        # an out-of-order frame is the chain losing step
        v.keeps(1, [1], [1])
        with pytest.raises(proto.ProtocolError, match="expected WINDOWS"):
            g1.next_frame("WINDOWS")
    finally:
        for a, b in pairs:
            a.close()
            b.close()


@pytest.mark.skipif(__import__("shutil").which("bash") is None, reason="no bash")
def test_scripts_parse_and_name_only_existing_modes():
    import re
    import subprocess

    scripts = T / "scripts" / "glm"
    for name in ("run_v2.sh", "start_stage.sh", "run_truncated.sh"):
        done = subprocess.run(["bash", "-n", str(scripts / name)], capture_output=True, text=True)
        assert done.returncode == 0, f"{name}: {done.stderr}"
    text = (scripts / "run_v2.sh").read_text(encoding="utf-8")
    body = text.split("case $MODE in")[1].split("esac")[0]
    modes = set(re.findall(r"(?:^|;;)\s*([A-Za-z0-9-]+)\)", body, re.M)) - {"all"}
    assert {"T0a-v2", "T0b-v2", "T1-v2", "T2m", "T1-v2-p1", "T1-v2-free", "T0e", "T1-v2-fault", "T1-v2-http"} <= modes
    # every flag the script gives the leader or a follower is a real one of serve.py's parser
    from tfpp.glm import serve

    known = {a for act in serve.build_parser()._actions for a in act.option_strings}
    other = {"--max-time", "--url", "--concurrency", "--out", "--rounds", "--check-out"}     # curl, v2bench http, v2check
    unknown = {f for f in set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", text)) if f not in known and f not in other}
    assert not unknown, f"run_v2.sh passes flags serve.py does not have: {sorted(unknown)}"
