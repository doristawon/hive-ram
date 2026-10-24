"""T-S (spec 15.6): what tfpp-ds41 can check on any CPU Python, without torch or a GPU.

  cd tfpp && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tfpp/ds41/tests
(the file's name differs from tfpp-glm's tests/test_static.py, so one pytest run collects both suites)

The fork's tree is found at TFPP_DS41_VENDOR, ../vendor/tensorfold-dsv41-tp2/src/tensorfold beside this checkout, or
the installed tensorfold; nothing is written into it. tests/data/ds41_config.json is the public checkpoint's
config.json (Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw).
"""

from __future__ import annotations

import json
import os
import py_compile
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
T = HERE.parents[2]                                        # the tfpp checkout (tfpp/ds41/tests -> tfpp)
if str(T) not in sys.path:
    sys.path.insert(0, str(T))

from tfpp.ds41 import (capacity, compare, env, load, oracle, patch, pins, plan as PL, proto,  # noqa: E402
                       testmodel)

RAW = json.loads((HERE / "data" / "ds41_config.json").read_text(encoding="utf-8"))
MV = PL.ModelView.from_config(RAW)
S11_RAW = load.selection_config(RAW, load.S11, engram_vocab=320000)
S11 = PL.ModelView.from_config(S11_RAW)
DS41 = T / "tfpp" / "ds41"


def vendor_root() -> Path | None:
    cands = [os.environ.get("TFPP_DS41_VENDOR", ""), str(T.parent / "vendor" / "tensorfold-dsv41-tp2" / "src" /
                                                          "tensorfold")]
    for c in cands:
        if c and (Path(c) / "__init__.py").is_file():
            return Path(c)
    try:
        return pins.installed_root()
    except RuntimeError:
        return None


ROOT = vendor_root()
needs_tree = pytest.mark.skipif(ROOT is None, reason="no fork tree found")


# -- the fork and the seams ----------------------------------------------------------------------------------------
@needs_tree
def test_tree_digest_is_the_fork():
    # a Windows work tree is CRLF: its LF reading is the digest git stores (the raw one is refused at start)
    assert pins.tree_sha_lf(ROOT) == pins.EXPECT_PKG_SHA
    assert pins.family_sha(ROOT) == pins.FAMILY_SHA
    raw = pins.tree_sha(ROOT)
    if raw != pins.EXPECT_PKG_SHA:
        assert raw.startswith(pins.CRLF_PKG_SHA)
        with pytest.raises(pins.TreeMismatch, match="CRLF"):
            pins.check_tree(ROOT)


@needs_tree
def test_every_seam_pinned_and_unchanged():
    assert len(pins.PINS) == len(pins.SEAMS)
    assert pins.diff(ROOT) == []
    assert all(v != "missing" for v in pins.pin_hashes(ROOT).values())


@needs_tree
def test_anchors_once_and_compile():
    lines = patch.check_tree(ROOT / "families" / "deepseek_v41" / "cuda", carry_post=True)
    assert len(lines) == 5
    for aid in ("M1", "M2", "R1", "R2", "R3", "MU1", "F1a", "F1b", "F1c", "F1d"):
        file, qual, old, _ = patch.ANCHORS[aid]
        src = patch.function_source(ROOT / "families" / "deepseek_v41" / "cuda" / file, qual)
        assert src.count(old) == 1, aid


def test_anchor_missing_refuses():
    with pytest.raises(patch.LoaderChanged):
        patch.apply_anchors("def f():\n    pass\n", ["M1"], "test")
    assert patch.group_ids("forward_k", False) == ("M1", "M2")
    assert patch.group_ids("forward_k", True) == ("M1", "M2", "F1c", "F1d")


# -- cut rules (spec 5.3 / 5.4) --------------------------------------------------------------------------------------
def carry(mv, c):
    k = PL.carry_into(mv, c)
    return (k.kv_layer, k.topk, k.cand, k.reps)


def test_real_cut_table():
    MV.check()
    assert carry(MV, 1) == (None, None, False, ())
    for c in (2, 8, 14, 20):
        assert carry(MV, c) == (None, None, False, ()), c
    for c, src in [(c, 2) for c in range(3, 8)] + [(c, 8) for c in range(9, 14)] + [(c, 14) for c in range(15, 20)]:
        assert carry(MV, c) == (src, (src, 2), False, ((src, False),)), c
    for c in (21, 22, 23):
        assert carry(MV, c) == (20, (20, 1), True, ((20, True),)), c
    for c in (24, 28, 32, 36):
        assert carry(MV, c) == (20, None, True, ((20, True),)), c
    for c, p in [(c, 24) for c in (25, 26, 27)] + [(c, 28) for c in (29, 30, 31)] + [(c, 32) for c in (33, 34, 35)]:
        assert carry(MV, c) == (20, (p, 1), True, ((20, True),)), c
    assert carry(MV, 37) == (20, (36, 1), False, ((20, False),))
    for c in (38, 39):
        with pytest.raises(PL.PlanError, match="tap"):
            PL.plan_chain(MV, [(0, c), (c, 40)], drafts=True)
        PL.plan_chain(MV, [(0, c), (c, 40)], drafts=False)          # --no-drafts relaxes the tap rule


def test_s11_selection_config():
    t = S11_RAW["text_config"]
    assert t["num_hidden_layers"] == 11
    assert t["compress_ratios"] == [0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert t["kv_source_layer_ids"] == [2, 4, 5]
    assert t["index_source_layer_ids"] == [2, 4, 5, 7]
    assert t["candidate_source_layer_id"] == 5
    assert t["engram_layer_ids"] == [1, 4]
    assert t["engram_vocab_size"] == 320000
    assert t["engram_num_embeddings"] == [7682786, 7689630]
    assert all(r > testmodel.AIO_PROBE_ROW for r in t["engram_num_embeddings"])
    assert t["dspark_target_layer_ids"] == [8, 9, 10]
    assert S11.half == 5
    with pytest.raises(ValueError, match="kv source"):
        load.selection_config(RAW, [0, 1, 3, 14, 20, 21, 24, 37, 38, 39])        # 3 without its source 2
    with pytest.raises(ValueError, match="tap"):
        load.selection_config(RAW, [0, 1, 2, 3, 14, 20, 21, 24, 37, 38])         # a tap layer left out


def test_s11_cut_table():
    S11.check()
    assert carry(S11, 1) == (None, None, False, ())
    assert carry(S11, 2) == (None, None, False, ())
    assert carry(S11, 3) == (2, (2, 2), False, ((2, False),))
    assert carry(S11, 4) == (None, None, False, ())
    assert carry(S11, 5) == (None, None, False, ())
    assert carry(S11, 6) == (5, (5, 1), True, ((5, True),))
    assert carry(S11, 7) == (5, None, True, ((5, True),))
    assert carry(S11, 8) == (5, (7, 1), False, ((5, False),))
    for c in (9, 10):
        with pytest.raises(PL.PlanError):
            PL.plan_chain(S11, [(0, c), (c, 11)], drafts=True)


def test_stage_plans_and_chains():
    p = PL.plan_chain(S11, [(0, 3), (3, 11)], drafts=True)
    assert p[0].planes == (2,) and p[0].index_planes == (2,) and p[0].engram_owned == (1,)
    assert p[1].planes == (2, 4, 5) and p[1].index_planes == (4, 5) and p[1].engram_owned == (4,)
    assert p[1].own_raw == (4,) and p[0].own_raw == (2,)
    p = PL.plan_chain(S11, [(0, 6), (6, 11)], drafts=True)
    assert p[1].planes == (5,) and p[1].index_planes == (5,) and p[1].after_replay and not p[1].own_sources
    p3 = PL.plan_chain(S11, [(0, 3), (3, 6), (6, 11)], drafts=True)       # a middle stage forwards
    assert p3[1].planes == (2, 4, 5) and p3[1].cout.reps == ((5, True),)
    p4 = PL.plan_chain(S11, [(0, 2), (2, 5), (5, 8), (8, 11)], drafts=True)
    assert [x.index for x in p4] == [0, 1, 2, 3] and p4[2].planes == (5,) and p4[3].planes == (5,)
    assert p4[2].index_planes == (5,) and p4[3].index_planes == ()
    for bad in ([(0, 3), (4, 11)], [(0, 3), (3, 10)], [(0, 0), (0, 11)], []):
        with pytest.raises(PL.PlanError):
            PL.plan_chain(S11, bad, drafts=True)
    assert PL.chain_digest(p) == PL.chain_digest(PL.plan_chain(S11, [(0, 6), (6, 11)], drafts=True))
    assert PL.chain_digest(p) != PL.chain_digest(p3)
    nd = PL.plan_chain(S11, [(0, 9), (9, 11)], drafts=False)
    assert nd[1].up_taps == 1 and not nd[1].drafts


def test_real_placements_plan():
    for ranges in ([(0, 17), (17, 25), (25, 40)], [(0, 17), (17, 24), (24, 40)], [(0, 14), (14, 23), (23, 40)],
                   [(0, 7), (7, 25), (25, 40)]):
        plans = PL.plan_chain(MV, ranges, drafts=True)
        assert plans[-1].last and plans[0].first


# -- prompt chunks (spec 6.2) ----------------------------------------------------------------------------------------
def test_chunk_in():
    p = PL.plan_chain(S11, [(0, 6), (6, 11)], drafts=True)
    before, after = p[0], p[1]
    assert PL.chunk_in(before, 0, 2048, 4000) == ("chunk", 0, 2048, 0)
    assert PL.chunk_in(after, 0, 2048, None) == ("chunk", 0, 2048, 0)        # no replay: the whole chunk
    assert PL.chunk_in(after, 0, 2048, 4000) == ("kv", 2048, 0, 4000)        # encoder-only there
    assert PL.chunk_in(after, 2048, 2048, 4000) == ("chunk", 4000, 96, 4000)
    assert PL.chunk_in(after, 4096, 17, 4000) == ("chunk", 4096, 17, 4000)
    assert PL.chunk_in(after, 0, 130, 2) == ("chunk", 2, 128, 2)
    assert PL.chunk_in(after, 2048, 2048, 4096) == ("kv", 4096, 0, 4096)     # first == n: encoder-only
    owner = PL.plan_chain(S11, [(0, 5), (5, 11)], drafts=True)[1]           # owns the replay layer: whole chunks
    assert not owner.after_replay and PL.chunk_in(owner, 2048, 2048, 4000) == ("chunk", 2048, 2048, 0)
    assert PL.replay_row(4500, 128, True) == 4372 and PL.replay_row(5, 128, True) == 0
    assert PL.replay_row(4500, 128, False) is None


def test_part_shapes():
    k6 = PL.carry_into(S11, 6)
    parts = PL.chunk_parts(S11, k6, 2048, 2048, 4000, 96)
    d = {n: (t, s) for n, t, s in parts}
    assert d["h"] == ("bf16", [96, 4, 5120]) and d["pre"] == ("f32", [96, 4])
    assert d["topk"] == ("i64", [96, 512])                                  # min(512, 4096 // 1)
    assert d["cand"] == ("u8", [96, 512])                                   # ceil(4096 / 8)
    assert d["c5"] == ("u8", [2048, 256]) and d["cs5"] == ("u8", [2048, 32])
    assert d["k5"] == ("u8", [2048, 64]) and d["ks5"] == ("u8", [2048, 4])
    k3 = PL.carry_into(S11, 3)
    d = {n: (t, s) for n, t, s in PL.chunk_parts(S11, k3, 0, 5, 0, 5)}
    assert d["topk"] == ("i64", [5, 2]) and d["c2"] == ("u8", [2, 256]) and "k2" not in d and "cand" not in d
    d = {n: (t, s) for n, t, s in PL.chunk_parts(S11, k3, 0, 1, 0, 1)}
    assert d["topk"] == ("i64", [1, 0]) and d["c2"] == ("u8", [0, 256])     # n_comp_end == 0 (F/model.py:1049)
    assert [n for n, _, _ in PL.kv_parts(S11, k6, 0, 2048)] == ["c5", "cs5", "k5", "ks5"]
    f1 = {n: s for n, _, s in PL.chunk_parts(S11, k6, 0, 33, 0, 33, f1=True)}
    assert f1["pend"] == [1, 33, 5120] and f1["post"] == [33, 4] and f1["comb"] == [33, 4, 4]
    # decode rounds (spec 7.1): the candidate pool as a mask up to 16384 entries, block ids past them
    assert PL.cand_form(S11, 16384, idx_on=True, cand_only_on=True) == "cand"
    assert PL.cand_form(S11, 32768, idx_on=True, cand_only_on=True) == "cblk"
    assert PL.cand_form(S11, 32768, idx_on=False, cand_only_on=True) == "cand"
    r = {n: (t, s) for n, t, s in PL.round_parts(S11, k6, 6, 1024, "cand")}
    assert r["topk"] == ("i64", [6, 512]) and r["cand"] == ("u8", [6, 128]) and r["c5"] == ("u8", [6, 256])
    r = {n: (t, s) for n, t, s in PL.round_parts(S11, k6, 16, 65536, "cblk")}
    assert r["cblk"] == ("i32", [16, 2048]) and "cand" not in r
    r = {n: (t, s) for n, t, s in PL.round_parts(S11, k3, 4, 1024, "cand")}
    assert r["topk"] == ("i64", [4, 512])                                   # min(512, 1024 // 2)
    assert PL.buckets(8206) == [1024, 2048, 4096, 8192, 8206]
    assert PL.buckets(1000) == [1024]
    assert PL.parts_bytes(PL.round_parts(S11, PL.carry_into(S11, 5), 6, 1024, "cand")) == 6 * 40976


# -- frames, the stage info, the digest ------------------------------------------------------------------------------
def test_frame_roundtrip_and_stage_info():
    hdr = proto.make_header(proto.ACT, 3, {"kind": "round", "round": 7, "R": 2, "b": 1024, "ck": "cand"},
                            [("h", "bf16", [2, 4, 8]), ("topk", "i64", [2, 3]), ("cand", "u8", [2, 1])])
    payload = [b"\x01" * 128, b"\x02" * 48, b"\x03" * 2]
    got, parts, used = proto.split_frame(proto.frame_bytes(hdr, payload) + b"rest")
    assert got["op"] == "ACT" and got["seq"] == 3 and got["round"] == 7
    assert parts["topk"] == b"\x02" * 48 and used == len(proto.frame_bytes(hdr, payload))
    s = proto.DsStageInfo(layers=(3, 11), role="leader", settings={"a": 1}, versions={"torch": "2.13.0"},
                          extra={"engram": None})
    assert proto.DsStageInfo.from_json(json.loads(json.dumps(s.to_json()))) == s
    t = proto.DsStageInfo(layers=(0, 3), role="follower", settings={"a": 2}, versions={"torch": "2.13.0"})
    probs = proto.compare_stages([t, s])
    assert len(probs) == 1 and "setting a" in probs[0]
    assert proto.digest([1, [0, 1]]) == proto.digest([1, [0, 1]]) != proto.digest([1, [1, 0]])
    assert proto.parse_layers("3:11") == (3, 11)
    with pytest.raises(ValueError):
        proto.parse_layers("5:5")


def test_check_engram():
    ranges = [[0, 3], [3, 11]]
    e1 = {"engram": {"layers": [1], "mode": "test-dense"}, "token_map_sha": "x"}
    e4 = {"engram": {"layers": [4], "mode": "test-dense"}, "token_map_sha": "x"}
    assert proto.check_engram(ranges, [1, 4], [e1, e4]) == []
    assert proto.check_engram(ranges, [1, 4], [e1, {"engram": None}])
    assert proto.check_engram(ranges, [1, 4], [e1, dict(e4, engram={"layers": [4], "mode": "real"})])
    assert proto.check_engram(ranges, [1, 4], [e1, dict(e4, token_map_sha="y")])
    assert proto.check_engram([[0, 6], [6, 11]], [1, 4], [{"engram": {"layers": [1, 4], "mode": "real"}},
                                                          {"engram": {"layers": [4], "mode": "real"}}])


def test_check_engram_wants_exactly_the_owned_tables():
    # stage [0, 3) owns L1 only: the "both" folder (L1 + L4) passes the old ownership test but StageEngram would refuse
    # it after every stage loaded its weights; HELLO refuses it now
    both = {"engram": {"layers": [1, 4], "mode": "test-dense"}, "token_map_sha": "x"}
    e4 = {"engram": {"layers": [4], "mode": "test-dense"}, "token_map_sha": "x"}
    probs = proto.check_engram([[0, 3], [3, 11]], [1, 4], [both, e4])
    assert len(probs) == 1 and "also hold layers [4]" in probs[0] and "stage 0" in probs[0], probs
    # the owner of both layers takes both tables
    assert proto.check_engram([[0, 6], [6, 11]], [1, 4], [both, {"engram": None}]) == []
    # a missing layer is still named once (the old message), not twice
    probs = proto.check_engram([[0, 6], [6, 11]], [1, 4], [dict(e4), {"engram": None}])
    assert len(probs) == 1 and "not 1" in probs[0], probs


# -- the cut's state on the wire (no torch: a stand-in module just for the import) ---------------------------------
def _wire_without_torch(monkeypatch):
    import importlib.util
    import types

    fake = types.ModuleType("torch")
    for n in ("bfloat16", "float32", "int32", "int64", "uint8", "bool"):
        setattr(fake, n, f"torch.{n}")
    fake.zeros = lambda shape, dtype=None, device=None: types.SimpleNamespace(shape=list(shape), dtype=dtype)
    glm_wire = types.ModuleType("tfpp.glm.wire")

    class LinkError(RuntimeError):
        pass

    glm_wire.Link, glm_wire.LinkError, glm_wire.wait_any = type("Link", (), {}), LinkError, lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "tfpp.glm.wire", glm_wire)
    spec = importlib.util.spec_from_file_location("tfpp.ds41._wire_under_test", DS41 / "wire.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_send_chunk_refuses_a_missing_candidate_pool(monkeypatch):
    wire = _wire_without_torch(monkeypatch)
    P = PL.plan_chain(S11, [(0, 6), (6, 11)], drafts=True)
    assert P[0].cout.cand and P[0].cout.topk is not None
    io = wire.StageIO(P[0], S11, None, None)
    # the cut carries the pool but the stage computed none: an error, never an all-zero ("no candidate") mask
    with pytest.raises(RuntimeError, match="candidate pool"):
        io.send_chunk(None, 0, 1, 0, 1, 0, object(), object(), {"topk": object()})


# -- traces ------------------------------------------------------------------------------------------------------------
def test_trace_records_round_taps_only_with_a_drafter(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from tfpp.ds41 import trace

    monkeypatch.setattr(trace, "devhash_rows", lambda t: [[1, 2] for _ in range(len(t))])

    def engine(drafter):
        runner = NS(forward=lambda windows, replay=True: (["l0", "l1"], ["t0", "t1"]), bucket=lambda deepest: 1024)
        mu = NS(_admit=lambda *a: None, _fill=lambda s: None, runner=runner, streams={}, _round=lambda live, k: [])
        return NS(multi=mu, model=NS(forward=lambda *a, **k: None), drafter=drafter, generate=lambda *a, **k: {},
                  request=NS())

    # a --no-drafts leader past a tap layer holds zeros for the upstream taps where the reference has the real ones:
    # without a drafter nothing reads the taps, so they are not part of the trace (both runs alike)
    for drafter, recorded in ((object(), True), (None, False)):
        path = tmp_path / f"{recorded}.jsonl"
        tr = trace.Tracer(path, "leader", (6, 11))
        eng = engine(drafter)
        tr.install(eng)
        eng.multi.runner.forward([(0, 0, 4096, 10, [5, 6], [])])
        tr.close()
        step = [r for r in map(json.loads, path.read_text().splitlines()) if r["t"] == "step"]
        assert len(step) == 1 and step[0]["logits"] == [[1, 2], [1, 2]]
        assert ("taps" in step[0]) is recorded


# -- the environment ---------------------------------------------------------------------------------------------------
def test_check_env_sets_and_refuses():
    environ: dict = {}
    set_now = env.check_env("follower", environ, {}, triton_version="3.7.1")
    assert environ["TF_DS_KEEP"] == "0" and environ["TF_TP_WORLD"] == "1" and set_now["TF_DS_KV"] == "native"
    for bad in ({"TF_DS_KEEP": "1"}, {"NCCL_PROTO": "Simple"}, {"TF_DS_RANK_CACHE": "/x"}, {"TF_DS_RDMA_MB": "8"},
                {"TF_TP_MASTER": "h"}, {"TFPP_DS41_CUT_POST": "maybe"}):
        with pytest.raises(env.EnvError):
            env.check_env("leader", dict(bad), {}, triton_version="3.7.1")
    with pytest.raises(env.EnvError, match="already imported"):
        env.check_env("leader", {}, {"tensorfold.families.deepseek_v41.cuda.model": object()}, triton_version="3.7.1")
    with pytest.raises(env.EnvError, match="Triton"):
        env.check_env("leader", {}, {}, triton_version="3.6.0")
    env.check_env("leader", {"TFPP_DS41_ALLOW_TRITON": "1"}, {}, triton_version="3.6.0")


def test_env_digest_rules():
    assert env.digested("TF_DS_REPLAY_FLOOR") and env.digested("TF_EXL3_PDL") and env.digested("TRITON_F32_DEFAULT")
    assert env.digested("TFPP_DS41_CUT_POST") and env.digested("TF_TP_WORLD") and env.digested("TF_SEED")
    for own in ("TF_DS_ENGRAM", "TF_DS_TOKEN_MAP", "TF_DS_ENGRAM_AIO", "TF_DS_TIMING", "TRITON_CACHE_DIR",
                "TFPP_DS41_VENV", "TFPP_DS41_TF_ROOT", "TFPP_WARM_MS", "TF_API_KEY_FILE", "TF_DS_SECRET_THING",
                "HOME"):
        assert not env.digested(own), own
    a = env.env_digest({"TF_DS_REPLAY": "1", "TF_DS_ENGRAM": "/a"})
    assert a == env.env_digest({"TF_DS_REPLAY": "1", "TF_DS_ENGRAM": "/b"})
    assert a != env.env_digest({"TF_DS_REPLAY": "0", "TF_DS_ENGRAM": "/a"})
    names = [n for n, _ in env.source_files()]
    assert "ds41/plan.py" in names and "warm.py" in names and "glm/wire.py" in names and "glm/__init__.py" in names
    assert len(env.tfpp_sha()) == 64


def test_settings_vector_reads_modules_only_after_check():
    import inspect

    src = inspect.getsource(env.settings_vector)
    assert "from tensorfold.families.deepseek_v41.cuda import" in src     # imported inside, after check_env


# -- the loader's pure parts ---------------------------------------------------------------------------------------------
def test_stage_layers():
    layers = load.StageLayers(3, 6, 11, ["a", "b", "c"])
    assert len(layers) == 11 and list(layers) == ["a", "b", "c"] and layers[3] == "a" and layers[5] == "c"
    for bad in (2, 6, -1, True):
        with pytest.raises(IndexError):
            layers[bad]
    with pytest.raises(IndexError):
        layers[0:2]


def test_mine_and_needed_shards():
    kw = dict(first=False, last=True, drafter=True)
    assert load.mine("layers.8.ffn.gate.weight", {8, 9, 10}, **kw)
    assert not load.mine("layers.3.attn.wq_a.trellis", {8, 9, 10}, **kw)
    assert load.mine("embed.weight", {8}, **kw) and load.mine("mtp.2.norm.weight", {8}, **kw)
    assert not load.mine("embed.weight", {3}, first=False, last=False, drafter=True)
    assert load.mine("head.trellis", {8}, **kw) and not load.mine("head.trellis", {3}, first=True, last=False,
                                                                  drafter=False)
    assert not load.mine("layers.8.ffn.gate.bias_vl", {8}, **kw)
    assert not load.mine("vision.blocks.0.x", {8}, **kw)
    with pytest.raises(ValueError):
        load.mine("something.else", {0}, first=True, last=False, drafter=False)
    wm = {"embed.weight": "a", "layers.0.x": "a", "layers.1.x": "b", "layers.14.x": "c", "layers.20.x": "d",
          "layers.21.x": "d", "layers.39.x": "e", "head.trellis": "e", "norm.weight": "e", "mtp.0.x": "e",
          "mtp.2.x": "f", "layers.5.x": "g", "vision.a": "v", "odd.thing": "e"}
    files, unknown = load.needed_shards(wm, range(0, 3), first=True, last=False, drafter=True,
                                        rename=load.selection_namer(load.S11))
    assert files == ["a", "b"] and unknown == ["odd.thing"]
    files, _ = load.needed_shards(wm, range(4, 11), first=False, last=True, drafter=True,
                                  rename=load.selection_namer(load.S11))
    assert files == ["a", "c", "d", "e", "f"]                                # "g" (layer 5) is not selected
    n = load.selection_namer(load.S11)
    assert n("layers.24.attn.x") == "layers.7.attn.x" and n("layers.5.attn.x") == "~layers.5.attn.x"
    assert n("mtp.0.x") == "mtp.0.x"


def _fake_shard(path, names):
    import struct

    h = json.dumps({n: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for n in names}).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0\0")


def test_stage_files_takes_an_extracted_embedding(tmp_path):
    s1, s31, s38 = (testmodel.shard_name(k) for k in (1, 31, 38))
    wm = {"embed.weight": s1, "layers.0.x": s1, "layers.30.x": s31, "head.trellis": s38, "norm.weight": s38,
          "mtp.0.x": s38}
    kw = dict(first=False, last=True, drafter=True)
    # no extracted file: the leader reads shard 1 for the embedding
    files, _ = load.stage_files(tmp_path, wm, range(30, 40), **kw)
    assert files == sorted([s1, s31, s38])
    # `testmodel extract` output in the folder (a name the index does not use) stands in for shard 1
    _fake_shard(tmp_path / "embed.weight.safetensors", ["embed.weight"])
    files, _ = load.stage_files(tmp_path, wm, range(30, 40), **kw)
    assert files == sorted([s31, s38, "embed.weight.safetensors"])
    # a leader without drafts does not read the embedding at all
    files, _ = load.stage_files(tmp_path, wm, range(30, 40), first=False, last=True, drafter=False)
    assert files == sorted([s31, s38])
    # a stage that holds layer 0 needs the real shard (its other tensors), embedding file or not
    files, _ = load.stage_files(tmp_path, wm, range(0, 5), first=True, last=False, drafter=False)
    assert files == [s1]
    # a file the index names is never taken for the extract
    _fake_shard(tmp_path / s1, ["embed.weight", "layers.0.x"])
    assert load.embed_only_file(tmp_path, wm) == "embed.weight.safetensors"
    (tmp_path / "embed.weight.safetensors").unlink()
    assert load.embed_only_file(tmp_path, wm) is None


def test_source_revision(tmp_path):
    full = "45ee5d0f" + "a" * 32
    assert testmodel.source_revision(tmp_path) == "unknown"
    assert testmodel.source_revision(tmp_path, full) == full
    with pytest.raises(ValueError):
        testmodel.source_revision(tmp_path, "45ee5d0f<full-commit>")
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text(full + "\nabc\n123\n")
    assert testmodel.source_revision(tmp_path) == full and testmodel.source_revision(tmp_path, full) == full
    with pytest.raises(ValueError):
        testmodel.source_revision(tmp_path, "b" * 40)


def test_engram_rows_match_the_fork_rule():
    # the real config: engram_num_embeddings is the primes rule at 16M (FO/ops.py:261-277, 379-380)
    t = RAW["text_config"]
    assert load.engram_rows(t["engram_vocab_size"], 2, t["engram_max_ngram_size"], t["engram_n_heads"]) == \
        t["engram_num_embeddings"]


# -- the test model and its tables -------------------------------------------------------------------------------------
def test_mode_d_selftest(tmp_path):
    lines = testmodel.selftest(str(tmp_path))
    assert lines[-1].startswith("selftest ok")
    a = testmodel.write_table(tmp_path / "a.safetensors", 1, 300, "dense", seed=3)
    b = testmodel.write_table(tmp_path / "b.safetensors", 1, 300, "dense", seed=3)
    assert a.split()[-1] == b.split()[-1]                                    # the same bytes on every node
    ident = testmodel.table_identity([tmp_path / "a.safetensors"])
    assert ident["layers"] == [1] and ident["rows"]["layers.1.engram.embed.weight"] == 300


def test_build_selection_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(RAW))
    wm = {"embed.weight": testmodel.shard_name(1), "layers.0.x": testmodel.shard_name(1)}
    for k, o in enumerate(load.S11):
        wm[f"layers.{o}.attn.x"] = testmodel.shard_name(o + 1)
    wm.update({"head.trellis": testmodel.shard_name(38), "norm.weight": testmodel.shard_name(38),
               "mtp.0.x": testmodel.shard_name(38), "mtp.1.x": testmodel.shard_name(39),
               "layers.30.attn.x": testmodel.shard_name(31)})
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    for f in set(wm.values()):
        (src / f).write_bytes(b"")
    (src / "tokenizer.json").write_text("{}")
    out = tmp_path / "s11"
    try:
        files = testmodel.build(str(src), str(out), list(load.S11), 320000, log=lambda *_: None)
    except OSError as exc:                                                   # no symlinks on this Windows account
        pytest.skip(f"symlinks unavailable: {exc}")
    assert testmodel.shard_name(31) not in files and testmodel.shard_name(1) in files
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["text_config"]["num_hidden_layers"] == 11
    sel = load.read_selection(out)
    assert sel["orig_layers"] == list(load.S11) and sel["engram_vocab"] == 320000


def test_link_shard_keeps_a_regular_file(tmp_path):
    src = tmp_path / "src.safetensors"
    src.write_bytes(b"src")
    out = tmp_path / "shard.safetensors"
    out.write_bytes(b"real")
    notes: list = []
    testmodel.link_shard(src.resolve(), out, notes.append)                  # never deleted, only noted
    assert out.read_bytes() == b"real" and len(notes) == 1 and "kept" in notes[0]


def test_link_shard_replaces_stale_links(tmp_path):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    (tmp_path / "out").mkdir()
    link = tmp_path / "out" / "shard.safetensors"
    notes: list = []
    try:
        testmodel.link_shard(a.resolve(), link, notes.append)
    except OSError as exc:                                                   # no symlinks on this Windows account
        pytest.skip(f"symlinks unavailable: {exc}")
    assert link.read_bytes() == b"a" and notes == []
    testmodel.link_shard(a.resolve(), link, notes.append)                   # the same source: left alone
    assert notes == []
    testmodel.link_shard(b.resolve(), link, notes.append)                   # a rebuild from another source
    assert link.read_bytes() == b"b" and "relinked" in notes[-1]
    b.unlink()                                                               # a dangling link (exists() is False)
    testmodel.link_shard(a.resolve(), link, notes.append)
    assert link.read_bytes() == b"a" and len(notes) == 2


def test_download_commands():
    lines = testmodel.download_commands(index=None, revision="REV", local_dir="D")
    text = "\n".join(lines)
    assert "64,269,037,656" in text and text.count("model-000") == 11
    # REV is one line to edit and is checked: a placeholder never reaches hf download as a shell redirection
    assert "REV='REV'" in text and '--revision "$REV"' in text and "[0-9a-f]{32}" in text and "<full-commit>" not in text
    ph = "\n".join(testmodel.download_commands(index=None, revision=None, local_dir="D"))
    assert "REV='<FULL_40_HEX_COMMIT>'" in ph and "set -euo pipefail" in ph
    assert sum(b for _, b in testmodel.S11_SHARDS.values()) == 64269037656


# -- capacity ----------------------------------------------------------------------------------------------------------
def test_pool_bytes_and_caps():
    P = PL.plan_chain(S11, [(0, 6), (6, 11)], drafts=True)
    pb = capacity.pool_bytes(S11, P[1], 8206, 2, drafter=True)
    assert pb["ring"] == 5 * 2 * 144 * 512 * 2
    assert pb["comp"] == (8206 + 2) * 288 and pb["index"] == (8206 + 2) * 68 and pb["raw"] == 0
    assert pb["markov"] == 256 * 129280 * 2 and pb["total"] == sum(v for k, v in pb.items() if k != "total")
    pa = capacity.pool_bytes(S11, P[0], 8206, 2, drafter=False)
    assert pa["raw"] == 2 * 2 * 64 * 512 * 4 * 2 and "markov" not in pa
    rx, tx = capacity.stage_staging(S11, P[1], 2048, 8206, f1=False)
    assert tx == 0 and rx > 2048 * 4 * 5120 * 2
    caps = [{"layers": [0, 6], "need": 10, "budget": 20, "pool": 1, "workspace": 1, "staging": 1, "weights": 1},
            {"layers": [6, 11], "need": 30, "budget": 20, "pool": 1, "workspace": 1, "staging": 1, "weights": 1}]
    probs = capacity.check_caps(caps)
    assert len(probs) == 2 and "[6, 11]" in probs[0]
    # staging is taken out of a measured budget (measure_budget), so it is not charged in the need again; a declared
    # --budget-gib is outright and keeps it
    G = 2 ** 30
    base = {"layers": [6, 11], "need": 10 * G, "budget": 8 * G, "pool": G, "workspace": 6 * G, "staging": 3 * G,
            "weights": G}
    assert capacity.check_caps([dict(base, declared=False)]) == []
    assert capacity.check_caps([dict(base, declared=True)])
    assert capacity.check_caps([dict(base)])                           # unknown: the conservative reading
    # stages on one GPU together: weights + need + a context each within the GPU's total less the reserve
    sh = [dict(layers=[0, 6], need=5 * G, budget=40 * G, pool=G, workspace=4 * G, staging=0, weights=50 * G,
               host="node-a", total=102 * G, reserve=3 * G, declared=True),
          dict(layers=[6, 11], need=5 * G, budget=40 * G, pool=G, workspace=4 * G, staging=0, weights=45 * G,
               host="node-a", total=102 * G, reserve=3 * G, declared=True)]
    probs = capacity.check_caps(sh)
    assert any("share node-a" in x for x in probs), probs
    sh[1]["weights"] = 30 * G
    assert capacity.check_caps(sh) == []
    sh[1]["host"] = "node-b"                                            # another machine's GPU: not summed
    sh[0]["weights"] = 90 * G
    assert capacity.check_caps(sh) == []
    built = [{"layers": [0, 6], "graphs": [[1, 1024]], "switches_sha": "a", "free_after": 5 << 30},
             {"layers": [6, 11], "graphs": [[1, 1024]], "switches_sha": "a", "free_after": 1 << 29}]
    assert capacity.check_built(built) and not capacity.check_built([dict(built[0]), dict(built[0])])


def test_warm_buckets_and_the_ones_captured_while_serving():
    # the fork's warm-up stops at 8192: the pool's top bucket (and every one past 8192) is captured by the first round
    # that reaches it, after CAP and READY looked at memory
    assert capacity.warm_buckets(8206, 1024, {}) is None
    assert capacity.lazy_buckets(8206, 1024) == [8206]
    assert capacity.lazy_buckets(5000, 1024) == [5000]
    cap = 262144 + 6 + 8                                                     # production: context + drafts + 1 + 8
    assert capacity.lazy_buckets(cap, 1024) == [16384, 32768, 65536, 131072, 262144, cap]
    # TFPP_DS41_WARM_ALL_BUCKETS=1: every bucket at start, nothing left for later; digested (every stage alike)
    every = capacity.warm_buckets(cap, 1024, {"TFPP_DS41_WARM_ALL_BUCKETS": "1"})
    assert every == tuple(PL.buckets(cap, 1024)) and capacity.lazy_buckets(cap, 1024, every) == []
    assert capacity.warm_buckets(cap, 1024, {"TFPP_DS41_WARM_ALL_BUCKETS": "0"}) is None
    assert env.digested(capacity.WARM_ALL)


# -- compare -----------------------------------------------------------------------------------------------------------
def _trace(logit=5):
    return [{"t": "meta"}, {"t": "req", "n": 1, "prompt_len": 3, "prompt_sha": "p", "draft": True},
            {"t": "admit", "sid": 0, "slot": 0, "base": 0, "size": 2048, "prompt_sha": "p", "prompt_len": 3},
            {"t": "chunk", "sid": 0, "start": 0, "rows": 3, "logits": [1, 2]},
            {"t": "step", "round": 1, "b": 1024, "rows": [[0, 0, 3, 2]], "tokens": [7, 8],
             "logits": [[1, 1], [2, logit]], "wall_ms": 3.0},
            {"t": "keep", "round": 1, "k": 1, "kept": [[0, 1]]},
            {"t": "done", "n": 1, "prompt_sha": "p", "tokens": [7, 8, 9], "stats": {"rounds": 1}}]


def test_compare_modes():
    quiet = []
    assert compare.compare(_trace(), _trace(), out=quiet.append) == 0
    assert compare.compare(_trace(), _trace(6), out=quiet.append) == 1
    assert any("DIFFERS at record 3 (step" in q for q in quiet)
    assert compare.compare(_trace(), _trace(6), mode="tokens", out=quiet.append) == 0
    assert compare.compare(_trace(), _trace(6), mode="position", out=quiet.append) == 1
    assert compare.compare(_trace(), _trace(), mode="position", out=quiet.append) == 0
    b = _trace()
    b[-1] = dict(b[-1], tokens=[7, 8, 10])
    assert compare.compare(_trace(), b, mode="tokens", out=quiet.append) == 1


def test_compare_refuses_runs_made_under_different_settings(tmp_path, capsys):
    def with_meta(t, **run):
        return [{"t": "meta", "role": "x", "run": {"settings": {"env": "e1", "graphs_env": True, "depth": ["conf", 1]},
                                                  "versions": {"torch": "2.13.0"}, "graphs": True, **run}}] + t[1:]

    a = with_meta(_trace())
    assert compare.meta_differences(a, with_meta(_trace())) == []
    assert compare.meta_differences(a, _trace()) is None                     # an old trace: no meta
    b = with_meta(_trace(), graphs=False)
    b[0]["run"]["settings"]["env"] = "e2"
    diffs = compare.meta_differences(a, b)
    assert [d.split(":")[0] for d in diffs] == ["graphs", "settings.env"]
    assert compare.meta_differences(a, b, ["graphs", "settings.env"]) == []
    assert compare.meta_differences(a, b, ["settings"]) == ["graphs: True vs False"]      # a prefix covers what is below
    b[0]["run"]["settings"]["depth"] = ["fixed", 1]
    assert compare.meta_differences(a, b, ["graphs,settings.env"]) != []                 # one --allow item per key
    # the command line: 2 = not comparable, 0 once the keys are allowed, 0 with --ignore-meta
    import json as _json

    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    b[0]["run"]["settings"]["depth"] = ["conf", 1]
    pa.write_text("\n".join(_json.dumps(r) for r in a) + "\n")
    pb.write_text("\n".join(_json.dumps(r) for r in b) + "\n")
    assert compare.main([str(pa), str(pb)]) == 2
    assert "graphs: True vs False" in capsys.readouterr().out
    assert compare.main([str(pa), str(pb), "--allow", "graphs,settings.env"]) == 0
    assert compare.main([str(pa), str(pb), "--ignore-meta"]) == 0


def test_oracle_planting_is_positional():
    hits = [p for p in range(2000) if oracle.planted(p)]
    assert 300 < len(hits) < 700 and hits == [p for p in range(2000) if oracle.planted(p)]


# -- the command lines -------------------------------------------------------------------------------------------------
def test_cli():
    from tfpp.ds41 import ref, run, serve

    ap = serve.build_parser()
    a = ap.parse_args(["--role", "leader", "--model", "m", "--layers", "6:11", "--listen", "127.0.0.1:5501",
                       "--bench", "p.jsonl", "--oracle", "o.json", "--context", "8192"])
    serve.check_args(a)
    assert a.layers == (6, 11) and a.parallel == 2 and a.mtp_drafts == 5 and a.port == 18891
    with pytest.raises(SystemExit):
        serve.check_args(ap.parse_args(["--role", "leader", "--model", "m", "--layers", "6:11"]))   # no --listen
    with pytest.raises(SystemExit):
        serve.check_args(ap.parse_args(["--role", "follower", "--model", "m", "--layers", "0:6"]))   # no --down
    with pytest.raises(SystemExit):
        serve.check_args(ap.parse_args(["--role", "follower", "--model", "m", "--layers", "0:6", "--down", "h:1",
                                        "--bench", "x"]))
    with pytest.raises(SystemExit):
        ap.parse_args(["--role", "leader", "--model", "m", "--layers", "0:11", "--no-drafts", "--mtp-drafts", "3"])
    assert run.translate(["middle", "--layers", "3:6", "--listen", "h:1", "--down", "h:2"])[:2] == ["--role",
                                                                                                    "follower"]
    with pytest.raises(SystemExit):
        run.translate(["first", "--layers", "3:6"])
    r = ref.build_parser().parse_args(["--model", "m", "--engram", "e", "--bench", "p", "--no-graphs", "--experts",
                                       "16"])
    assert r.no_graphs and r.experts == 16 and r.parallel == 2


def test_bench_prompts_file():
    from tfpp.ds41 import bench

    items = bench.load_prompts(DS41 / "bench_prompts.jsonl")
    lens = sorted(i["rand_len"] for i in items if "rand_len" in i)
    for n in (1, 5, 16, 17, 127, 128, 129, 131, 300, 2047, 2048, 2049, 2113, 4200, 4500, 6000, 2050, 2064, 4098, 4112):
        assert n in lens, n
    # last chunks of 2..16 rows under the replay floor (F/model.py:37-41), after a long prompt left keys in the ring
    order = [i["rand_len"] for i in items if "rand_len" in i]
    assert all(order.index(n) > order.index(6000) for n in (2050, 2064, 4098, 4112))
    assert any(i.get("thinking") for i in items) and any("text" in i and not i.get("thinking") for i in items)
    assert any(i.get("stop_after") for i in items) and any(i.get("repeat") == 2 for i in items)
    assert any(i.get("temperature") for i in items)
    assert all("group" not in i for i in items)
    multi = bench.load_prompts(DS41 / "bench_multi.jsonl")
    assert len({i.get("group") for i in multi}) >= 1 and len(multi) >= 4


# -- every source compiles (no torch needed to compile) ----------------------------------------------------------------
def test_every_source_compiles(tmp_path):
    files = sorted(DS41.glob("*.py")) + sorted((DS41 / "tests").glob("*.py"))
    assert len(files) >= 25
    for i, p in enumerate(files):
        py_compile.compile(str(p), cfile=str(tmp_path / f"{i}.pyc"), doraise=True)


def test_scripts_are_lf():
    scripts = T / "scripts" / "ds41"
    for p in sorted(scripts.glob("*")):
        if p.suffix in (".sh", ".env", ".example", ".service", "") and p.is_file():
            assert b"\r\n" not in p.read_bytes(), p.name


def test_scripts_keep_each_stage_apart_and_read_the_guard_ports():
    scripts = T / "scripts" / "ds41"
    start, stop, shortened = ((scripts / n).read_text(encoding="utf-8") for n in ("start_stage.sh", "stop_stage.sh",
                                                                                 "run_shortened.sh"))
    # one state folder a stage, derived alike by the two scripts (<root>/<role>-<A>-<B>)
    root = "STATE_ROOT=${TFPP_DS41_STATE:-$HOME/.local/state/tfpp-ds41}"
    assert root in start and root in stop
    assert "STATE=$STATE_ROOT/$ROLE-$L0-$L1" in start
    assert "STATE=$STATE_ROOT/$ROLE-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}" in stop
    # the other GPU servers' ports come from node.env / the environment, never a hard-coded list
    for text in (start, shortened):
        assert "for p in ${GPU_GUARD_PORTS:-18891}; do" in text
    # T1-http: no inherited key, the readiness probe on the route a key never closes, the test's own env restored
    assert "unset TF_API_KEY TF_API_KEY_FILE" in shortened and '/health" > /dev/null' in shortened
    assert "export TF_DS_REPLAY=1" not in shortened and "export TF_DS_DEPTH_POLICY=conf" not in shortened
