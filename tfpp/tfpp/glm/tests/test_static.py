"""T-S (spec 13.1): what tfpp-glm can check on any CPU Python, without torch or a GPU.

  cd tfpp && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tfpp/glm/tests/test_static.py

The MIA tree is found at TFPP_GLM_VENDOR, ../vendor/tensorfold-mia-v111pre/tensorfold beside this checkout, or the
installed tensorfold; nothing is written into it (no bytecode).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
T = HERE.parents[2]                                     # the tfpp checkout (tfpp/glm/tests -> tfpp)
if str(T) not in sys.path:
    sys.path.insert(0, str(T))

from tfpp.glm import capacity, compare, env, load, pins, proto  # noqa: E402


def vendor_root() -> Path | None:
    cands = [os.environ.get("TFPP_GLM_VENDOR", ""), str(T.parent / "vendor" / "tensorfold-mia-v111pre" / "tensorfold")]
    for c in cands:
        if c and (Path(c) / "__init__.py").is_file():
            return Path(c)
    try:
        return pins.installed_root()
    except RuntimeError:
        return None


ROOT = vendor_root()
needs_tree = pytest.mark.skipif(ROOT is None, reason="no MIA tensorfold tree found")
CONFIG = json.loads((HERE / "data" / "glm53_config.json").read_text(encoding="utf-8"))
TEXT = dict(CONFIG["text_config"])
TEXT["_quantization"] = CONFIG["quantization_config"]
KINDS = ["kda" if k == "linear_attention" else "dsa" for k in TEXT["layer_types"]]
TAPS = (5, 14, 24, 33, 42)


def importable_tensorfold() -> bool:
    if ROOT is None:
        return False
    parent = str(ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        import tensorfold.cuda.geometry  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# -- the package tree and the seams --------------------------------------------------------------------------------
@needs_tree
def test_tree_digest_is_the_mia_package():
    assert pins.tree_sha(ROOT) == pins.EXPECT_PKG_SHA


@needs_tree
def test_every_seam_unchanged():
    assert pins.diff(ROOT) == []
    assert all(v != "missing" for v in pins.pin_hashes(ROOT).values())


@needs_tree
def test_loader_anchors_once_and_compile():
    src = (ROOT / "families" / "glm5_next" / "cuda" / "weights.py").read_text(encoding="utf-8")
    for old, _ in load.ANCHORS:
        assert src.count(old) == 1, old
    patched = load.patched_source(src)
    compile(patched, "weights_tfpp", "exec")
    assert "list(_TFPP_LAYERS)" in patched and "_TFPP_NEXT" in patched and "_TFPP_EMBED" in patched
    assert patched.count("_TFPP_HEAD") == 2


def test_loader_anchor_missing_refuses():
    with pytest.raises(load.LoaderChanged):
        load.patched_source("def load():\n    pass\n")


def test_mine_rules():
    kw = dict(is_first=False, is_last=True, mtp=False, drafter=True, layers=45)
    assert load.mine("model.language_model.layers.30.mlp.gate.weight", 26, 45, **kw)
    assert not load.mine("model.language_model.layers.3.mlp.gate.weight", 26, 45, **kw)
    assert load.mine("model.language_model.embed_tokens.weight", 26, 45, **kw)          # DFlash2 reads it
    assert load.mine("lm_head.weight", 26, 45, **kw)
    assert not load.mine("lm_head.weight", 0, 7, is_first=True, is_last=False, mtp=False)
    assert load.mine("model.language_model.layers.45.eh_proj.weight", 26, 45, **dict(kw, mtp=True))
    assert not load.mine("model.visual.blocks.0.attn.qkv.weight", 26, 45, **kw)
    with pytest.raises(ValueError):
        load.mine("something.else", 0, 7, is_first=True, is_last=False, mtp=False)


def test_needed_shards(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    wm = {"model.language_model.embed_tokens.weight": "a.safetensors",
          "model.language_model.layers.0.mlp.gate_proj.weight": "a.safetensors",
          "model.language_model.layers.7.mlp.gate.weight": "b.safetensors",
          "model.language_model.layers.44.mlp.gate.weight": "c.safetensors",
          "model.language_model.norm.weight": "c.safetensors", "lm_head.weight": "d.safetensors",
          "model.visual.patch.weight": "v.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (tmp_path / "a.safetensors").write_bytes(b"")
    files, missing, unknown = load.needed_shards(tmp_path, 0, 7, is_first=True, is_last=False, mtp=False,
                                                 drafter=False)
    assert files == ["a.safetensors"] and missing == [] and unknown == []
    files, missing, _ = load.needed_shards(tmp_path, 26, 45, is_first=False, is_last=True, mtp=False, drafter=True,
                                           vision=True)
    assert files == ["a.safetensors", "c.safetensors", "d.safetensors", "v.safetensors"]
    assert missing == ["c.safetensors", "d.safetensors", "v.safetensors"]


# -- the protocol --------------------------------------------------------------------------------------------------
def test_frame_round_trip():
    ids = (1234).to_bytes(4, "little") * 3
    hdr = proto.make_header(proto.STEP, 7, {"req": 2, "pos": 100, "R": 3}, [("ids", "i32", [3])])
    data = proto.frame_bytes(hdr, [ids])
    got, parts, used = proto.split_frame(data + b"next")
    assert used == len(data)
    assert got["op"] == "STEP" and got["seq"] == 7 and got["pos"] == 100 and got["parts"] == [["ids", "i32", [3]]]
    assert parts["ids"] == ids


def test_header_checks():
    with pytest.raises(proto.ProtocolError):
        proto.decode_header(b"not json")
    with pytest.raises(proto.ProtocolError):
        proto.decode_header(json.dumps({"op": "STEP"}).encode())                     # no seq
    with pytest.raises(proto.ProtocolError):
        proto.decode_header(json.dumps({"op": "ACT", "seq": 0, "parts": [["x", "f16", [1]]]}).encode())
    with pytest.raises(proto.ProtocolError):
        proto.make_header("STEP", 0, {"seq": 1}, [])


def test_staging_sizes():
    assert proto.link_staging(2048, 1) == 80 << 20                                   # S0 -> S1 (x + tap 5)
    assert proto.link_staging(2048, 3) == 112 << 20                                  # S1 -> leader
    assert capacity.staging_bytes(2048, 1, 3, first=False, last=False) == 192 << 20
    assert capacity.staging_bytes(2048, 0, 1, first=True, last=False) == 80 << 20


def test_tap_routes_c_a_b():
    ranges = [(0, 7), (7, 26), (26, 45)]
    routes = proto.tap_routes(TAPS, ranges)
    assert routes[0] == ((), (0,))
    assert routes[1] == ((0,), (0, 1, 2))
    assert routes[2] == ((0, 1, 2), ())


def test_tap_routes_truncated_cross_every_cut():
    for taps, ranges in (((1, 2, 4, 5, 7), [(0, 4), (4, 8)]), ((1, 4, 6, 9, 10), [(0, 4), (4, 8), (8, 12)]),
                         ((1, 4, 6, 9, 10), [(0, 6), (6, 9), (9, 12)])):
        routes = proto.tap_routes(taps, ranges)
        for (cin, _), (a, b) in zip(routes[1:], ranges[1:]):
            assert cin, f"no tap crosses the cut at {a}"
        assert routes[-1][0] == tuple(s for s, t in enumerate(taps) if t < ranges[-1][0])


def test_check_chain():
    proto.check_chain([(0, 7), (7, 26), (26, 45)], 45, KINDS, TAPS)
    proto.check_chain([(0, 4), (4, 8)], 8, KINDS, (1, 2, 4, 5, 7))
    with pytest.raises(ValueError, match="contiguous"):
        proto.check_chain([(0, 7), (8, 45)], 45, KINDS, TAPS)
    with pytest.raises(ValueError, match="no DSA"):
        proto.check_chain([(0, 3), (3, 45)], 45, KINDS, TAPS)
    with pytest.raises(ValueError, match="ends at"):
        proto.check_chain([(0, 7), (7, 26)], 45, KINDS, TAPS)
    with pytest.raises(ValueError, match="outside"):
        proto.check_chain([(0, 4), (4, 8)], 8, KINDS, (1, 2, 4, 5, 9))


def test_compare_stages():
    a = proto.StageInfo((0, 7), "follower", settings={"dense": "q4"}, versions={"pkg_sha": "x"}, eos=[1, 2])
    b = proto.StageInfo((7, 45), "leader", settings={"dense": "q4"}, versions={"pkg_sha": "x"}, eos=[1, 2])
    assert proto.compare_stages([a, b]) == []
    c = proto.StageInfo.from_json(dict(a.to_json(), settings={"dense": "fp8"}))
    assert c.layers == (0, 7)
    assert any("dense" in p for p in proto.compare_stages([c, b]))


def test_prompt_view():
    v = proto.PromptView(2048, list(range(100)))
    assert len(v) == 2148
    assert v[2048:2051] == [0, 1, 2]
    assert v[2140:2200] == list(range(92, 100))
    with pytest.raises(IndexError):
        v[0:4]
    assert proto.chunk_count(0, 5000, 2048) == 3 and proto.chunk_count(64, 64, 2048) == 0


# -- admission -----------------------------------------------------------------------------------------------------
def test_kinds():
    assert capacity.layer_kinds(TEXT, 0, 7) == (6, 1)
    assert capacity.layer_kinds(TEXT, 7, 26) == (14, 5)
    assert capacity.layer_kinds(TEXT, 26, 45) == (14, 5)


@pytest.mark.skipif(not importable_tensorfold(), reason="tensorfold's geometry not importable")
def test_agree_on_the_package_geometry():
    flags = capacity.GeoFlags()
    ranges = [(0, 7), (7, 26), (26, 45)]
    geos = [capacity.stage_geometry(TEXT, a, b, last=i == 2, mtp=False, parallel=1, prefill_rows=2048, flags=flags)
            for i, (a, b) in enumerate(ranges)]
    gib = 2 ** 30
    budgets = [int(14 * gib), int(18 * gib), int(14 * gib)]
    got = capacity.agree(native=int(TEXT["max_position_embeddings"]), requested=262144, explicit=True,
                         budgets=budgets, geos=geos, kinds=[capacity.layer_kinds(TEXT, a, b) for a, b in ranges],
                         cache_bytes_wanted=int(6.2 * gib), dense_limit=2051, max_rows=16, dense_capacity=2560)
    assert got["window"] == 262144 and got["capacity"] == 262144 + 16 and got["long_context"]
    assert got["scale"][0] == pytest.approx(6 / 14, abs=1e-4) and got["scale"][1] == 1.0 and got["scale"][2] == 1.0
    room = min(s / c for s, c in zip(got["spare"], got["scale"]))
    assert got["cache_bytes"] == max(0, min(int(6.2 * gib), int(room)))
    # the stage without the head holds less than the leader with it at the same layer counts
    assert geos[1].needed(262144) < geos[2].needed(262144)
    with pytest.raises(ValueError):                                         # an explicit window that cannot fit
        capacity.agree(native=1048576, requested=1048576, explicit=True, budgets=[gib, gib, gib], geos=geos,
                       kinds=[(6, 1), (14, 5), (14, 5)], cache_bytes_wanted=0, dense_limit=2051, max_rows=16,
                       dense_capacity=2560)


def test_agree_with_fake_plans():
    class G:
        reserve, minimum_slots = 16, 2560

        def __init__(self, per):
            self.per = per

        def bytes_at(self, n):
            return 1000 + self.per * n

        def needed(self, w):
            return self.bytes_at(max(self.minimum_slots, w + self.reserve))

    class P:
        def __init__(self, fitting):
            self.fitting, self.largest, self.settings = fitting, fitting, [1048576, 8192, 1]

    def make_plan(native, requested, explicit, budget, weights, geo):
        return P(min(requested, (budget - 1000) // geo.per - 16))

    def choose(plan, rows):
        return min(min(r[3] for r in rows), plan.settings[1])

    geos = [G(10), G(30), G(30)]
    got = capacity.agree(native=1048576, requested=8192, explicit=True, budgets=[10 ** 6, 10 ** 6, 10 ** 6],
                         geos=geos, kinds=[(6, 1), (14, 5), (14, 5)], cache_bytes_wanted=10 ** 9, dense_limit=2051,
                         max_rows=16, dense_capacity=2560, make_plan=make_plan, choose=choose)
    assert got["window"] == 8192
    assert got["cache_bytes"] == int(min((10 ** 6 - geos[0].needed(8192)) / (6 / 14),
                                         10 ** 6 - geos[2].needed(8192)))


def test_kept_room_after_the_build():
    gib = 2 ** 30
    caps = [{"reserve": int(2.5 * gib), "declared": False}, {"reserve": 3 * gib, "declared": False},
            {"reserve": 3 * gib, "declared": False}]
    built = [{"layers": [0, 7], "free_after": 20 * gib}, {"layers": [7, 26], "free_after": 21 * gib},
             {"layers": [26, 45], "free_after": int(16.5 * gib)}]
    scale = [6 / 14, 1.0, 1.0]
    assert capacity.kept_room(caps, built, scale, int(6.2 * gib)) == (int(6.2 * gib), [])     # room to spare
    # the leader's build took more than the geometry said: kept prompts shrink to keep half its reserve free
    built[2]["free_after"] = 4 * gib
    kept, refusals = capacity.kept_room(caps, built, scale, int(6.2 * gib))
    assert refusals == [] and kept == int(4 * gib - 1.5 * gib)
    # stage 0 has less than half its reserve left: refused, by name
    built[0]["free_after"] = gib
    _, refusals = capacity.kept_room(caps, built, scale, int(6.2 * gib))
    assert len(refusals) == 1 and "[0, 7]" in refusals[0]
    # --budget-gib stages (several on one GPU) are not checked
    caps = [dict(c, declared=True) for c in caps]
    assert capacity.kept_room(caps, built, scale, int(6.2 * gib)) == (int(6.2 * gib), [])


# -- environment ---------------------------------------------------------------------------------------------------
def test_env_sets_and_refuses():
    environ = {"TF_GLM_DENSE": "q4"}
    set_now = env.check_env("t", environ, modules={}, triton_version="3.7.1")
    assert environ["TF_GLM_HC_SPLIT"] == "0" and environ["TF_GLM_FILL_BUDGET_MS"] == "0"
    assert set(set_now) == set(env.ENFORCED)
    for bad in ({"TF_GLM_FILL_BUDGET_MS": "200"}, {"TF_GLM_HC_SPLIT": "1"}, {"TF_ROCE_MAX_KB": "512"},
                {"NCCL_DEBUG": "INFO"}, {"TF_GLM_COMM": "roce"}, {"TF_GLM_DRAFT_DUMP": "/tmp/x"}):
        with pytest.raises(env.EnvError):
            env.check_env("t", dict(bad), modules={}, triton_version="3.7.1")
    with pytest.raises(env.EnvError, match="Triton"):
        env.check_env("t", {}, modules={}, triton_version="3.6.0")
    with pytest.raises(env.EnvError, match="import order"):
        env.check_env("t", {}, modules={"tensorfold.families.glm5_next.cuda.engine": object()}, triton_version="3.7.1")


def test_env_digest_ignores_each_nodes_own():
    a = {"TF_GLM_DENSE": "q4", "TF_GLM_CACHE_GIB": "6.2", "TFPP_WARM_MS": "1", "TFPP_GLM_CONF": "/x", "PATH": "/a"}
    b = {"TF_GLM_DENSE": "q4", "TFPP_FAULT": "2:3", "PATH": "/b"}
    assert env.env_digest(a) == env.env_digest(b)
    assert env.env_digest(a) != env.env_digest(dict(a, TF_GLM_DENSE="fp8"))
    # numerics knobs outside the packages' prefixes count; each node's Triton folders and tool paths do not
    for knob in ("TRITON_DEFAULT_FP_FUSION", "TRITON_F32_DEFAULT", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
                 "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        assert env.env_digest(a) != env.env_digest(dict(a, **{knob: "0"})), knob
    assert env.env_digest(a) == env.env_digest(dict(a, TRITON_CACHE_DIR="/h/x", TRITON_PTXAS_PATH="/p",
                                                    TRITON_HOME="/h"))


def test_model_revision(tmp_path):
    assert env.model_revision(tmp_path) == "unknown"
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    rev = "57edefd2" + "0" * 32
    (meta / "config.json.metadata").write_text(f"{rev}\n{'1' * 40}\n1761200000.0\n")
    assert env.model_revision(tmp_path) == rev
    snap = tmp_path / "hub" / "snapshots" / ("76c0b517" + "f" * 32)
    snap.mkdir(parents=True)
    assert env.model_revision(snap) == "76c0b517" + "f" * 32


def test_verify_shards(tmp_path):
    import hashlib

    (tmp_path / "a.safetensors").write_bytes(b"abliterated")
    (tmp_path / "b.safetensors").write_bytes(b"original")
    (tmp_path / "c.safetensors").write_bytes(b"no metadata")
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    for name, body in (("a", b"abliterated"), ("b", b"something else")):
        (meta / f"{name}.safetensors.metadata").write_text(
            f"{'0' * 40}\n{hashlib.sha256(body).hexdigest()}\n0\n")
    cache = tmp_path / "state" / "sha.json"
    files = ["a.safetensors", "b.safetensors", "c.safetensors"]
    assert load.verify_shards(tmp_path, files, cache) == (["b.safetensors"], ["c.safetensors"])
    assert cache.is_file() and len(json.loads(cache.read_text())) == 2
    assert load.verify_shards(tmp_path, files, cache) == (["b.safetensors"], ["c.safetensors"])   # from the cache


# -- the CLI and the tools -----------------------------------------------------------------------------------------
@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="posix_fadvise is Linux only")
def test_drop_cached(tmp_path):
    for i in (1, 2):
        (tmp_path / f"model-0000{i}-of-00002.safetensors").write_bytes(b"\0" * 8192)
    (tmp_path / "config.json").write_text("{}")
    assert load.drop_cached(tmp_path) == 2
    assert load.drop_cached(tmp_path / "missing") == 0


def test_cli():
    from tfpp.glm import run, serve

    p = serve.build_parser()
    a = p.parse_args(["--role", "follower", "--model", "/m", "--layers", "7:26", "--down", "h:1", "--listen", "0:2"])
    serve.check_args(a)
    assert a.layers == (7, 26)
    a = p.parse_args(["--role", "leader", "--model", "/m", "--layers", "0:8", "--truncate", "8", "--tap-layers",
                      "1,2,4,5,7", "--drafter", "/d", "--pure"])
    serve.check_args(a)
    assert a.tap_layers == (1, 2, 4, 5, 7)
    for bad in (["--role", "follower", "--model", "/m", "--layers", "7:26", "--down", "h:1"],      # no --listen
                ["--role", "leader", "--model", "/m", "--layers", "26:45"],                         # no --listen
                ["--role", "leader", "--model", "/m", "--layers", "0:8", "--truncate", "12"],
                ["--role", "leader", "--model", "/m", "--layers", "0:45", "--parallel", "4"]):
        with pytest.raises(SystemExit):
            serve.check_args(p.parse_args(bad))
    assert run.translate(["last", "--layers", "26:45"])[:2] == ["--role", "leader"]
    assert run.translate(["first", "--layers", "0:7", "--down", "h:1"])[:2] == ["--role", "follower"]
    with pytest.raises(SystemExit):
        run.translate(["middle", "--layers", "7:26"])


def test_compare_tool(tmp_path):
    rows = [{"t": "meta"}, {"t": "req", "n": 1, "prompt_len": 3}, {"t": "chunk", "start": 0, "R": 3, "head": [1, 2]},
            {"t": "step", "pos": 3, "R": 2, "window": [5, 6], "logits": [[1, 1], [2, 2]], "wall_ms": 3.0},
            {"t": "keep", "pos": 3, "R": 2, "keep": 1}, {"t": "done", "n": 1, "tokens": [5, 9]}]
    other = [dict(r) for r in rows]
    other[3] = dict(other[3], wall_ms=99.0)                                        # timings are not compared
    assert compare.compare(rows, other, show=0) == 0
    other[3] = dict(other[3], logits=[[1, 1], [2, 3]])
    assert compare.compare(rows, other, show=0) == 1
    assert "logits[1]" in compare.first_difference(rows[3], other[3])
    # --tokens-only: a resumed prefill against a fresh one (other chunks and windows), the same reply
    resumed = [rows[0], rows[1], {"t": "chunk", "start": 2, "R": 1, "head": [7, 7]}, rows[3], rows[4], rows[5]]
    assert compare.compare(rows, resumed, show=0) == 1
    assert compare.compare(rows, resumed, tokens_only=True, show=0) == 0
    assert compare.compare(rows, rows[:-1] + [dict(rows[-1], tokens=[5, 8])], tokens_only=True, show=0) == 1


def test_bench_prompts_parse():
    from tfpp.glm import bench

    items = bench.load_prompts(HERE.parent / "bench_prompts.jsonl")
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)) and "copy" in ids and "one-token" in ids
    assert any(i.get("draft") is False for i in items) and any(i.get("stop_after") for i in items)


def test_sources_compile():
    for p in sorted(HERE.parent.glob("*.py")):
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
