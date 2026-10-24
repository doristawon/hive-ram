"""Which fork tfpp-ds41 was written against (no torch: runs on any CPU Python).

``EXPECT_PKG_SHA``: tfpp.glm.pins.tree_sha of the fork's ``src/tensorfold`` at af26d42b1 as git stores it (LF bytes:
``git archive``, or a checkout with core.autocrlf=false; 474 files). A Windows work tree checked out with
core.autocrlf=true has CRLF bytes and the digest ccc7c2a8...: refused (copy the tree with ``git archive``, spec 1.4).

``PINS``: the sha256 of each seam's source (``ast.get_source_segment`` of the file read with universal newlines), so a
new fork lists exactly the seams to re-read: ``python -m tfpp.ds41.pins --diff <new src/tensorfold>``. A stage checks
them after the tree digest (``check_seams``), once the tree is known.

  python -m tfpp.ds41.pins --check [TREE]   tree digest of TREE (default: the importable tensorfold) == EXPECT_PKG_SHA
  python -m tfpp.ds41.pins --print TREE     the digests and the pins of TREE (to refresh this file after a review)
  python -m tfpp.ds41.pins --diff TREE      the seams whose source differs in TREE
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

from tfpp.glm.pins import SUFFIXES, _find, installed_root, tree_sha  # noqa: F401

EXPECT_COMMIT = "af26d42b1f618e9e653903a6c043448b52729c64"
# tree_sha of src/tensorfold at EXPECT_COMMIT, LF bytes (474 files)
EXPECT_PKG_SHA = "c64d3e0ad718683a88feb9ef87447c47de3514124597d636eb0a859bcde995b8"
# the same over families/deepseek_v41/ alone (diagnostics: which part of a tree differs)
FAMILY_SHA = "2db03159b66a764b86b700fb3b9e45b0def060431861503fa2e7a2de38387542"
CRLF_PKG_SHA = "ccc7c2a8"      # the prefix the CRLF Windows work tree gives (refused)

F = "families/deepseek_v41/cuda/"
FO = "families/deepseek_v41/"

# (file relative to the tensorfold package, qualified name): what tfpp-ds41 patches, overrides, copies or relies on
SEAMS = (
    # model.py: StageModel's overrides and anchors (M1, M2, F1c, F1d), StageEngram's base
    (F + "model.py", "Model.__init__"),
    (F + "model.py", "Model.new_cache"),
    (F + "model.py", "Model.new_pool"),
    (F + "model.py", "Model.pool_view"),
    (F + "model.py", "Model.forward"),
    (F + "model.py", "Model._forward_k"),
    (F + "model.py", "Model._hc_pf2"),
    (F + "model.py", "Model.attention_k"),
    (F + "model.py", "Model.kv_source_update"),
    (F + "model.py", "Model._compress"),
    (F + "model.py", "Engram.__init__"),
    (F + "model.py", "Engram._aio_open"),
    (F + "model.py", "Engram.hashes"),
    (F + "model.py", "Engram.prefetch"),
    (F + "model.py", "Engram.touch"),
    (F + "model.py", "Engram.rows"),
    (F + "model.py", "_candidates"),
    (F + "model.py", "apply_candidates"),
    # rounds.py: StageRoundDecoder / StageRoundRunner (R1-R3, F1a, F1b)
    (F + "rounds.py", "RoundDecoder.__init__"),
    (F + "rounds.py", "RoundDecoder._ix"),
    (F + "rounds.py", "RoundDecoder._kv_idx"),
    (F + "rounds.py", "RoundDecoder._stretch"),
    (F + "rounds.py", "RoundDecoder._l2_next"),
    (F + "rounds.py", "RoundDecoder._layers_fused"),
    (F + "rounds.py", "RoundDecoder.capture"),
    (F + "rounds.py", "RoundDecoder.set"),
    (F + "rounds.py", "RoundDecoder.run"),
    (F + "rounds.py", "RoundRunner.__init__"),
    (F + "rounds.py", "RoundRunner.forward"),
    (F + "rounds.py", "RoundRunner._forward"),
    (F + "rounds.py", "RoundRunner.engram_touch"),
    (F + "rounds.py", "_candidates_fast"),
    (F + "rounds.py", "_candidate_blocks"),
    # multi.py: StageMulti (MU1), the mirrored control flow, PipeOps' op shapes
    (F + "multi.py", "MultiDecoder.__init__"),
    (F + "multi.py", "MultiDecoder.warm"),
    (F + "multi.py", "MultiDecoder._drafts"),
    (F + "multi.py", "MultiDecoder._send"),
    (F + "multi.py", "MultiDecoder._step"),
    (F + "multi.py", "MultiDecoder._broken"),
    (F + "multi.py", "MultiDecoder._alive"),
    (F + "multi.py", "MultiDecoder._shape"),
    (F + "multi.py", "MultiDecoder.admit"),
    (F + "multi.py", "MultiDecoder._admit"),
    (F + "multi.py", "MultiDecoder._need"),
    (F + "multi.py", "MultiDecoder._place"),
    (F + "multi.py", "MultiDecoder._fill"),
    (F + "multi.py", "MultiDecoder.round"),
    (F + "multi.py", "MultiDecoder._round"),
    (F + "multi.py", "MultiDecoder.finish"),
    (F + "multi.py", "MultiDecoder._finish"),
    (F + "multi.py", "MultiDecoder._release"),
    (F + "multi.py", "MultiDecoder.drop"),
    (F + "multi.py", "MultiDecoder._drop"),
    (F + "multi.py", "MultiDecoder._snapshot"),
    (F + "multi.py", "MultiDecoder._restore"),
    (F + "multi.py", "MultiDecoder._copy_rows"),
    # engine.py: PipeDsEngine / FollowerEngine follow __init__ step by step; the rest is inherited
    (F + "engine.py", "DsEngine.__init__"),
    (F + "engine.py", "DsEngine.warm"),
    (F + "engine.py", "DsEngine.prefill"),
    (F + "engine.py", "DsEngine.prefill_steps"),
    (F + "engine.py", "DsEngine.generate"),
    (F + "engine.py", "DsEngine._sample"),
    (F + "engine.py", "DsEngine._share_rows"),
    # weights.py: load.py's AliasShards and load_stage / load_full (the reference's weights.load)
    (F + "weights.py", "Shards.__init__"),
    (F + "weights.py", "load_block"),
    (F + "weights.py", "load"),
    (F + "weights.py", "_load"),
    (F + "weights.py", "linear"),
    (F + "weights.py", "_kept"),
    (F + "weights.py", "attach_vl_bias"),
    (F + "graph.py", "bucket_for"),
    (F + "graph.py", "GraphRunner.__init__"),
    # kernels.py: what T-K' compares (spec 3.6) and the round-rows bound
    (F + "kernels.py", "hc_post"),
    (F + "kernels.py", "hc_pre2"),
    (F + "kernels.py", "hc_pre_pf"),
    (F + "kernels.py", "round_rows"),
    # ops.py / config.py: host ids, Engram hashing, the token map, the config (plan.ModelView reads the same keys)
    (FO + "ops.py", "HostIds.set"),
    (FO + "ops.py", "EngramHasher.__init__"),
    (FO + "ops.py", "engram_primes"),
    (FO + "ops.py", "engram_multipliers"),
    (FO + "ops.py", "compressed_token_map"),
    (FO + "config.py", "Cfg.from_dict"),
    (FO + "__init__.py", "fp32_gemms"),
    # the leader's drafting and serving, used as they are
    (F + "dspark.py", "Drafter.__init__"),
    (F + "dspark.py", "Drafter.absorb"),
    (F + "dspark.py", "DraftPool"),
    (F + "dspark.py", "BatchDraftGraph"),
    (F + "app.py", "DsApp"),
    ("cuda/scheduler.py", "Scheduler"),
    ("cuda/streams.py", "Stream"),
)

PINS: dict[str, str] = {
    "families/deepseek_v41/cuda/model.py::Model.__init__":
        "561aa7eddb3167b8ca04b8038d92338d932676a7829c4af8f8c46d1eaebeff8d",
    "families/deepseek_v41/cuda/model.py::Model.new_cache":
        "bce6a796e754db7151f6a19da15903f700ec924d2333488580e2336d15e9e3f3",
    "families/deepseek_v41/cuda/model.py::Model.new_pool":
        "7b9a2aa0d4ba51d11f80140fe5231729b5bce1bb5698cf33d0ddce6bbe1dba85",
    "families/deepseek_v41/cuda/model.py::Model.pool_view":
        "39311bd45479c4d02722ee94b35c40de30441b1be95b8344ac77938a51a1bb44",
    "families/deepseek_v41/cuda/model.py::Model.forward":
        "fe1d4f834c95d2fe2bb238f849e0ba1cef538586d2270b6c5660293f9a7c0156",
    "families/deepseek_v41/cuda/model.py::Model._forward_k":
        "558bea5ad1a3feb48222ebecd04395fb467e1767e75f3293d2069a0362349e81",
    "families/deepseek_v41/cuda/model.py::Model._hc_pf2":
        "070196bf865e5e73a834f52073f567179255443c11f6e5c6795f6685292763a9",
    "families/deepseek_v41/cuda/model.py::Model.attention_k":
        "05ebbd559579f4bcc9784cbdf0b1f2056aebb5baf186b2607157e2005abddb9a",
    "families/deepseek_v41/cuda/model.py::Model.kv_source_update":
        "64f175b946f0485e529bd260f2904ab967249023c7dfc328882eff94968f4e59",
    "families/deepseek_v41/cuda/model.py::Model._compress":
        "8f080e7badd9cebc6bf6b0e3f5045fabb3180c4b7a7103edef7b03adba334059",
    "families/deepseek_v41/cuda/model.py::Engram.__init__":
        "aee9e6641030af252cca1278d939a014ccbfdda64ebde8e24d111971bdfe5db3",
    "families/deepseek_v41/cuda/model.py::Engram._aio_open":
        "18218d8005d0e4c01c053d1a77fb1a952f85f333e39af9964bc9762d8b0a70dd",
    "families/deepseek_v41/cuda/model.py::Engram.hashes":
        "7bfc8c13fadc06a621aeefc6868f295c699492f7751dd11936ad5c3de49e7220",
    "families/deepseek_v41/cuda/model.py::Engram.prefetch":
        "fa843dd9e49506faa138cd36eb0bc2b405c8bcbc92f8c32518476fbf750b2fbb",
    "families/deepseek_v41/cuda/model.py::Engram.touch":
        "16f99e6e23e7c1efd05e6dad7603f08fe1596f2496fa42b224d8f38ef8b2ff01",
    "families/deepseek_v41/cuda/model.py::Engram.rows":
        "98e800ccb5d5b267feaf19249f87e2924149ee2f85f71856afdf13db07011fb3",
    "families/deepseek_v41/cuda/model.py::_candidates":
        "2509bc5c5604f8884a5d3ee07fc8e12fc26a0d26744ce79875d121d6c3a116b0",
    "families/deepseek_v41/cuda/model.py::apply_candidates":
        "7da19770f36b4f38f368cfed5176c7162086b40d41cc5afbe972b661784a729e",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder.__init__":
        "e858e457d74b4ea22e27ce81ae042e971b8cd707b5fc0e883b2bd789c187b504",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder._ix":
        "73576ed2f28fa4dafe95c5cac8621be043c18e85ae020a3630094ce95193050e",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder._kv_idx":
        "4a4ce4abd79a5934318cf2667194da8be0b4dba700136047d42b923b2dd5e1a9",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder._stretch":
        "95a84f4549d8d1094d96e4fc7a657fcec2d44cf9e4c62108daa2503106c453e0",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder._l2_next":
        "c679f1be33f2ed4f7a0af2ec6d2507f666b044286f84af15ee13f363a9c51dd4",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder._layers_fused":
        "5871bd71f4d3e5087eb76345ec56f1f956602e0a382e1392f500d81f42649984",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder.capture":
        "9443dcb889464d868b18b667bf4fd327377d4b05f74c9da2f99a9d6dd215352c",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder.set":
        "d551c3971033b290f26cc38a37e5f75cf4e817271428f3d978533c98706f3c24",
    "families/deepseek_v41/cuda/rounds.py::RoundDecoder.run":
        "5a1f6e4fcf19238a9537c6323e6072c9247cf5b569e1ae475f60bffcb3773291",
    "families/deepseek_v41/cuda/rounds.py::RoundRunner.__init__":
        "502aad402623325304dc046b01fe40aae5e986885fac226e98caeea63fd112e8",
    "families/deepseek_v41/cuda/rounds.py::RoundRunner.forward":
        "8eaf2dfd8cbe953d656d2ae6452671626af6c70b904a2fdab48afbccfb60a2b1",
    "families/deepseek_v41/cuda/rounds.py::RoundRunner._forward":
        "3579fda3ebd1d00722b324e7e5eb81464e764bc9f2eaf26d0c45258b540088c0",
    "families/deepseek_v41/cuda/rounds.py::RoundRunner.engram_touch":
        "7230eefa27bfde18c087b692c8cc70d2ad277f36ba1aa6ebd727b55e3473e9d0",
    "families/deepseek_v41/cuda/rounds.py::_candidates_fast":
        "0343f670471316603cf36fd80d60e13f97b7bead76adf3ac0270162694357a5c",
    "families/deepseek_v41/cuda/rounds.py::_candidate_blocks":
        "c34c0285b01400cfc52fb5d53afccdf0c9db88ac2a2d8187e3b61922553eeb1e",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder.__init__":
        "07cb794e7f65df4f05fc76da461c8cbfb8dca8d6ad751fab3ca4ca79d2ce8eec",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder.warm":
        "09e652e22a011db7ef78da1dbf4560f1ea69cda3685bdd11c72f84230ebb8fca",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._drafts":
        "d0dfa6fb31a50404bf75f6d2dd01ab8b3bb97042bf1e8027f7ee91781beeaa65",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._send":
        "46abf081ba982a3a3d3e5c54a493c161428834923798b19646a8dc002c9be799",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._step":
        "1dba39b8e44ae2736667da184413f74004259ba14f9a5c4072adc602161318fc",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._broken":
        "15d159c30b95a047648d3c337b89d42a2e461bde12df6a0d1de0bbe7fb502c49",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._alive":
        "b301a1f158c143561ebd95f6b96b1e8769d7680fcd65f0e9efbe71639b287889",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._shape":
        "4377f8f40d9206d375df339705d61c4154e54fecb093535e7744bcac8e5de9ed",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder.admit":
        "bb20da9a6712fd6589b73d6f6e9f80bcf4e8fdbbc889b3907952b9a33b5988ec",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._admit":
        "29439c7ce9dccbd374342263db36790b8ef3b0c925837606da7924216c9bfcd4",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._need":
        "1bb98715bf6ea4437b6a0524ebad2eec7e7204c0306c300291ac7932eb8e3f5d",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._place":
        "190066416ec4af80e2b8711be7ffa6c8c71c5183e2223f70c357f2d0d3bf730b",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._fill":
        "010603b8b41265f12e10ece5934c7af1c8bfa295895774e9e3c4b84a0489a6ec",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder.round":
        "af61dcf464651782ad3232fd68563a0699bb83262e581377c16cc7273ebdf8dd",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._round":
        "af3ef1b9b9c644d56c4c40c6880f140c6b2bca9cbf579b95cdd757f79af1be09",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder.finish":
        "7c4b0ad3f14178f897485af46e99e5bdbcc166da2df0053cd7af188d4923d73e",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._finish":
        "5d1f2434ff9f09e57939662b2e476659fbd136d88019f599b7fbd9dce55fa7fe",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._release":
        "5e75af2edc6edc909bdd6bd7246100f4647248ca239fbe97f1e12aa854052e75",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder.drop":
        "f5beb77732fdc29ec1786404d1121af0f7219db9fc82f811022cf0cd0cc58488",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._drop":
        "44e7c134849d445ffd3fd1f7e8507bdabbb118942b9e840504402c6377cd89ab",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._snapshot":
        "32314d5c217559ed211d56fae99b04d3c63140144ec1807210c5f6a15438257b",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._restore":
        "ddf2cf2a37916519b609d69a36abddc090479b906978414ae27f4a3406820a1e",
    "families/deepseek_v41/cuda/multi.py::MultiDecoder._copy_rows":
        "7c7ba83a9d178bb17c5402b87d7f94e0a386bf9a4b16f3ecf83b5e4ab30bea49",
    "families/deepseek_v41/cuda/engine.py::DsEngine.__init__":
        "0b89d0a850dbc0ad95356841134ae7b21592d18a35d7bc968154e0a589b9cae9",
    "families/deepseek_v41/cuda/engine.py::DsEngine.warm":
        "0e0c152ecdf0e393470e14d2650029544f258cb1b05977faa1c642e2d7e4a7b9",
    "families/deepseek_v41/cuda/engine.py::DsEngine.prefill":
        "b687953ca1368a75e3e8bd6c17193053f82ba6e99d76a40074bb6b193e96f412",
    "families/deepseek_v41/cuda/engine.py::DsEngine.prefill_steps":
        "f505affde0185ae4b8373e32a37871c2564fdecf1d0c7dcb076e8e409ae97bbc",
    "families/deepseek_v41/cuda/engine.py::DsEngine.generate":
        "0155b95aafae4ce4792d8dd0ad04400a7d68ea259d2964eeb87a103c70e30932",
    "families/deepseek_v41/cuda/engine.py::DsEngine._sample":
        "d7a0f9236524ed4ec6e0870631df4c3216fdeda13c97164d58511aafc6ba81db",
    "families/deepseek_v41/cuda/engine.py::DsEngine._share_rows":
        "ebc8543bd69977cd1473c2db2ae0b9637216e66cc1e1825762e6725d9d85bae7",
    "families/deepseek_v41/cuda/weights.py::Shards.__init__":
        "c02b277f48d561c1052c02a83e2243efe37bb7abef213e0d69a7dd290a9d4d77",
    "families/deepseek_v41/cuda/weights.py::load_block":
        "e71d1f5a0406cd1fdd348fd25bd365dd8da3fa21246261919de9120e1495e555",
    "families/deepseek_v41/cuda/weights.py::load":
        "6a98935e5c4422e79274326336ec07bae73bdc9f4071c2e5f4b1fc367cda434d",
    "families/deepseek_v41/cuda/weights.py::_load":
        "ee433ead49447b48266b45180c1eb95ce9a2d45fd3c71d29987f8b537c044e38",
    "families/deepseek_v41/cuda/weights.py::linear":
        "d6e43103d81bd5e0fbfafbb8d3401ff76bdfcb91ee64617c20ebde2cc7beaecb",
    "families/deepseek_v41/cuda/weights.py::_kept":
        "c1417a0b147b9746f10f5ce23b9ed0b492eefd198943fd3e1d02cb31f54cf906",
    "families/deepseek_v41/cuda/weights.py::attach_vl_bias":
        "dbac9e02ce57cde2ac14fd2a5dfbc9cbdc3ac5a26b6df1e21840ca243babae0f",
    "families/deepseek_v41/cuda/graph.py::bucket_for":
        "3be565fe9b32a7341321153c386865153c7dad4f096e381d71e1c7fa14aaae46",
    "families/deepseek_v41/cuda/graph.py::GraphRunner.__init__":
        "803be9e412341db2edb94b98e7e9cce2a43fef5f8ab34c622c5f16adb60cfb9c",
    "families/deepseek_v41/cuda/kernels.py::hc_post":
        "4ad7713d5bd5bbcdaacc71d0930f864c63b0d0fd0ac3a7d226dc96a0be1f1b13",
    "families/deepseek_v41/cuda/kernels.py::hc_pre2":
        "ce9636a1de9b74d38324a5ece4e94ded335c792ee439cb8e37af7818319e6a47",
    "families/deepseek_v41/cuda/kernels.py::hc_pre_pf":
        "d8dee55e6c3515a9bc796311ae84721c13f8b673875feda9c6fc1883e24fe260",
    "families/deepseek_v41/cuda/kernels.py::round_rows":
        "9c922a26a1e54dad16fa168b0428cafb264a0f7dc6b29bd45c5e750ca7ea8c9d",
    "families/deepseek_v41/ops.py::HostIds.set":
        "95cb5d701eb7d2c3c411914392d6da82def5116d568a3b82b7807e643be9ead4",
    "families/deepseek_v41/ops.py::EngramHasher.__init__":
        "c57def68cb40ea2e5da448ca0f427f0e83109211aa0a75d86bbc90d8f98f8304",
    "families/deepseek_v41/ops.py::engram_primes":
        "1c6f634073c5cfbf7ae7de473cd1d712f9b0f924cec13b6cde187022bfe0599d",
    "families/deepseek_v41/ops.py::engram_multipliers":
        "16b0138aa16aa84af5c8d30ab7aaf4ea455439c393cc39a1f761bbf4b0d7b226",
    "families/deepseek_v41/ops.py::compressed_token_map":
        "110a140c4e5964779b7155bfebf820a8f1f51c6b603d7ace32e53e626a60b984",
    "families/deepseek_v41/config.py::Cfg.from_dict":
        "c171f1db9ec03b02ae52847012281ea4fc8f24d461a3ad40185a19ba3eb9b50a",
    "families/deepseek_v41/__init__.py::fp32_gemms":
        "7a043ec3d27683abbfe3413e2c275d0fe9da246ee71425a13459efc98a36ad61",
    "families/deepseek_v41/cuda/dspark.py::Drafter.__init__":
        "e3ee9c95b022b98dce3bf127cefefc3670eeb80d50fdfd88e19655a3a1e56c58",
    "families/deepseek_v41/cuda/dspark.py::Drafter.absorb":
        "b5289f2d58240362efcfe0afb75a14552bdfc139c3f72bb9264777770b0acb49",
    "families/deepseek_v41/cuda/dspark.py::DraftPool":
        "4cf355638700588e223340e361f702d657c6e0b580ba200e77c98d4dd62d84e5",
    "families/deepseek_v41/cuda/dspark.py::BatchDraftGraph":
        "1496ee24300f1775620ab11192e785e29c403db0493c8fbf65d210e0992162ca",
    "families/deepseek_v41/cuda/app.py::DsApp":
        "2b4b8d0bf14d4c84c0764d85e60328ed56faa022dfa6534cdbb6157cfa0ca1a0",
    "cuda/scheduler.py::Scheduler":
        "61f576f905b9528d7e4e617efec2ba9cca4a9960f8bb59c987d323622f542d9c",
    "cuda/streams.py::Stream":
        "d9bfddb91f73f7dd17a1e6a2548dc002d77f19e6c7ddc0dfe880f671efd9f3ad",
}


class TreeMismatch(RuntimeError):
    pass


def tree_sha_lf(root: str | Path) -> str:
    """tree_sha with CRLF read as LF (a Windows work tree's files as git stores them; tests and diagnostics only)."""

    root = Path(root)
    items = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts:
            items.append((p.relative_to(root).as_posix(),
                          hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()))
    h = hashlib.sha256()
    for rel, digest in sorted(items):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def family_sha(root: str | Path, lf: bool = True) -> str:
    """The digest of families/deepseek_v41/ alone, paths relative to the tensorfold folder (diagnostics)."""

    root = Path(root)
    fam = root / "families" / "deepseek_v41"
    items = []
    for p in fam.rglob("*"):
        if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts:
            b = p.read_bytes()
            items.append((p.relative_to(root).as_posix(), hashlib.sha256(b.replace(b"\r\n", b"\n") if lf else b)
                          .hexdigest()))
    h = hashlib.sha256()
    for rel, digest in sorted(items):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def check_tree(root: str | Path | None = None, expect: str = EXPECT_PKG_SHA) -> str:
    """The tree digest of ``root`` (default: the importable tensorfold), refusing anything but ``expect``."""

    root = installed_root() if root is None else Path(root)
    got = tree_sha(root)
    if got != expect:
        hint = ""
        if tree_sha_lf(root) == expect:
            hint = (" -- it is the right fork with CRLF line ends (a Windows checkout, core.autocrlf=true): install "
                    "it with git -c core.autocrlf=false archive af26d42b1 src | tar -x (spec 1.4)")
        raise TreeMismatch(f"the tensorfold tree at {root} has digest {got[:16]}, not the fork tfpp-ds41 was written "
                           f"against (deepseek-v41-tp2 @ {EXPECT_COMMIT[:9]}, {expect[:16]}){hint}; or re-review the "
                           "seams: python -m tfpp.ds41.pins --diff TREE")
    return got


def pin_hashes(root: str | Path) -> dict[str, str]:
    """``file::qualname`` -> sha256 of its source segment (``missing`` when the tree has no such definition)."""

    root = Path(root)
    out: dict[str, str] = {}
    cache: dict[str, tuple[str, ast.Module]] = {}
    for rel, qual in SEAMS:
        key = f"{rel}::{qual}"
        path = root / rel
        if not path.is_file():
            out[key] = "missing"
            continue
        if rel not in cache:
            src = path.read_text(encoding="utf-8")            # universal newlines: LF and CRLF trees pin alike
            cache[rel] = (src, ast.parse(src))
        src, tree = cache[rel]
        node = _find(tree, qual)
        seg = None if node is None else ast.get_source_segment(src, node)
        out[key] = "missing" if seg is None else hashlib.sha256(seg.encode("utf-8")).hexdigest()
    return out


def diff(root: str | Path, pins: dict[str, str] | None = None) -> list[str]:
    pins = PINS if pins is None else pins
    now = pin_hashes(root)
    return [k for k in now if now[k] != pins.get(k)]


def check_seams(root: str | Path | None = None) -> None:
    root = installed_root() if root is None else Path(root)
    changed = diff(root)
    if changed:
        raise TreeMismatch(f"{len(changed)} seams differ from the ones tfpp-ds41 was written against: "
                           + ", ".join(changed[:6]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.ds41.pins", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", nargs="?", const="", metavar="TREE")
    g.add_argument("--print", dest="show", metavar="TREE")
    g.add_argument("--diff", metavar="TREE")
    a = ap.parse_args(argv)
    if a.check is not None:
        try:
            got = check_tree(a.check or None)
            check_seams(a.check or None)
        except TreeMismatch as exc:
            print(exc)
            return 1
        print(f"tensorfold tree {got}: the fork tfpp-ds41 was written against ({EXPECT_COMMIT[:9]}); "
              f"{len(SEAMS)} seams as pinned")
        return 0
    if a.show:
        print(json.dumps({"tree_sha": tree_sha(a.show), "tree_sha_lf": tree_sha_lf(a.show),
                          "family_sha_lf": family_sha(a.show), "pins": pin_hashes(a.show)}, indent=1))
        return 0
    changed = diff(a.diff)
    print(f"tree {tree_sha(a.diff)[:16]} (LF {tree_sha_lf(a.diff)[:16]}; expected {EXPECT_PKG_SHA[:16]}); "
          f"{len(changed)} of {len(SEAMS)} seams changed")
    for k in changed:
        print(f"  re-read: {k}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
