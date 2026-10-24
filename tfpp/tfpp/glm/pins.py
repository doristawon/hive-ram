"""Which MIA package tfpp-glm was written against (no torch: runs on any CPU Python).

``EXPECT_PKG_SHA``: the sha256 of the tensorfold tree's sources (*.py *.cu *.cuh *.cpp *.c *.h), each file's
(relative path, sha256) sorted, hashed as ``path\\0digest\\n`` lines. A stage refuses to start on another tree: the
seams below (module attributes swapped, methods overridden or copied) are only known to be right on this one.

``PINS``: the sha256 of each seam's source (``ast.get_source_segment``), so a new package version lists exactly the
seams to re-read: ``python -m tfpp.glm.pins --diff <new tensorfold tree>``. They are not a start condition.

  python -m tfpp.glm.pins --check [TREE]   tree digest of TREE (default: the installed package) against EXPECT_PKG_SHA
  python -m tfpp.glm.pins --print TREE     the digest and the pins of TREE (to refresh this file after a review)
  python -m tfpp.glm.pins --diff TREE      the seams whose source differs in TREE
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".c", ".h")

# the MIA package v0.6.0 + 107 patches (v1.11-pre), byte-identical to the package inside MIA's published v1.11-pre
# container image (a vendored copy: tensorfold-mia-v111pre/tensorfold, computed by tree_sha on 2026-10-23)
EXPECT_PKG_SHA = "739ce9917c1d75a04307e1e6d80eae6b7cb128f6877c45fad66e90f0e2ffc198"

# (file relative to the tensorfold package, qualified name): what tfpp-glm swaps, overrides, copies or relies on
SEAMS = (
    ("families/glm5_next/cuda/forward.py", "_compute"),          # replaced by part._compute_pp
    ("families/glm5_next/cuda/forward.py", "compute"),           # looks _compute up at call time
    ("families/glm5_next/cuda/forward.py", "commit"),            # KEEP's commit on every stage
    ("families/glm5_next/cuda/forward.py", "layer_forward"),     # b.x is the only state between layers
    ("families/glm5_next/cuda/forward.py", "stage"),
    ("families/glm5_next/cuda/forward.py", "check_room"),
    ("families/glm5_next/cuda/forward.py", "gather"),            # comm None: this rank's partial only
    ("families/glm5_next/cuda/forward.py", "Buffers.__init__"),  # x torch.empty, logits by w.head.n
    ("families/glm5_next/cuda/forward.py", "Buffers.set_taps"),
    ("families/glm5_next/cuda/graphs.py", "Graphs.__init__"),    # two warm-up runs, then the capture (scrub)
    ("families/glm5_next/cuda/graphs.py", "Graphs._capture_sparse"),
    ("families/glm5_next/cuda/decode.py", "sample_rows"),        # the comm None branch
    ("families/glm5_next/cuda/decode.py", "Engine.__init__"),
    ("families/glm5_next/cuda/decode.py", "Engine.forward"),     # TailEngine.forward wraps it
    ("families/glm5_next/cuda/decode.py", "take_snapshot"),
    ("families/glm5_next/cuda/decode.py", "save_rows"),
    ("families/glm5_next/cuda/decode.py", "load_rows"),
    ("families/glm5_next/cuda/decode.py", "restore"),
    ("families/glm5_next/cuda/decode.py", "whole"),
    ("families/glm5_next/cuda/decode.py", "prefill"),            # wrapped (PREFILL), its checks copied
    ("families/glm5_next/cuda/decode.py", "prefill_chunk"),      # the followers' prompt path
    ("families/glm5_next/cuda/decode.py", "serial_decode"),      # call decode.commit by global name
    ("families/glm5_next/cuda/decode.py", "mtp_decode"),
    ("families/glm5_next/cuda/decode.py", "dflash_decode"),
    ("families/glm5_next/cuda/drafter_choice.py", "auto_decode"),  # drafter_choice.commit (bound at import)
    ("families/glm5_next/cuda/dflash2.py", "Drafter.__init__"),  # tap_layers from the config
    ("families/glm5_next/cuda/weights.py", "load"),              # the five loader anchors
    ("families/glm5_next/cuda/engine.py", "GlmEngine.__init__"),  # PipeGlmEngine.__init__ follows it step by step
    ("families/glm5_next/cuda/engine.py", "GlmEngine._share"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._drop"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._run"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._run_once"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._effective"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._take_over"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._remember"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._resume"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._ring"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine._await_bell"),
    ("families/glm5_next/cuda/engine.py", "GlmEngine.generate"),
    ("families/glm5_next/cuda/engine.py", "VisionFeed"),
    ("families/glm5_next/cuda/engine.py", "mtp_head"),
    ("families/glm5_next/cuda/engine.py", "decode_policy"),
    ("families/glm5_next/cuda/app.py", "GlmApp"),
    ("cuda/capacity.py", "make_plan"),
    ("cuda/capacity.py", "choose"),
    ("cuda/geometry.py", "mla_geometry"),
    ("cuda/geometry.py", "dflash2_geometry"),
    ("cuda/geometry.py", "dflash2_weights"),
    # v2 (spec section 14): reviewed when the multi-stream port is written
    ("families/glm5_next/cuda/multi.py", "MultiDecoder.apply"),
    ("families/glm5_next/cuda/multi.py", "MultiDecoder._fill"),
    ("families/glm5_next/cuda/multi.py", "MultiDecoder._fill_many"),
    ("families/glm5_next/cuda/multi.py", "MultiDecoder._round"),
    ("families/glm5_next/cuda/multi.py", "MultiDecoder.follow"),
    ("families/glm5_next/cuda/verify.py", "BatchedVerify.forward"),
    ("families/glm5_next/cuda/verify.py", "BatchedVerify.commit"),
    ("families/glm5_next/cuda/verify.py", "BatchedVerify.capture"),
)

PINS: dict[str, str] = {
    "families/glm5_next/cuda/forward.py::_compute": "19696c74b610ccc362153fbdaaf1ec151963e50c18b13dfa06e257b084fd71c5",
    "families/glm5_next/cuda/forward.py::compute": "73bcd4484d80ff2455fb245e55ac4991b1f34447e4ac2221e85e4c351555a49e",
    "families/glm5_next/cuda/forward.py::commit": "961072be190f9c1a97a8e74f69f30701f40d0b2a362b761432fc62e89a10102c",
    "families/glm5_next/cuda/forward.py::layer_forward": "aa91444d26dc8c1aa6f90127c40834eb19ca6ad26061b10118db178f2157bcda",
    "families/glm5_next/cuda/forward.py::stage": "463c2a342aedc3005829a12179dc09b53b190c25e0d89f6b702463c128e9b014",
    "families/glm5_next/cuda/forward.py::check_room": "a66efc6c53f00d6149b5b60b884a97e7b9bf2490a60b7c2ff1b259abda2076cc",
    "families/glm5_next/cuda/forward.py::gather": "a676b10a6b1ef1894732cf334d7856c4cd86a11d1d5e6cb79c29b7669c65e07c",
    "families/glm5_next/cuda/forward.py::Buffers.__init__": "36d1ae5a23dcae14acc15d15a76cb808b1d7652756039971fed62bbb716e9780",
    "families/glm5_next/cuda/forward.py::Buffers.set_taps": "f37b886d1d9e11c5afbc18883dbc8c685df37ac9d617b6d82698a95088c386d0",
    "families/glm5_next/cuda/graphs.py::Graphs.__init__": "18a3cf69292f628c5587d8fc030b1db98350e10e83ed30c8244bf9649d889b5a",
    "families/glm5_next/cuda/graphs.py::Graphs._capture_sparse": "01093a1a8d31b45148b8b44892834b9bfe5abe9ce089bf435d6005a1685f9749",
    "families/glm5_next/cuda/decode.py::sample_rows": "6475993ed04ae7720d59b8abb5449c85ddc9376f03ae944a3222df77b4aeed8b",
    "families/glm5_next/cuda/decode.py::Engine.__init__": "e5dade882011916a2bd4c66a79d9e857b98a65403b9a18261d50361d7d1fabb0",
    "families/glm5_next/cuda/decode.py::Engine.forward": "7c809bbd644f143bb4dbbf13e64484121a2ef894076732157bd3558cb5d790d5",
    "families/glm5_next/cuda/decode.py::take_snapshot": "e0c14ffb63575bf6f3f8ff32083d9b5c3919bfb6353b030be53d379cfee347bd",
    "families/glm5_next/cuda/decode.py::save_rows": "bbc276d75be057a16d784493c5bc82313d3d193a9bef2cf47637e40f8f6e2dec",
    "families/glm5_next/cuda/decode.py::load_rows": "e0bc52b2aa80b3e9f3ffc49175a014d048897781f7cd2fe404975988257147a3",
    "families/glm5_next/cuda/decode.py::restore": "5a2f22be78214956c05311345526da18dae6432fbbdb371fb66a5a5df9a76c3d",
    "families/glm5_next/cuda/decode.py::whole": "8f453be27a1aab488487d0bebf45eaba2718cc9073ae9df3c4a883f684b16149",
    "families/glm5_next/cuda/decode.py::prefill": "b85d1b3dfe7924f8d7c73cb28c63867c7fe8270cd0b71ae6a480a1fb079a4436",
    "families/glm5_next/cuda/decode.py::prefill_chunk": "b5a88f257b8ac659cef1401c5b50608c285e7e5c9b66b4f6ebaffcf0a96cabd7",
    "families/glm5_next/cuda/decode.py::serial_decode": "20db4241e1eb22474bc0695c72abda471c6bca89bc612a55e191293df201d4dc",
    "families/glm5_next/cuda/decode.py::mtp_decode": "a728320dc7cc63c0ad962e32a0feddb518ca054922a2dbbb2f55f8b732351930",
    "families/glm5_next/cuda/decode.py::dflash_decode": "8aebef474753f968cce98c35f3539338aab88efb6cfa37fdfeae0daf0c6f90f0",
    "families/glm5_next/cuda/drafter_choice.py::auto_decode": "7038b88a5b2f31d29ecba4c133923b2a79e78ddf9e69e4695f777b7d4fe00b11",
    "families/glm5_next/cuda/dflash2.py::Drafter.__init__": "576f0ae6d6e57001ae33597997e4105826de6fe25650cd4c6ca77292068e454e",
    "families/glm5_next/cuda/weights.py::load": "57fde4f7a53447d7cb95c2db184e4b573ad91fb2800391f47b0121fb3bca42e3",
    "families/glm5_next/cuda/engine.py::GlmEngine.__init__": "3e644a336247bec2a3e5a4c6aa6d6530ca65e71e9ffb08c3a0d068839a765584",
    "families/glm5_next/cuda/engine.py::GlmEngine._share": "e392fe26936a8a3a95c0939415c49e14f2a544aa049876f1c9d3f3c301c6a511",
    "families/glm5_next/cuda/engine.py::GlmEngine._drop": "f2afca936d4d83f23a613c1d796d7f667b77f4a5ab992a4510a1cf86d48f3963",
    "families/glm5_next/cuda/engine.py::GlmEngine._run": "094dc5514f8ead1a9ea8ca26ddc1e1202e6961622cdf6321250e46666b8f9a4f",
    "families/glm5_next/cuda/engine.py::GlmEngine._run_once": "17bded85bcb3c54bf8b8572caf3201792e47f366c16dfadb8201253dd652c759",
    "families/glm5_next/cuda/engine.py::GlmEngine._effective": "d752b0d0d2f511d49760267bf90d0649109e4a3557eb0b0f69bfa3bd635b284b",
    "families/glm5_next/cuda/engine.py::GlmEngine._take_over": "770290fa6c6eb44d9db4a039b6863cfe49b35888296b941199335ce04bdb5ca4",
    "families/glm5_next/cuda/engine.py::GlmEngine._remember": "0ccafe3b814105e6ca18d5ef315c5e05ea0547b3f05c0759dcd19fe8c16433cf",
    "families/glm5_next/cuda/engine.py::GlmEngine._resume": "d55abeebaf8e6fd8dd32280cbbbdf502d134ef118ac2d88da3da7e24bf0b7b7f",
    "families/glm5_next/cuda/engine.py::GlmEngine._ring": "02042f09db04da4ba73afa71597137c5ad6e31bd650236a683859a13c8c7d026",
    "families/glm5_next/cuda/engine.py::GlmEngine._await_bell": "c1e08d83538a3644422724656897da3df66cd71c23906b23655e61c91a51ac7d",
    "families/glm5_next/cuda/engine.py::GlmEngine.generate": "6d06066ef3b205b3e56701f566af44d314daa1641215de465e11aec6463d0e75",
    "families/glm5_next/cuda/engine.py::VisionFeed": "a47912251a5df772a848dbe87d90cf8660498627a88dd57f760df0ef04d65208",
    "families/glm5_next/cuda/engine.py::mtp_head": "e4777afb0cbbc0d423f84f33ec7f2e611519730ebb3fdbcb07f69e0c70b76f21",
    "families/glm5_next/cuda/engine.py::decode_policy": "6ee1898c2a1d73ae520af6ff7ebebc2e69b8960003f8dce8ca156df89d638bb3",
    "families/glm5_next/cuda/app.py::GlmApp": "1197e99cc2293bb21b06357c8391dd9d0936c983aaffa2d428a0231975eb20c1",
    "cuda/capacity.py::make_plan": "747eac78ca031b01ab12ec9cb59374081ede02be084d8a8e98a97a9ab817cd0f",
    "cuda/capacity.py::choose": "3231aad6ffc1be7b6ff897295502083a8e241d39b93b0ecb4b1d2fa359ab4c33",
    "cuda/geometry.py::mla_geometry": "3fb263fba4f921dc6216f7a63070a5d37832c244f2972789a3889cf4c892f524",
    "cuda/geometry.py::dflash2_geometry": "baa6a3cc4f083f3112849188cbe87bd65f01aca538f4f6d6c9b6cc78afcedb8f",
    "cuda/geometry.py::dflash2_weights": "a17a9a8cdedb106ff52539e0e91e12c5c27fbb64967700505f2e75ecb7724f67",
    "families/glm5_next/cuda/multi.py::MultiDecoder.apply": "ed13cbff1be3414173fdfd0a714cb073f714cad8b9fc25a2ae15323b1207607a",
    "families/glm5_next/cuda/multi.py::MultiDecoder._fill": "3a28e6831f5d548824ae7b5f05bf6cd303711744372d44dd40641638302fce0f",
    "families/glm5_next/cuda/multi.py::MultiDecoder._fill_many": "96d67123b8b009f3b6e9f12911757079dd463a42e53ee276671813a922aba419",
    "families/glm5_next/cuda/multi.py::MultiDecoder._round": "26d34bc45f34994563974dc5643700cfc8ca1da75613bbbec0c504096419cf0e",
    "families/glm5_next/cuda/multi.py::MultiDecoder.follow": "ed54e2e4640112aab0525f356592236c7a617a97918f374619248b7f02fd57a4",
    "families/glm5_next/cuda/verify.py::BatchedVerify.forward": "8c31a91458765de40ae2b85698cf35a5a3b35ffcff9db8f47637313b9fb9c349",
    "families/glm5_next/cuda/verify.py::BatchedVerify.commit": "7479b296dc767720fcba450eca49d7f6bfde320d801febafa5e76fd1d09b6aaa",
    "families/glm5_next/cuda/verify.py::BatchedVerify.capture": "a366d0fd26d9527f85c569150758cb468322ccf1c0f31d12a95e156327e48614",
}


def tree_sha(root: str | Path) -> str:
    root = Path(root)
    if not (root / "__init__.py").is_file():
        raise FileNotFoundError(f"{root} is not the tensorfold package folder")
    items = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts:
            items.append((p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()))
    h = hashlib.sha256()
    for rel, digest in sorted(items):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def installed_root() -> Path:
    """The tensorfold package folder Python would import (found without importing it)."""

    spec = importlib.util.find_spec("tensorfold")
    if spec is None or not spec.origin:
        raise RuntimeError("no tensorfold package on this Python's path")
    return Path(spec.origin).resolve().parent


class TreeMismatch(RuntimeError):
    pass


def check_tree(root: str | Path | None = None, expect: str = EXPECT_PKG_SHA) -> str:
    """The tree digest of ``root`` (default: the importable tensorfold), refusing anything but ``expect``."""

    root = installed_root() if root is None else Path(root)
    got = tree_sha(root)
    if got != expect:
        raise TreeMismatch(f"the tensorfold tree at {root} has digest {got[:16]}, not the MIA v1.11-pre package "
                           f"tfpp-glm was written against ({expect[:16]}): rsync vendor/tensorfold-mia-v111pre/"
                           "tensorfold into this venv, or re-review the seams (python -m tfpp.glm.pins --diff TREE)")
    return got


def _find(tree: ast.Module, qualname: str):
    node: ast.AST = tree
    for part in qualname.split("."):
        body = getattr(node, "body", [])
        found = None
        for child in body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part:
                found = child
        if found is None:
            return None
        node = found
    return node


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
            src = path.read_text(encoding="utf-8")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.glm.pins", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", nargs="?", const="", metavar="TREE")
    g.add_argument("--print", dest="show", metavar="TREE")
    g.add_argument("--diff", metavar="TREE")
    a = ap.parse_args(argv)
    if a.check is not None:
        try:
            got = check_tree(a.check or None)
        except TreeMismatch as exc:
            print(exc)
            return 1
        print(f"tensorfold tree {got}: the MIA package tfpp-glm was written against")
        return 0
    if a.show:
        print(json.dumps({"tree_sha": tree_sha(a.show), "pins": pin_hashes(a.show)}, indent=1))
        return 0
    changed = diff(a.diff)
    print(f"tree {tree_sha(a.diff)[:16]} (expected {EXPECT_PKG_SHA[:16]}); {len(changed)} of {len(SEAMS)} seams changed")
    for k in changed:
        print(f"  re-read: {k}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
