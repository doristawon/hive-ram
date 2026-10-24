"""Which MIA seams tfpp-glm v2 relies on (no torch: runs on any CPU Python), as ``pins.py`` does for v1 (spec v2 section 18).

v2 does not replace MIA's scheduler or decoder: it subclasses ``MultiDecoder`` / ``BatchedVerify`` and drives
``GlmScheduler`` from outside. So what must not change under it is the source of the methods it overrides, wraps or
re-implements a part of, and of the helpers it calls with assumptions about their return (e.g. that ``_flush`` is
the only place a message leaves, that ``_admit`` does no link I/O). ``PINS`` is the sha256 of each seam's source
(``ast.get_source_segment``); a new package version lists exactly the seams to re-read. Not a start condition.

  python -m tfpp.glm.v2pins --check TREE    the v2 seams of TREE against PINS (exit 1: re-read the listed ones)
  python -m tfpp.glm.v2pins --print TREE    the pins of TREE (to refresh PINS after a review)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

from . import pins

M = "families/glm5_next/cuda/multi.py"
V = "families/glm5_next/cuda/verify.py"
MP = "families/glm5_next/cuda/multi_prefill.py"
E = "families/glm5_next/cuda/engine.py"
DEC = "families/glm5_next/cuda/decode.py"
FWD = "families/glm5_next/cuda/forward.py"

SEAMS = (
    # MultiDecoder: overridden (round, finish, _admitted, apply, _fill, _fill_many, _round), wrapped or relied on
    (M, "MultiDecoder.__init__"), (M, "MultiDecoder.round"), (M, "MultiDecoder._iterate"),
    (M, "MultiDecoder._flush"), (M, "MultiDecoder._emit"), (M, "MultiDecoder.parse"), (M, "MultiDecoder.admit"),
    (M, "MultiDecoder._admitted"), (M, "MultiDecoder._parse_admit"), (M, "MultiDecoder._keep"),
    (M, "MultiDecoder._fill"), (M, "MultiDecoder._fill_many"), (M, "MultiDecoder._round"),
    (M, "MultiDecoder._started"), (M, "MultiDecoder._stop_bound"), (M, "MultiDecoder.apply"),
    (M, "MultiDecoder.follow"), (M, "MultiDecoder.finish"), (M, "MultiDecoder._finish"),
    # message helpers
    (M, "seal"), (M, "unseal"), (M, "digest"), (M, "trim"), (M, "cap_victim"), (M, "multi_code"),
    # the scheduler, driven from outside (lockstep, tags, the warm-up's wake-up item)
    (M, "GlmScheduler.__init__"), (M, "GlmScheduler.submit"), (M, "GlmScheduler._iteration"),
    (M, "GlmScheduler._loop"), (M, "GlmScheduler._admit"), (M, "GlmScheduler._yield"),
    (M, "GlmScheduler._requeue"), (M, "GlmScheduler._cancel_item"), (M, "GlmScheduler._Flush"),
    # the batched verify seam (the whole class), the serial one's forward
    (V, "BatchedVerify"), (V, "SerialVerify.forward"),
    # multi-prompt chunks
    (MP, "prefill_pieces"), (MP, "layout"), (MP, "PieceMixer.__init__"),
    # the engine's arithmetic and idle hooks
    (E, "multi_draft_bytes"), (E, "slot_bytes"), (E, "kept_entries"), (E, "multi_window"),
    (E, "multi_decode_rows"), (E, "GlmEngine._ring"), (E, "GlmEngine._await_bell"),
    (DEC, "Engine.__init__"), (DEC, "Engine.forward"), (DEC, "prefill_chunk"),
    (FWD, "compute_steps"),
    # the package's scheduler and streams
    ("cuda/scheduler.py", "Waiting"), ("cuda/scheduler.py", "Scheduler.__init__"),
    ("cuda/scheduler.py", "Scheduler._reply"),
    ("cuda/streams.py", "Stream.take"), ("cuda/streams.py", "Stream.continued"),
)

PINS: dict[str, str] = {
    "families/glm5_next/cuda/multi.py::MultiDecoder.__init__":
        "78f5293df9b981184f761fe0d499aff07e78050f9969e0cb323d4897bc0bccec",
    "families/glm5_next/cuda/multi.py::MultiDecoder.round":
        "59e7bbd4dae4442cc50802c139427fbdb2665b29fbd17ce964260e6f44e52fb2",
    "families/glm5_next/cuda/multi.py::MultiDecoder._iterate":
        "0675bd5519c21119dcca25c5d94929e412e73ce99047d0abf5ba9f9a1773e55e",
    "families/glm5_next/cuda/multi.py::MultiDecoder._flush":
        "544106ab91d3a72c303aad27f07723b1147da251f00abd473729c05b88e760bc",
    "families/glm5_next/cuda/multi.py::MultiDecoder._emit":
        "514e546258c80d56756528bb4172ffdbf78b72cf9aac489ea54b64570532b1b9",
    "families/glm5_next/cuda/multi.py::MultiDecoder.parse":
        "f418327fd743c69ce2918315e76e4919f2cb4bab7926cdca192b577d9fe50966",
    "families/glm5_next/cuda/multi.py::MultiDecoder.admit":
        "84395b9c7eb78c5183984d7fa134de2bc677f230fa6e4d9e9fef2e521cde0282",
    "families/glm5_next/cuda/multi.py::MultiDecoder._admitted":
        "24aca7123eda79eef87b6ef95e226e49cffecc489fb478f7ac4999626f295859",
    "families/glm5_next/cuda/multi.py::MultiDecoder._parse_admit":
        "8607af710fd683da3516125ee89cc73da54c82e8c63cbec8b8f530ce46afb0ab",
    "families/glm5_next/cuda/multi.py::MultiDecoder._keep":
        "d799600f72450f1eb57319ce5c97283dd3caedbbd28d92bb5c4ec2fb40e0a85e",
    "families/glm5_next/cuda/multi.py::MultiDecoder._fill":
        "3a28e6831f5d548824ae7b5f05bf6cd303711744372d44dd40641638302fce0f",
    "families/glm5_next/cuda/multi.py::MultiDecoder._fill_many":
        "96d67123b8b009f3b6e9f12911757079dd463a42e53ee276671813a922aba419",
    "families/glm5_next/cuda/multi.py::MultiDecoder._round":
        "26d34bc45f34994563974dc5643700cfc8ca1da75613bbbec0c504096419cf0e",
    "families/glm5_next/cuda/multi.py::MultiDecoder._started":
        "be988de0c66f7f4bb1b6551b131b03b94fc910a505b5aa1c3f94d9cf6c57a0e9",
    "families/glm5_next/cuda/multi.py::MultiDecoder._stop_bound":
        "8ca95de55446aca97f2d4d1ede2c9f9154279d1b6cb2ff1a4bd2c65ea25b0345",
    "families/glm5_next/cuda/multi.py::MultiDecoder.apply":
        "ed13cbff1be3414173fdfd0a714cb073f714cad8b9fc25a2ae15323b1207607a",
    "families/glm5_next/cuda/multi.py::MultiDecoder.follow":
        "ed54e2e4640112aab0525f356592236c7a617a97918f374619248b7f02fd57a4",
    "families/glm5_next/cuda/multi.py::MultiDecoder.finish":
        "2e543c8ae5a6a9b505cd7cec1e141d8cebd165f0424a385cd75245b51880fbfa",
    "families/glm5_next/cuda/multi.py::MultiDecoder._finish":
        "2573be857c25ba8c2988d4f4ed2614046031d584fe513432842178958213dd82",
    "families/glm5_next/cuda/multi.py::seal":
        "da04ca1334b035388b6efe8fd3b5c1049d67c20aa5b6c4cf0eb26257d202b81a",
    "families/glm5_next/cuda/multi.py::unseal":
        "ddb91e8f3e7bd7ee9cd58092201b83d86a7042f28c989ed1abb763fc7a6cda93",
    "families/glm5_next/cuda/multi.py::digest":
        "69a946b7b2c0c204f1d1b883aa94015b8d90cc3a0896e13afa381022ca603f8d",
    "families/glm5_next/cuda/multi.py::trim":
        "c48f03e0a837b16266347b6f10b87a4927b6917086f12259a99dd042b616ffad",
    "families/glm5_next/cuda/multi.py::cap_victim":
        "d3e2cf8541c73b69478693916a2c25e520857f53ab7253fba5a06069aa2a5872",
    "families/glm5_next/cuda/multi.py::multi_code":
        "15611dc6ae9ff40eae8e3efcf68335f3bc86d10faf0f67e437107613a21e268d",
    "families/glm5_next/cuda/multi.py::GlmScheduler.__init__":
        "3848639d6d891732ce3657a925885e95225797ee926a6bbd84d73431932111b8",
    "families/glm5_next/cuda/multi.py::GlmScheduler.submit":
        "ba37b58ae3eadb40ab99eb80059502f61bf52343125d1de8e826b7905453228f",
    "families/glm5_next/cuda/multi.py::GlmScheduler._iteration":
        "10166346d81ef15ba70f67c60b6525aede346d4913a6097680c327abefa76d01",
    "families/glm5_next/cuda/multi.py::GlmScheduler._loop":
        "a2aa4b5afe85d625fbefb196998a1eb41f3bad078fbf5ad948671081e8921680",
    "families/glm5_next/cuda/multi.py::GlmScheduler._admit":
        "06a5ee5b4942a1225adaedc0ec6739a4e26124b0cba886070f34908769c5dc53",
    "families/glm5_next/cuda/multi.py::GlmScheduler._yield":
        "8fc1703539c6e6d01d2e07fd9aef0e2405eb1d64a23fd543a8dbd587ec79efa7",
    "families/glm5_next/cuda/multi.py::GlmScheduler._requeue":
        "495c35f353e1f9689bda29eca45288998a3e9961dc82aab9b2d429852d0340f9",
    "families/glm5_next/cuda/multi.py::GlmScheduler._cancel_item":
        "d3bb360d4fe5703d1f14de40a21c6bad1776ae1e8cd2d1b31650f0597b1d9e8c",
    "families/glm5_next/cuda/multi.py::GlmScheduler._Flush":
        "9733e0d21e2584d2628268383ccbbc453f1e56f15a5c2ff8a68323a8157934a9",
    "families/glm5_next/cuda/verify.py::BatchedVerify":
        "551eec4058e7d2f38c075383bb3fa1d77891808ff8df5459ac6ae907ac90a99f",
    "families/glm5_next/cuda/verify.py::SerialVerify.forward":
        "7938119970eb7b10a6189645361424a0db80c437e455d67a88cc36a7b20ba18d",
    "families/glm5_next/cuda/multi_prefill.py::prefill_pieces":
        "38cba4bcaa06ab24d80ef27cab652e40e21955837aef63d3f98282355d473872",
    "families/glm5_next/cuda/multi_prefill.py::layout":
        "128e56a84dfc4ec716301397487ad68109ee495a333b726d08504d5b81335ee5",
    "families/glm5_next/cuda/multi_prefill.py::PieceMixer.__init__":
        "621abebf1a79eb803accec5670f19b909942aba3803aa680acd1ec6fef7e8a3a",
    "families/glm5_next/cuda/engine.py::multi_draft_bytes":
        "f35ac2a7a89d6cc254ee3888032481dbe91a275844d1f96ffb54a48840dbc94b",
    "families/glm5_next/cuda/engine.py::slot_bytes":
        "3a450de9fe55b1c4f67362969d2d0930db42b849fac42e1e4046e92abc242d9f",
    "families/glm5_next/cuda/engine.py::kept_entries":
        "6a5ad0e90cd07c08c98b92b9531bba6112369453c16b4ba0bb193afee3ac338c",
    "families/glm5_next/cuda/engine.py::multi_window":
        "8c30bafadf5edd7d05be95909e1c50f64bb2040a2ec5967d137cf4377ed0174c",
    "families/glm5_next/cuda/engine.py::multi_decode_rows":
        "c0922e1925b34b51cc70952450e877c08762a7f9320334b5268992a782b11188",
    "families/glm5_next/cuda/engine.py::GlmEngine._ring":
        "02042f09db04da4ba73afa71597137c5ad6e31bd650236a683859a13c8c7d026",
    "families/glm5_next/cuda/engine.py::GlmEngine._await_bell":
        "c1e08d83538a3644422724656897da3df66cd71c23906b23655e61c91a51ac7d",
    "families/glm5_next/cuda/decode.py::Engine.__init__":
        "e5dade882011916a2bd4c66a79d9e857b98a65403b9a18261d50361d7d1fabb0",
    "families/glm5_next/cuda/decode.py::Engine.forward":
        "7c809bbd644f143bb4dbbf13e64484121a2ef894076732157bd3558cb5d790d5",
    "families/glm5_next/cuda/decode.py::prefill_chunk":
        "b5a88f257b8ac659cef1401c5b50608c285e7e5c9b66b4f6ebaffcf0a96cabd7",
    "families/glm5_next/cuda/forward.py::compute_steps":
        "a4fa81b436c98976db9c3b84299e84a9e52a6e1160f21f1b063854265905cb71",
    "cuda/scheduler.py::Waiting":
        "435b332d6be0ee7781bc6b7b5568e2d37141ec637d69cbfde608264f47815954",
    "cuda/scheduler.py::Scheduler.__init__":
        "69e0127bfd77a1b9d58f45db3c2cfeb6b1556d160557e7e3d9e1344c7b737d69",
    "cuda/scheduler.py::Scheduler._reply":
        "00a8ada0b9f4652631c16af8a64374b6f383300537d4d47aee2d8b2ce786cb4d",
    "cuda/streams.py::Stream.take":
        "9e4641fd8eece1ead2830872ecf35d2af0a2be2f370e81e4925832bee59bfb00",
    "cuda/streams.py::Stream.continued":
        "bf4e1840de2790d54244284096ba05b6ff6ab43a6caedf5d55a8dc6b6d3d00b8",
}


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
        node = pins._find(tree, qual)
        seg = None if node is None else ast.get_source_segment(src, node)
        out[key] = "missing" if seg is None else hashlib.sha256(seg.encode("utf-8")).hexdigest()
    return out


def diff(root: str | Path, expected: dict[str, str] | None = None) -> list[str]:
    expected = PINS if expected is None else expected
    now = pin_hashes(root)
    return [k for k in now if now[k] != expected.get(k)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.glm.v2pins", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", metavar="TREE")
    g.add_argument("--print", dest="show", metavar="TREE")
    a = ap.parse_args(argv)
    if a.show:
        print(json.dumps(pin_hashes(a.show), indent=1))
        return 0
    changed = diff(a.check)
    print(f"{len(changed)} of {len(SEAMS)} v2 seams changed")
    for k in changed:
        print(f"  re-read: {k}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
