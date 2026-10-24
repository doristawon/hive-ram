"""The shortened test model S11 and its Engram test tables (spec 15.1-15.4), without torch.

  build SHARDS OUT [--sel 0,1,2,3,14,20,21,24,37,38,39] [--engram-vocab 320000] [--revision FULL_COMMIT]
        OUT: config.json (the selection renumbered, load.selection_config), tfpp_selection.json (the original layer
        numbers and the source revision), the source's model.safetensors.index.json unchanged, symlinks to the shards
        the selection reads (and their .cache/huggingface/download/<shard>.metadata, so VERIFY_SHARDS can check them
        against the Hub), the tokenizer / template / generation files copied. load.AliasShards renames on read.
        The revision is the source's (hf download's .metadata); a copy that lost the hidden .cache folder needs
        --revision FULL_COMMIT, the same on both nodes: tfpp_selection.json must be byte-identical across nodes.
  table OUT LAYER ROWS dense|zero [--seed S]
        one Engram table file as Engram.__init__ reads it (F/model.py:517-528): ``layers.<LAYER>.engram.embed.weight``
        F8_E4M3 [ROWS, 256] then ``.scale`` F8_E8M0 [ROWS, 8]. dense (Mode D, every equivalence test): deterministic
        bytes (counter splitmix64; the same file on every node: compare the printed sha256), finite E4M3 codes only,
        scale exponents 119..123 (2^-8 .. 2^-4). zero (Mode Z, smoke tests): a hole, every row reads as zeros.
  extract SHARD NAME OUT
        one tensor of a shard as a single-tensor safetensors file: a drafting leader that holds no layer 0 needs
        ``embed.weight`` (1.3 GB) but not shard 1 (6.5 GB). Write it into the leader's model folder under a name the index
        does not use (e.g. ``embed.weight.safetensors``) and leave shard 1 out of that folder: load.stage_files (the
        preflight) takes the file for the shard, AliasShards reads it, and a folder holding both is refused.
  selftest DIR
        small tables written and read back by the reader's offsets; S11's config numbers when DIR holds a config.
  downloads [--index INDEX] [--revision FULL_COMMIT] [--dir DIR] [--need-bytes]
        prints a script with the hf download commands for S11 (never runs them; REV is a line to edit and is checked);
        --need-bytes prints the free space the download wants (download_list.sh checks it).

The S11 numbers (spec 5.4 / 15.1): kv [2,4,5], index [2,4,5,7], cand 5, engram [1,4], taps [8,9,10], replay layer 5,
engram_num_embeddings [7682786, 7689630] at engram_vocab_size 320000 (both above the AIO self-test's row 7,654,321:
F/model.py:557).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

from .load import S11, SELECTION_FILE, needed_shards, selection_config, selection_namer, selection_record

ROW_W, ROW_S = 256, 8                    # engram_head_dim FP8 bytes + one UE8M0 scale a 32 (F/model.py:717-723)
REPO = "Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw"
REVISION_PREFIX = "45ee5d0f"             # spec 15.2 (pin the full commit when downloading)
# spec 15.2's small files (downloaded); build also copies generation_config.json when the source has one
SMALL_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "quantization_config.json")
COPIED = SMALL_FILES[1:] + ("generation_config.json",)
# spec 15.2: the shards S11 reads (model-000NN-of-00039), their content and bytes
S11_SHARDS = {
    1: ("L00, embed", 6547553300), 2: ("L01 (Engram wkv)", 5321110552), 3: ("L02", 5229561892),
    4: ("L03", 5222579220), 15: ("L14 (Engram wkv)", 5304008744), 20: ("L20, L21", 7044257844),
    23: ("L24", 5221770776), 36: ("L37", 5222579220), 37: ("L38", 5222579220),
    38: ("L39, head, norm, mtp.0", 8115809584), 39: ("mtp.1, mtp.2", 5817227304),
}
AIO_PROBE_ROW = 7654321
DISK_NEED_BYTES = 75 * 10**9              # spec 15.3: S11 (64.3 GB) + the Mode D tables (4.1 GB) + room; WSL df and C: both


def shard_name(k: int) -> str:
    return f"model-{k:05d}-of-00039.safetensors"


# -- Engram test tables ------------------------------------------------------------------------------------------------
def table_header(layer: int, rows: int, note: str) -> bytes:
    hdr = {f"layers.{layer}.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [rows, ROW_W],
                                                   "data_offsets": [0, rows * ROW_W]},
           f"layers.{layer}.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [rows, ROW_S],
                                                  "data_offsets": [rows * ROW_W, rows * (ROW_W + ROW_S)]},
           "__metadata__": {"tfpp": note}}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    return h + b" " * (-len(h) % 8)


M64 = 2 ** 64 - 1


def _mix_py(seed: int, start: int, n: int) -> bytearray:
    """splitmix64's low byte of counters start .. start + n - 1 (pure Python: self-tests and cross-checks)."""

    k = (seed * 0x9E3779B97F4A7C15) & M64
    out = bytearray(n)
    for j in range(n):
        x = (start + j + k) & M64
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
        out[j] = (x ^ (x >> 31)) & 0xFF
    return out


def _mix(seed: int, start: int, n: int):
    try:
        import numpy as np
    except ImportError:
        return _mix_py(seed, start, n)
    with np.errstate(over="ignore"):
        x = np.arange(start, start + n, dtype=np.uint64) + np.uint64((seed * 0x9E3779B97F4A7C15) & M64)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x & np.uint64(0xFF)).astype(np.uint8)


def _finish(region: str, b) -> bytes:
    """weight: E4M3FN NaN codes (0x7F, 0xFF) -> +-448 (0x7E, 0xFE); scale: exponents 119..123 (2^-8 .. 2^-4)."""

    if hasattr(b, "dtype"):
        if region == "weight":
            b[(b & 0x7F) == 0x7F] &= 0xFE
            return b.tobytes()
        return (119 + b % 5).astype(b.dtype).tobytes()
    if region == "weight":
        return bytes(x & 0xFE if (x & 0x7F) == 0x7F else x for x in b)
    return bytes(119 + x % 5 for x in b)


def write_table(path, layer: int, rows: int, mode: str, seed: int = 0, chunk: int = 1 << 26) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = table_header(layer, rows, f"tfpp-ds41 engram test table mode={mode} seed={seed}")
    digest = hashlib.sha256()
    with open(path, "wb") as f:
        head = struct.pack("<Q", len(h)) + h
        f.write(head)
        digest.update(head)
        base = len(head)
        if mode == "zero":
            f.truncate(base + rows * (ROW_W + ROW_S))
            return f"{path}: sparse, {base + rows * (ROW_W + ROW_S)} bytes apparent (header sha256 {digest.hexdigest()})"
        if mode != "dense":
            raise ValueError(f"table mode {mode!r}: dense or zero")
        for region, n, sd in (("weight", rows * ROW_W, 2 * seed + 1), ("scale", rows * ROW_S, 2 * seed + 2)):
            for a in range(0, n, chunk):
                b = _finish(region, _mix(sd * 1000003 + layer, a, min(chunk, n - a)))
                f.write(b)
                digest.update(b)
    return f"{path}: dense sha256 {digest.hexdigest()}"


def read_rows_like_engram(path, layer: int, ids) -> list[tuple[bytes, bytes]]:
    """F/model.py:517-528's offsets (8 + size + lo) and one read a row: what gather_rows2 reads (F/engram_io.cpp)."""

    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        w = header[f"layers.{layer}.engram.embed.weight"]
        s = header[f"layers.{layer}.engram.embed.scale"]
        out = []
        for i in ids:
            f.seek(8 + size + w["data_offsets"][0] + i * ROW_W)
            rw = f.read(ROW_W)
            f.seek(8 + size + s["data_offsets"][0] + i * ROW_S)
            out.append((rw, f.read(ROW_S)))
        return out


_ENGRAM_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.engram\.embed\.(?:weight|scale)$")


def engram_layer_of(name: str) -> int | None:
    """The layer of an Engram table tensor (``layers.<i>.engram.embed.weight`` / ``.scale``: F/model.py:525, 611)."""

    m = _ENGRAM_NAME.search(name)
    return int(m.group(1)) if m else None


def table_identity(paths) -> dict:
    """What HELLO compares of a stage's Engram tables: the engram layers they hold, their (name, dtype, shape,
    offsets) sha, file sizes, and the kind (real / dense / zero test table)."""

    names, sizes, modes, layers = [], [], set(), set()
    for p in sorted(Path(x) for x in paths):
        with open(p, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        meta = header.pop("__metadata__", {}) or {}
        note = str(meta.get("tfpp", ""))
        modes.add("test-dense" if "mode=dense" in note else "test-zero" if "mode=zero" in note else "real")
        for n, e in sorted(header.items()):
            if n.endswith("engram.embed.weight") or n.endswith("engram.embed.scale"):
                names.append([n, e["dtype"], e["shape"], e["data_offsets"]])
                layer = engram_layer_of(n)
                if layer is not None:
                    layers.add(layer)
        sizes.append(p.stat().st_size)
    return {"layers": sorted(layers), "sha": hashlib.sha256(json.dumps(names).encode()).hexdigest()[:16],
            "sizes": sizes, "mode": "+".join(sorted(modes)), "rows": {n[0]: n[2][0] for n in names}}


# -- the shortened model -----------------------------------------------------------------------------------------------
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
HF_META = Path(".cache") / "huggingface" / "download"


def source_revision(src: Path, explicit: str | None = None) -> str:
    """The source checkpoint's Hub commit: ``explicit`` (--revision) when given, checked against what the download left
    in .cache/huggingface; else that; else "unknown" (every stage must then say unknown)."""

    from tfpp.glm.env import model_revision

    found = model_revision(src)
    if explicit:
        explicit = explicit.strip().lower()
        if not FULL_COMMIT.match(explicit):
            raise ValueError(f"--revision {explicit!r}: the full 40-hex commit")
        if found != "unknown" and found != explicit:
            raise ValueError(f"--revision {explicit} but {src} was downloaded at {found}")
        return explicit
    return found


def link_shard(target: Path, link: Path, log=print) -> None:
    """``link`` -> ``target``. A link left by an earlier build is replaced when it points anywhere else (another source:
    its config, selection and .metadata would describe one checkpoint while the shards came from another) or nowhere
    (Path.exists follows links: a dangling one reads as absent, and os.symlink then fails on it). A regular file there is
    never deleted: it is kept, with a note (VERIFY_SHARDS=1 checks its bytes against the Hub)."""

    if link.is_symlink():
        if Path(os.path.realpath(link)) == Path(os.path.realpath(target)):
            return
        log(f"{link}: relinked to {target} (it pointed at {os.readlink(link)})")
        link.unlink()
    elif link.exists():
        log(f"{link}: a regular file, not a link to {target}: kept as it is")
        return
    os.symlink(target, link)


def build(shards: str, out: str, sel, engram_vocab: int | None, log=print, revision: str | None = None) -> list[str]:
    src, dst = Path(shards), Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    raw = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg = selection_config(raw, sel, engram_vocab=engram_vocab)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    rev = source_revision(src, revision)
    if rev == "unknown":
        log(f"WARNING: {src} has no .cache/huggingface metadata and no --revision: the revision is 'unknown' (the other "
            "node's S11 must say unknown too: HELLO compares it) and VERIFY_SHARDS checks nothing")
    (dst / SELECTION_FILE).write_text(json.dumps(selection_record(sel, engram_vocab=engram_vocab, revision=rev),
                                                 indent=1) + "\n", encoding="utf-8")
    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    files, unknown = needed_shards(index["weight_map"], range(len(sel)), first=True, last=True, drafter=True,
                                   rename=selection_namer(sel))
    if unknown:
        raise ValueError(f"checkpoint tensors the stage rules do not know: {unknown[:5]}")
    for f in files:
        if not (src / f).is_file():
            raise FileNotFoundError(src / f)
        link_shard((src / f).resolve(), dst / f, log)
        meta = src / HF_META / f"{f}.metadata"                  # verify_shards reads <model>/.cache/.../<shard>.metadata
        if meta.is_file():
            (dst / HF_META).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(meta, dst / HF_META / meta.name)
    for f in COPIED:
        if (src / f).is_file():
            shutil.copyfile(src / f, dst / f)
    log(f"{dst}: {len(files)} shards linked, selection {list(sel)}, revision {rev}")
    return files


def extract(shard: str, name: str, out: str) -> str:
    """One tensor of a shard as its own safetensors file, streamed."""

    with open(shard, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        e = header[name]
        lo, hi = e["data_offsets"]
        hdr = {name: {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [0, hi - lo]},
               "__metadata__": {"tfpp": f"extracted from {Path(shard).name}"}}
        h = json.dumps(hdr, separators=(",", ":")).encode()
        h += b" " * (-len(h) % 8)
        digest = hashlib.sha256()
        with open(out, "wb") as g:
            g.write(struct.pack("<Q", len(h)) + h)
            f.seek(8 + size + lo)
            left = hi - lo
            while left:
                b = f.read(min(left, 64 << 20))
                if not b:
                    raise IOError(f"short read of {name} in {shard}")
                g.write(b)
                digest.update(b)
                left -= len(b)
    return f"{out}: {name} {e['dtype']} {e['shape']} ({hi - lo} bytes, data sha256 {digest.hexdigest()})"


def selftest(d: str) -> list[str]:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    out = [write_table(d / "z.safetensors", 4, 5000, "zero"), write_table(d / "d.safetensors", 4, 5000, "dense", seed=7)]
    z = read_rows_like_engram(d / "z.safetensors", 4, [0, 1, 4097, 4999])
    dn = read_rows_like_engram(d / "d.safetensors", 4, [0, 1, 4097, 4999])
    assert all(w == bytes(ROW_W) and s == bytes(ROW_S) for w, s in z), "zero table rows are not zeros"
    assert all(all((x & 0x7F) != 0x7F for x in w) and all(119 <= x <= 123 for x in s) for w, s in dn), \
        "dense rows hold NaN codes or scales out of 119..123"
    assert len({w for w, _ in dn}) == 4, "dense rows repeat"
    # numpy and the pure-Python mix agree (the table is the same bytes wherever it is made)
    a = bytes(_finish("weight", bytearray(_mix_py(3 * 1000003 + 4, 1000, 512))))
    b = _finish("weight", _mix(3 * 1000003 + 4, 1000, 512))
    assert a == bytes(b), "numpy and Python splitmix64 differ"
    ident = table_identity([d / "d.safetensors"])
    assert ident["layers"] == [4] and ident["mode"] == "test-dense", ident
    out.append(f"selftest ok: rows {[w[:4].hex() + '/' + s.hex() for w, s in dn]}")
    cfgp = d / "config.json"
    if cfgp.is_file():
        t = json.loads(cfgp.read_text(encoding="utf-8")).get("text_config", {})
        out.append(f"config: kv {t.get('kv_source_layer_ids')} engram {t.get('engram_layer_ids')} rows "
                   f"{t.get('engram_num_embeddings')}")
    return out


def download_commands(*, index: str | None, revision: str | None, local_dir: str, sel=S11) -> list[str]:
    """A bash script: REV is one line to edit (the repo's full 40-hex commit, which starts with REVISION_PREFIX) and the
    script stops when it is not, so a placeholder never reaches ``hf download`` (and nothing silently downloads main)."""

    files = [shard_name(k) for k in sorted(S11_SHARDS)]
    total = sum(b for _, b in S11_SHARDS.values())
    if index:
        wm = json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"]
        files, unknown = needed_shards(wm, range(len(sel)), first=True, last=True, drafter=True,
                                       rename=selection_namer(sel))
        total = None
        if unknown:
            raise ValueError(f"tensors the rules do not know: {unknown[:5]}")
    lines = [f"# S11 = layers {list(sel)} of {REPO} (spec 15.2): {len(files)} shards"
             + (f", {total:,} bytes ({total / 2**30:.3f} GiB)" if total else ""),
             "# run by hand: once, then copy over USB4; VERIFY_SHARDS=1 checks the Hub sha256 at start",
             f"# before: WSL df >= {DISK_NEED_BYTES // 10**9} GB free AND the vhdx already that large or C: free enough "
             "(spec 15.3; download_list.sh checks both)",
             f"# REV: the repo's full commit (huggingface.co/{REPO}/commits/main), it starts with {REVISION_PREFIX}",
             "set -euo pipefail",
             f"REV='{revision or '<FULL_40_HEX_COMMIT>'}'",
             f'[[ "$REV" =~ ^{REVISION_PREFIX}[0-9a-f]{{32}}$ ]] || {{ echo "edit REV: the full 40-hex commit starting '
             f'{REVISION_PREFIX}" >&2; exit 1; }}',
             f'hf download {REPO} --revision "$REV" --local-dir {local_dir} \\']
    lines.append("  " + " ".join(SMALL_FILES) + " \\")
    for i, f in enumerate(files):
        lines.append(f"  {f}" + (" \\" if i < len(files) - 1 else ""))
    lines.append("# NOT the original Engram shards 47 / 48 of deepseek-ai/DeepSeek-V4.1-Flash-1017 (Mode D tables "
                 "instead: make_testmodel.sh)")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.ds41.testmodel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("shards")
    b.add_argument("out")
    b.add_argument("--sel", default=",".join(str(x) for x in S11))
    b.add_argument("--engram-vocab", type=int, default=None)
    b.add_argument("--revision", default="", help="the source's full Hub commit (when a copy lost .cache/huggingface)")
    t = sub.add_parser("table")
    t.add_argument("out")
    t.add_argument("layer", type=int)
    t.add_argument("rows", type=int)
    t.add_argument("mode", choices=("dense", "zero"))
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("extract")
    e.add_argument("shard")
    e.add_argument("name")
    e.add_argument("out")
    s = sub.add_parser("selftest")
    s.add_argument("dir")
    d = sub.add_parser("downloads")
    d.add_argument("--index", default="")
    d.add_argument("--revision", default="")
    d.add_argument("--dir", default="~/models/ds41-src")
    d.add_argument("--need-bytes", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.shards, a.out, [int(x) for x in a.sel.split(",") if x.strip()], a.engram_vocab,
              revision=a.revision or None)
    elif a.cmd == "table":
        if a.rows <= AIO_PROBE_ROW:
            print(f"note: {a.rows} rows: the AIO self-test reads row {AIO_PROBE_ROW} (F/model.py:557); it fails and "
                  "decode reads stay on the reader pool")
        print(write_table(a.out, a.layer, a.rows, a.mode, a.seed))
    elif a.cmd == "extract":
        print(extract(a.shard, a.name, a.out))
    elif a.cmd == "selftest":
        print("\n".join(selftest(a.dir)))
    elif a.need_bytes:
        print(DISK_NEED_BYTES)
    else:
        print("\n".join(download_commands(index=a.index or None, revision=a.revision or None, local_dir=a.dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
