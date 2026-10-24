"""Load one stage's layer range of GLM-5.3-Flash with the MIA package's own loader (G/weights.py:283-490).

The loader already takes ``rank`` and ``world``; at world=1 ``RankReader`` and ``tp.Layout`` cut nothing (every
family's share is the whole axis), so only the layer list and the per-role tensors change. A copy of the module's
source is compiled with five string anchors replaced (each must occur exactly once), as tfpp/load.py does for Flash
Next; every tensor is still built by the unchanged code (TF_GLM_DENSE=q4, EXL3 experts, the FP8 head and its 4-bit
draft copy, kv_b's absorb). ``patched_source``, ``mine`` and ``needed_shards`` run without torch.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import types
from pathlib import Path

# the dataclasses weights.py defines: the compiled copy builds instances of the originals (Weights.nbytes' isinstance,
# G/weights.py:263; type identity everywhere else)
CLASSES = ("Config", "HCW", "KDAW", "IndexW", "DSAW", "MLPW", "MoEW", "LayerW", "MTPW", "Weights")

ANCHORS = (
    # L1 (G/weights.py:468) the layer list: this stage's range only
    ("        which = list(range(cfg.layers))\n",
     "        which = list(_TFPP_LAYERS)\n"),
    # L2 (G/weights.py:446) no read-ahead of a layer this stage does not build (its shard may be absent; reading it
    #    costs host buffers and IO for nothing). _TFPP_NEXT empty: one layer in flight (--no-prefetch-next, as on
    #    the 64 GB node)
    ("        rd.prefetch(expert_names(i + 1), up)",
     "        rd.prefetch(expert_names(i + 1) if (i + 1) in _TFPP_NEXT else [], up)"),
    # L3 (G/weights.py:462-463) the embedding: stage 0 (input) and the last stage when it drafts (DFlash2's block
    #    and the MTP head read it: G/dflash2.py:451, G/mtp.py:36)
    ('    if exl3:\n        embed = rd.get(PREFIX + "embed_tokens.weight")',
     '    if not _TFPP_EMBED:\n        embed = None\n    elif exl3:\n        embed = rd.get(PREFIX + "embed_tokens.weight")'),
    # L4 (G/weights.py:471-473) the head: the last stage only; elsewhere a 1-row BF16 stand-in, because
    #    forward.Buffers sizes its logits by w.head.n (G/forward.py:138) and decode.Engine reads it (G/decode.py:219)
    ('        draft_head = None\n        if exl3:\n            raw = rd.get("lm_head.weight")[v0:v1].to(dev)',
     '        draft_head = None\n        if not _TFPP_HEAD:\n'
     '            head = make_b16(torch.zeros((1, cfg.hidden), dtype=torch.bfloat16, device=dev))\n'
     '        elif exl3:\n            raw = rd.get("lm_head.weight")[v0:v1].to(dev)'),
    # L5 (G/weights.py:485) the final norm: the last stage only
    ('t("norm.weight")', '(t("norm.weight") if _TFPP_HEAD else None)'),
)


class LoaderChanged(RuntimeError):
    pass


def patched_source(src: str) -> str:
    for old, new in ANCHORS:
        n = src.count(old)
        if n != 1:
            raise LoaderChanged(f"the MIA GLM loader changed: anchor {old.strip()[:60]!r} occurs {n} times, not once "
                                "(python -m tfpp.glm.pins --diff)")
        src = src.replace(old, new)
    return src


_SEQ = 0


def module_for(layers: list[int], *, embed: bool, head: bool, next_ok) -> types.ModuleType:
    from tensorfold.families.glm5_next.cuda import weights as base

    global _SEQ
    _SEQ += 1
    mod = types.ModuleType(f"{base.__name__}_tfpp{_SEQ}")
    mod.__package__, mod.__file__ = base.__package__, base.__file__
    # dataclasses resolve string annotations through sys.modules[cls.__module__] (Python 3.14)
    sys.modules[mod.__name__] = mod
    mod._TFPP_LAYERS = list(layers)
    mod._TFPP_NEXT = frozenset(next_ok)
    mod._TFPP_EMBED = bool(embed)
    mod._TFPP_HEAD = bool(head)
    exec(compile(patched_source(inspect.getsource(base)), base.__file__, "exec"), mod.__dict__)
    for name in CLASSES:
        setattr(mod, name, getattr(base, name))
    return mod


def drop_cached(model_dir) -> int:
    """Ask the kernel to drop the page cache it holds for this checkpoint's shards (posix_fadvise DONTNEED; no root).
    The loader reads with O_DIRECT, but a download or copy leaves the shards cached, and under WSL2 the VM's page
    cache is Windows commit: 77 GB of it plus a 57 GiB stage exhausted the commit limit and grew the pagefile until
    the system drive was full. Returns the number of files advised."""

    import os

    n = 0
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        except (OSError, AttributeError):
            pass
        finally:
            os.close(fd)
    return n


def load_range(model_dir, first: int, end: int, *, is_first: bool, is_last: bool, mtp: bool = False,
               drafter: bool = True, prefetch_next: bool = True, device: str = "cuda", world: int = 1,
               rank: int = 0):
    """A stage's weights for layers [first, end): 64 MLA / 64 KDA heads, 2,048-wide experts, the whole vocabulary
    (world=1). embed: stage 0, and the last stage when it drafts (DFlash2 or MTP); head + norm: the last stage; mtp:
    the last stage only (layer ``cfg.layers``). ``world``/``rank``: a TP share of the range (kernel_check only)."""

    from tensorfold.families.glm5_next.cuda import tp
    from tensorfold.families.glm5_next.cuda.weights import Config

    drop_cached(model_dir)
    cfg = Config.read(model_dir)
    if not 0 <= first < end <= cfg.layers:
        raise ValueError(f"layers [{first}, {end}) outside 0..{cfg.layers}")
    if mtp and not is_last:
        raise ValueError("the MTP head belongs to the last stage")
    layers = list(range(first, end))
    nxt = set(layers) | ({cfg.layers} if mtp and end == cfg.layers else set())
    mod = module_for(layers, embed=is_first or (is_last and (drafter or mtp)), head=is_last,
                     next_ok=frozenset(nxt) if prefetch_next else frozenset())
    w = mod.load(model_dir, rank=rank, world=world, device=device, mtp=mtp)
    got = [layer.index for layer in w.layers]
    if got != layers:
        raise RuntimeError(f"loaded layers {got[:3]}..{got[-3:]}, wanted {first}..{end - 1}")
    lay = tp.of(w)
    if world == 1 and (w.world, lay.heads, lay.lin_heads, lay.moe, lay.vocab) != (
            1, cfg.heads, cfg.lin_heads, cfg.moe_width, cfg.vocab):
        raise RuntimeError("the world=1 layout is not full width")
    if (w.mtp is not None) != bool(mtp and cfg.mtp_layers):
        raise RuntimeError("the MTP head was not loaded as asked")
    w.meta["pp_range"] = (first, end)
    w.meta["pp_roles"] = {"first": is_first, "last": is_last, "mtp": mtp}
    return w


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def mine(name: str, first: int, end: int, *, is_first: bool, is_last: bool, mtp: bool, drafter: bool = True,
         layers: int = 45) -> bool:
    """Whether checkpoint tensor ``name`` is read by this stage (shard lists; vision is GlmVision's own)."""

    m = _LAYER.search(name)
    if m:
        i = int(m.group(1))
        return first <= i < end or (mtp and is_last and i == layers)
    if name.startswith("model.visual."):
        return False
    if name.endswith(("embed_tokens.weight", "embed_tokens.scales", "embed_tokens.biases")):    # MLX: + scales, biases
        return is_first or (is_last and (drafter or mtp))
    if name.startswith("lm_head.") or name.endswith("language_model.norm.weight"):
        return is_last
    raise ValueError(f"unexpected checkpoint tensor {name}")


def needed_shards(model_dir, first: int, end: int, *, is_first: bool, is_last: bool, mtp: bool, drafter: bool,
                  vision: bool = False) -> tuple[list[str], list[str], list[str]]:
    """(shard files this stage reads, those missing from ``model_dir``, tensors the rules do not know)."""

    model_dir = Path(model_dir)
    raw = json.loads((model_dir / "config.json").read_text())
    text = raw.get("text_config") or raw
    layers = int(text["num_hidden_layers"])
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    need: set[str] = set()
    unknown: list[str] = []
    for name, file in index.items():
        if name.startswith("model.visual."):
            if vision and is_last:
                need.add(file)
            continue
        try:
            if mine(name, first, end, is_first=is_first, is_last=is_last, mtp=mtp, drafter=drafter, layers=layers):
                need.add(file)
        except ValueError:
            unknown.append(name)
    files = sorted(need)
    missing = [f for f in files if not (model_dir / f).is_file()]
    return files, missing, unknown


def hub_etag(model_dir, file: str) -> str | None:
    """The sha256 the Hub gave a LFS file at download (``hf download --local-dir`` keeps it in
    .cache/huggingface/download/<file>.metadata: commit, etag, time), or None."""

    meta = Path(model_dir) / ".cache" / "huggingface" / "download" / f"{file}.metadata"
    try:
        lines = meta.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    etag = lines[1].strip().strip('"') if len(lines) > 1 else ""
    return etag if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag) else None


def verify_shards(model_dir, files, cache_path=None) -> tuple[list[str], list[str]]:
    """(files whose sha256 differs from their Hub etag, files without an etag to check): this stage's shards are the
    bytes the Hub has at the revision HELLO compares (env.model_revision). Hashes are cached by (inode, size, mtime)
    in ``cache_path`` (JSON), so only new or changed files are read again (~1 min per 80 GB)."""

    import hashlib
    import os

    model_dir = Path(model_dir)
    cache: dict = {}
    if cache_path is not None:
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    bad, unchecked = [], []
    for f in files:
        want = hub_etag(model_dir, f)
        if want is None:
            unchecked.append(f)
            continue
        p = model_dir / f
        st = os.stat(p)
        key = f"{p.resolve()}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
        got = cache.get(key)
        if got is None:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(16 << 20), b""):
                    h.update(block)
            got = cache[key] = h.hexdigest()
        if got != want:
            bad.append(f)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return bad, unchecked
