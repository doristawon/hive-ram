"""One stage's weights with the fork's own block loader at world=1 (spec section 4), and the test model's selection.

``load_stage`` calls the fork's public ``load_block`` for each layer of the stage (as T/forward_proxy.py:148-166 does)
and copies ``_load``'s few other lines (F/weights.py:451-477): no copy of weights.py is compiled. At world=1 every TP
slice ``load_block`` takes is the whole axis (64 heads, 8 output groups, 2,304-wide experts, the 24 Engram hash
columns: F/weights.py:357-379, 394, 413-416), so a stage's layer holds exactly the bits the one-process reference's
does. ``w.layers`` is a ``StageLayers``: ``len()`` is the model's layer count and ``[i]`` takes a global layer number,
iteration gives the stage's own layers only (spec 4.1).

``AliasShards`` reads the checkpoint names through a test selection's renaming (tfpp_selection.json: the shortened
model's layer ``k`` is the checkpoint's ``layers.<orig[k]>``); without that file names are as they are. The rank cache
is never used (F/weights.py:119-148 keys it by rank / world, sizes it by the whole checkpoint, deletes its siblings).

``load_full`` has ``weights.load``'s signature: the reference (ref.py) swaps the module attribute for it, so the
unmodified ``DsEngine`` reads a test selection too. ``TFPP_DS41_EXPERTS=K`` (T-X0's proxy): every MoE keeps its first K
routed experts and the shared one, the gate cut to them (T/forward_proxy.py:159-162).

The selection helpers, ``mine`` and ``needed_shards`` run without torch.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

SELECTION_FILE = "tfpp_selection.json"
S11 = (0, 1, 2, 3, 14, 20, 21, 24, 37, 38, 39)


class StageLayers:
    """A stage's layers under global layer numbers: len() == n_layers, iteration gives the own layers, [i] takes a
    global number in [lo, hi) and raises IndexError for any other (spec 4.1)."""

    def __init__(self, lo: int, hi: int, n: int, own) -> None:
        self.lo, self.hi, self.n, self.own = int(lo), int(hi), int(n), list(own)
        if len(self.own) != self.hi - self.lo:
            raise ValueError(f"{len(self.own)} layers for the range [{lo}, {hi})")

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        return iter(self.own)

    def __getitem__(self, i):
        if not isinstance(i, int) or isinstance(i, bool) or not self.lo <= i < self.hi:
            raise IndexError(f"layer {i!r} is not on this stage [{self.lo}, {self.hi})")
        return self.own[i - self.lo]

    def __repr__(self) -> str:
        return f"StageLayers([{self.lo}, {self.hi}) of {self.n})"


# -- which checkpoint tensors a stage reads (shard lists) --------------------------------------------------------------
_LAYER = re.compile(r"^layers\.(\d+)\.")
_VISION = ("vision.", "aligner.")
_IMAGE = ("image_start", "image_end", "image_newline")


def mine(name: str, layers, *, first: bool, last: bool, drafter: bool, vision: bool = False) -> bool:
    """Whether checkpoint tensor ``name`` (stage numbering) is read by a stage holding ``layers``."""

    m = _LAYER.match(name)
    if m:
        if int(m.group(1)) not in layers:
            return False
        return vision or not name.endswith(".ffn.gate.bias_vl")          # attach_vl_bias: --vision only
    if name.startswith("mtp."):
        return last and drafter
    if name == "embed.weight":
        return first or (last and drafter)                               # the drafter reads m.w.embed (F/dspark.py:157)
    if name.startswith("head.") or name == "norm.weight":
        return last
    if name.startswith(_VISION) or name in _IMAGE:
        return last and vision                                           # v2: the leader holds the tower
    raise ValueError(f"unexpected checkpoint tensor {name}")


def needed_shards(weight_map: dict, layers, *, first: bool, last: bool, drafter: bool, vision: bool = False,
                  rename=None) -> tuple[list[str], list[str]]:
    """(shard files a stage reads, tensors the rules do not know); ``weight_map`` from model.safetensors.index.json
    (checkpoint name -> file); ``rename``: a selection's namer (checkpoint name -> stage name, '~' prefix for an
    unselected layer)."""

    layers = set(layers)
    need, unknown = set(), []
    for name, file in weight_map.items():
        n = rename(name) if rename else name
        if n.startswith("~"):
            continue
        try:
            if mine(n, layers, first=first, last=last, drafter=drafter, vision=vision):
                need.add(file)
        except ValueError:
            unknown.append(name)
    return sorted(need), unknown


def embed_only_file(model_dir, weight_map: dict) -> str | None:
    """A single-tensor file in ``model_dir`` that stands in for the shard holding ``embed.weight`` (``testmodel extract``:
    a drafting leader needs the 1.3 GB embedding, not the 6.5 GB shard with layer 0 around it). A file counts when the
    index does not name it and its header holds ``embed.weight``; ``AliasShards`` reads it like any shard and refuses the
    chain's start when the real shard sits beside it (one name, one file)."""

    known = set(weight_map.values())
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        if path.name in known:
            continue
        try:
            with open(path, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
        except (OSError, ValueError, struct.error):
            continue
        if "embed.weight" in header:
            return path.name
    return None


def stage_files(model_dir, weight_map: dict, layers, *, first: bool, last: bool, drafter: bool, vision: bool = False,
                rename=None) -> tuple[list[str], list[str]]:
    """``needed_shards`` for a stage that runs here: the shard that holds only the stage's ``embed.weight`` (and no other
    tensor it reads) is replaced by an extracted embedding file in ``model_dir`` when there is one (the preflight then
    asks for the file that is on the disk)."""

    files, unknown = needed_shards(weight_map, layers, first=first, last=last, drafter=drafter, vision=vision,
                                   rename=rename)
    holder = weight_map.get("embed.weight")
    if holder in files:
        rest = {k: v for k, v in weight_map.items() if k != "embed.weight"}
        others, _ = needed_shards(rest, layers, first=first, last=last, drafter=drafter, vision=vision, rename=rename)
        alt = embed_only_file(model_dir, weight_map) if holder not in others else None
        if alt is not None:
            files = sorted((set(files) - {holder}) | {alt})
    return files, unknown


# -- a test-only layer SELECTION, renumbered 0..n-1 ---------------------------------------------------------------------
def selection_namer(sel):
    """Checkpoint name -> the renumbered model's name: layers.<orig>. -> layers.<new>.; an unselected layer's names get
    a '~' prefix (never requested, never shadowing a renumbered name)."""

    new = {int(o): n for n, o in enumerate(sel)}

    def name(n: str) -> str:
        m = _LAYER.match(n)
        if not m:
            return n
        o = int(m.group(1))
        return f"layers.{new[o]}." + n[m.end():] if o in new else "~" + n

    return name


def read_selection(model_dir) -> dict | None:
    p = Path(model_dir) / SELECTION_FILE
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    sel = [int(x) for x in d["orig_layers"]]
    if sorted(set(sel)) != sel:
        raise ValueError(f"{p}: orig_layers must be strictly increasing")
    return d


def namer_for(model_dir):
    d = read_selection(model_dir)
    return None if d is None else selection_namer(d["orig_layers"])


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def engram_rows(vocab: int, layers: int, ngram: int, heads: int) -> list[int]:
    """FO/ops.py:261-277's primes summed a layer: the table rows EngramHasher asserts (FO/ops.py:379-380). Primes are
    drawn in order and never reused, so they depend on the vocabulary, n-gram size, heads and Engram layer count only,
    never on the layer numbers."""

    seen, out = set(), []
    for _ in range(layers):
        total = 0
        for _ in range(ngram - 1):
            cur = vocab - 1
            for _ in range(heads):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                total += cur
        out.append(total)
    return out


def selection_config(raw: dict, sel, *, engram_vocab: int | None = None) -> dict:
    """config.json of the renumbered selection: every per-layer array remapped consistently (FO/config.py:66-89 reads
    them; spec 15.1). Checks: a selected compressed layer's kv source is selected (its planes are the source's), the
    DSpark taps stay three."""

    raw = json.loads(json.dumps(raw))
    t = raw.get("text_config", raw)
    L, ratios = int(t["num_hidden_layers"]), list(t["compress_ratios"])
    sel = [int(x) for x in sel]
    if not sel or sorted(set(sel)) != sel or sel[-1] >= L:
        raise ValueError("a selection is strictly increasing layer ids below num_hidden_layers")
    new = {o: n for n, o in enumerate(sel)}
    kvs = list(t["kv_source_layer_ids"])
    for o in sel:
        if ratios[o]:
            src = max(s for s in kvs if s <= o)
            if src not in new:
                raise ValueError(f"layer {o} reads kv source {src}, which the selection leaves out")

    def keep(ids):
        return [new[i] for i in ids if i in new]

    t["num_hidden_layers"] = len(sel)
    t["compress_ratios"] = [ratios[o] for o in sel] + ratios[L:]          # + the DSpark blocks' (mtp.0-2)
    t["kv_source_layer_ids"] = keep(kvs)
    t["index_source_layer_ids"] = keep(t["index_source_layer_ids"])
    cs = int(t.get("candidate_source_layer_id", -1))
    if cs >= 0 and cs not in new:
        raise ValueError(f"the candidate source {cs} is not selected")
    t["candidate_source_layer_id"] = new.get(cs, -1)
    eng = keep(t.get("engram_layer_ids", []))
    t["engram_layer_ids"] = eng
    if engram_vocab is not None:
        t["engram_vocab_size"] = int(engram_vocab)
    if eng:
        t["engram_num_embeddings"] = engram_rows(int(t["engram_vocab_size"]), len(eng), int(t["engram_max_ngram_size"]),
                                                 int(t["engram_n_heads"]))
    else:
        t["engram_num_embeddings"] = []
    taps = keep(t.get("dspark_target_layer_ids", []))
    if len(taps) != len(t.get("dspark_target_layer_ids", [])):
        raise ValueError("DSpark's main_proj takes every tap: the selection must keep the tap layers")
    t["dspark_target_layer_ids"] = taps
    return raw


def selection_record(sel, *, engram_vocab: int | None, revision: str) -> dict:
    """tfpp_selection.json: the same bytes on every node (HELLO compares its sha: no paths in it)."""

    return {"orig_layers": [int(x) for x in sel], "engram_vocab": engram_vocab, "revision": revision}


# -- loading (torch from here on) ---------------------------------------------------------------------------------------
_ALIAS = None


def AliasShards(root, namer=None):                                      # noqa: N802 (a class, made on first use)
    global _ALIAS
    if _ALIAS is None:
        from tensorfold.families.deepseek_v41.cuda.weights import Shards

        class _AliasShards(Shards):
            """weights.Shards (F/weights.py:29-84) whose names pass through ``namer`` (a selection's renaming): its
            header scan is copied so a name two shards both hold is refused, not silently the later one's
            (F/weights.py:38-44)."""

            def __init__(self, root, namer=None):                       # noqa: D401
                self.root = Path(root)
                self.entries = {}
                seen = {}
                for path in sorted(self.root.glob("*.safetensors")):
                    with open(path, "rb") as f:
                        size = struct.unpack("<Q", f.read(8))[0]
                        header = json.loads(f.read(size))
                    header.pop("__metadata__", None)
                    for name, e in header.items():
                        n = namer(name) if namer else name
                        if n.startswith("~"):
                            continue
                        if n in self.entries:
                            raise ValueError(f"tensor {n} is in both {seen[n]} and {path.name} (one name, one file: "
                                             "move the extra shard out of the model folder)")
                        self.entries[n] = (path, 8 + size, e)
                        seen[n] = path.name
                self._files = {}

        _ALIAS = _AliasShards
    return _ALIAS(root, namer)


def _cut_gate(lay, k: int) -> None:
    lay.gate_w = lay.gate_w[:k].contiguous()                  # the gate picks among the loaded experts
    lay.gate_b = lay.gate_b[:k].contiguous()


def _build(model_dir, lo: int, hi: int, *, embed: bool, head: bool, dspark: bool, experts: int, stage: bool, log):
    """Weights of layers [lo, hi) as F/weights.py:442-487 builds them at rank 0 of world 1."""

    import time

    import torch

    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import weights as W

    if os.environ.get("TF_DS_RANK_CACHE"):
        raise RuntimeError("tfpp-ds41: unset TF_DS_RANK_CACHE (keyed by rank / world, sized by the whole checkpoint: "
                           "F/weights.py:119-148)")
    t0 = time.time()
    model_dir = Path(model_dir)
    cfg = Cfg.read(model_dir)
    sh = AliasShards(model_dir, namer_for(model_dir))            # no sh.cache: _kept reads the shards (F/weights.py:238)
    try:
        def plain(name: str, dtype=None) -> torch.Tensor:      # F/weights.py:451-453
            return W._kept(sh, f"{name}|{dtype}", lambda: (sh.get(name).to(dtype) if dtype is not None
                                                            else sh.get(name)).contiguous()).cuda()

        w = W.Weights(cfg, 0, 1, embed=plain("embed.weight") if embed else None,
                      norm=plain("norm.weight") if head else None,
                      head=W.linear(sh, "head", cols=(0, cfg.vocab)) if head else None,
                      vocab_lo=0, vocab_hi=cfg.vocab, model_dir=model_dir)
        own = []
        for i in range(lo, hi):
            lay = W.load_block(sh, cfg, f"layers.{i}", i, 0, 1, experts or cfg.n_routed)
            if experts:
                _cut_gate(lay, experts)
            own.append(lay)
            if i % 5 == 4 or i == hi - 1:
                log(f"[tfpp-ds41] layers {lo}..{i} loaded ({time.time() - t0:.0f} s, "
                    f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
        w.layers = StageLayers(lo, hi, cfg.n_layers, own) if stage else own
        if dspark and cfg.dspark_block:                            # F/weights.py:466-477
            n = len(cfg.compress_ratios) - cfg.n_layers
            last = f"mtp.{n - 1}"
            nd = min(experts, cfg.dspark_routed) if experts else cfg.dspark_routed
            blocks = []
            for j in range(n):
                b = W.load_block(sh, cfg, f"mtp.{j}", cfg.n_layers + j, 0, 1, nd)
                if experts and nd < cfg.dspark_routed:
                    _cut_gate(b, nd)
                blocks.append(b)
            w.dspark = W.DSparkWeights(blocks, W.linear(sh, "mtp.0.main_proj"), plain("mtp.0.main_norm.weight"),
                                       plain(f"{last}.norm.weight"), plain(f"{last}.markov_head.embed.weight"),
                                       plain(f"{last}.markov_head.head.weight"),
                                       plain(f"{last}.confidence_head.proj.weight"))
            log(f"[tfpp-ds41] DSpark ({n} stages) loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    finally:
        sh.close()
        W._trim()
    return w


def _check(w, lo: int, hi: int, *, embed: bool, head: bool, dspark: bool) -> None:
    c = w.cfg
    got = [lay.idx for lay in w.layers]
    if got != list(range(lo, hi)):
        raise RuntimeError(f"loaded layers {got}, wanted [{lo}, {hi})")
    if (w.embed is not None) != embed or (w.head is not None) != head or (w.norm is not None) != head:
        raise RuntimeError("embed / head / norm not loaded as the stage's role asks")
    if (w.dspark is not None) != bool(dspark and c.dspark_block):
        raise RuntimeError("the DSpark blocks not loaded as asked")
    for lay in w.layers:                                       # world=1: full width (F/weights.py:357-379)
        if lay.sink.numel() != c.n_heads or len(lay.wo_a) != c.o_groups:
            raise RuntimeError(f"layer {lay.idx}: not the world=1 full-width layout")
    if head and (w.vocab_lo, w.vocab_hi) != (0, c.vocab):
        raise RuntimeError("the head is not the whole vocabulary")


def load_stage(model_dir, plan, *, drafter: bool, experts: int = 0, log=print):
    """A stage's Weights (spec 4.3): its layers; the embedding on stage 0 and on a drafting leader (F/dspark.py:157,
    210, 372 read m.w.embed); norm, head and DSpark on the leader."""

    embed = plan.first or (plan.last and drafter)
    w = _build(model_dir, plan.lo, plan.hi, embed=embed, head=plan.last, dspark=plan.last and drafter,
               experts=experts, stage=True, log=log)
    _check(w, plan.lo, plan.hi, embed=embed, head=plan.last, dspark=plan.last and drafter)
    return w


def load_full(model_dir, rank: int, world: int, n_layers: int | None = None, log=print, dspark: bool = False):
    """``weights.load``'s signature (F/weights.py:422-423) for the reference (spec 3.5): one stage holding every layer,
    ``w.layers`` a plain list as the fork's. Reads a test selection's names; TFPP_DS41_EXPERTS=K cuts the experts."""

    if (int(rank), int(world)) != (0, 1):
        raise RuntimeError(f"tfpp-ds41's reference loads world=1 rank 0, not rank {rank} of {world}")
    from tensorfold.families.deepseek_v41.config import Cfg

    L = Cfg.read(model_dir).n_layers if n_layers is None else int(n_layers)
    k = int(os.environ.get("TFPP_DS41_EXPERTS") or 0)
    w = _build(model_dir, 0, L, embed=True, head=True, dspark=dspark, experts=k, stage=False, log=log)
    _check(w, 0, L, embed=True, head=True, dspark=dspark)
    return w


def token_map(model_dir, cfg) -> tuple[list[int], str]:
    """The Engram token map (F/engine.py:69-80): TF_DS_TOKEN_MAP's cache, else built from tokenizer.json (its size
    must be the config's compressed vocabulary). Returns (map, sha256 of its JSON)."""

    import hashlib

    from tensorfold.families.deepseek_v41.ops import compressed_token_map

    cache = Path(os.environ.get("TF_DS_TOKEN_MAP") or (Path.home() / ".cache" / "dsv41_token_map.json"))
    tm = None
    if cache.exists():
        try:
            tm = json.loads(cache.read_text())
        except (OSError, ValueError):
            tm = None                       # unreadable or half written by a stage on this node: built again below
    if tm is not None:
        if not tm or max(tm) + 1 != cfg.engram_cvocab:
            raise RuntimeError(f"{cache}: a token map of {max(tm) + 1 if tm else 0} compressed ids, the config says "
                               f"{cfg.engram_cvocab} (delete the cache, it is rebuilt from tokenizer.json)")
    else:
        tm, n = compressed_token_map(Path(model_dir) / "tokenizer.json")
        if n != cfg.engram_cvocab:
            raise RuntimeError(f"the token map has {n} compressed ids, the config says {cfg.engram_cvocab}")
        try:                                # T0c / T1-3 put several stages on one node: a whole file or none (os.replace)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(tm))
            os.replace(tmp, cache)
        except OSError:
            pass
    return tm, hashlib.sha256(json.dumps(tm).encode()).hexdigest()
