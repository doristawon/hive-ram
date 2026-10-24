"""Where a DeepSeek-V4.1 chain may be cut and what crosses each cut (spec sections 5, 6.2, 6.4, 7.1; no torch).

A stage is a contiguous layer range [lo, hi). What the layers from ``lo`` on read of the layers before ``lo`` is
``carry_into(lo)``, a function of the config alone (FO/config.py:66-89):

- ``kv_layer``: the kv-source layer the first compressed layer of the suffix reads (``shared["kv_layer"]``,
  F/model.py:991, 1045; F/rounds.py:304, 323) when that layer is not a kv source itself;
- ``topk``: (p, ratio of p) when the first compressed layer of the suffix is not an index source: it reuses the
  top-k of index layer p (F/model.py:1091-1092; F/rounds.py:378);
- ``cand``: the candidate pool of the candidate-source layer, when it lies before ``lo`` and an index layer after it
  lies in the suffix (F/model.py:1082-1085; F/rounds.py:339-375);
- ``reps``: the kv-source planes the suffix attends (``sc.comp[src]``, F/model.py:1093) though their source layer
  runs before ``lo`` -- the stage after the cut keeps a replica, written with the rows the source computed; ``True``
  when the suffix also scores the source's indexer keys (``sc.index_k[src]``, F/model.py:1075-1081).

A middle stage forwards what its downstream needs, so ``carry_into`` looks at the whole suffix [lo, L), not [lo, hi).
``python -m tfpp.ds41.plan rules --config CFG`` prints every cut of a config; ``fit`` ranks three-node placements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

GIB = 2 ** 30


class PlanError(ValueError):
    """A chain the rules refuse (or a config they do not cover)."""


# -- the config, as the planner reads it ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelView:
    n_layers: int
    ratios: tuple            # compress_ratios, the DSpark blocks' included (n_layers + 3 entries)
    kv_sources: tuple
    index_sources: tuple
    cand_source: int
    engram_layers: tuple
    taps: tuple              # dspark_target_layer_ids
    window: int
    dim: int
    hc: int
    head_dim: int
    idx_dim: int
    idx_topk: int
    cand_blocks: int
    cand_block: int
    vocab: int
    dspark_block: int
    engram_rows: tuple = ()
    engram_dim: int = 0

    @classmethod
    def from_config(cls, raw: dict) -> "ModelView":
        t = raw.get("text_config", raw)
        return cls(n_layers=int(t["num_hidden_layers"]), ratios=tuple(int(r) for r in t["compress_ratios"]),
                   kv_sources=tuple(int(i) for i in t["kv_source_layer_ids"]),
                   index_sources=tuple(int(i) for i in t["index_source_layer_ids"]),
                   cand_source=int(t.get("candidate_source_layer_id", -1)),
                   engram_layers=tuple(int(i) for i in t.get("engram_layer_ids", [])),
                   taps=tuple(int(i) for i in t.get("dspark_target_layer_ids", [])),
                   window=int(t["sliding_window"]), dim=int(t["hidden_size"]), hc=int(t["hc_mult"]),
                   head_dim=int(t["head_dim"]), idx_dim=int(t["index_head_dim"]), idx_topk=int(t["index_topk"]),
                   cand_blocks=int(t.get("candidate_topk_blocks", 0)), cand_block=int(t.get("candidate_block_size", 0)),
                   vocab=int(t["vocab_size"]), dspark_block=int(t.get("dspark_block_size", 0)),
                   engram_rows=tuple(int(x) for x in t.get("engram_num_embeddings", [])),
                   engram_dim=int(t.get("engram_head_dim", 0)))

    @classmethod
    def read(cls, model_dir) -> "ModelView":
        return cls.from_config(json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8")))

    @property
    def half(self) -> int:
        """The decoder replay's first layer (F/model.py:1334)."""
        return self.n_layers // 2

    def check(self) -> None:
        """What the cut rules assume of a config (spec 5.1), checked once."""

        L = self.n_layers
        if len(self.ratios) < L:
            raise PlanError(f"compress_ratios has {len(self.ratios)} entries for {L} layers")
        if self.dspark_block and len(self.ratios) != L + 3:
            raise PlanError(f"compress_ratios needs n_layers + 3 = {L + 3} entries (the DSpark blocks' last: "
                            f"F/weights.py:467), not {len(self.ratios)}")
        if not set(self.kv_sources) <= set(self.index_sources):
            raise PlanError("every kv source must be an index source")
        if self.cand_source >= 0 and self.cand_source not in self.kv_sources:
            raise PlanError(f"the candidate source {self.cand_source} is not a kv source")
        for j in range(L):
            if self.ratios[j] and not any(s <= j for s in self.kv_sources):
                raise PlanError(f"compressed layer {j} has no kv source at or before it")
        # the prompt replay truncates h and pre at the half, not shared["topk"] / ["cand"]: the top-k, the pool and the
        # replica rows a chunk carries past the half are all computed at or after it only when the half is a kv and
        # index source and the candidate source does not precede it (the real model and S11: 20 / 5)
        h = self.half
        if h < L and self.ratios[h] and (h not in self.kv_sources or h not in self.index_sources):
            raise PlanError(f"the replay layer {h} is not a kv and index source: the carry rules do not cover it")
        if 0 <= self.cand_source < h:
            raise PlanError(f"the candidate source {self.cand_source} precedes the replay layer {h}")
        if any(s > h for s in self.kv_sources):
            raise PlanError("a kv source after the replay layer: a stage after it would need its rows of an encoder-only "
                            "chunk")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def src_of(mv: ModelView, j: int) -> int | None:
    return max(s for s in mv.kv_sources if s <= j) if mv.ratios[j] else None


def first_c(mv: ModelView, lo: int) -> int | None:
    return next((j for j in range(lo, mv.n_layers) if mv.ratios[j]), None)


@dataclass(frozen=True)
class Carry:
    """What crosses the cut before layer ``lo`` (spec 5.1)."""

    lo: int
    first_c: int | None
    kv_layer: int | None
    topk: tuple | None                       # (index layer p, its ratio)
    cand: bool
    reps: tuple = ()                         # ((source, with_index), ...) sorted

    @property
    def rep_map(self) -> dict:
        return dict(self.reps)

    def describe(self) -> str:
        r = ",".join(f"{s}{'+idx' if v else ''}" for s, v in self.reps) or "-"
        return (f"kv_layer={self.kv_layer} topk={self.topk} cand={'yes' if self.cand else '-'} reps={r}")


def carry_into(mv: ModelView, lo: int) -> Carry | None:
    """None for lo = 0 (stage 0 embeds) and lo = L (nothing after the head)."""

    L = mv.n_layers
    if lo <= 0 or lo >= L:
        return None
    fc = first_c(mv, lo)
    kv_layer = None if fc is None or fc in mv.kv_sources else src_of(mv, fc)
    topk = None
    if fc is not None and fc not in mv.index_sources:
        p = max(i for i in mv.index_sources if i < fc)
        topk = (p, mv.ratios[p])
    cand = 0 <= mv.cand_source < lo and any(lo <= j < L and j > mv.cand_source for j in mv.index_sources)
    reps: dict[int, bool] = {}
    for j in range(lo, L):
        s = src_of(mv, j)
        if s is not None and s < lo:
            with_index = j in mv.index_sources and j not in mv.kv_sources and src_of(mv, j) == s
            reps[s] = reps.get(s, False) or with_index
    return Carry(lo=lo, first_c=fc, kv_layer=kv_layer, topk=topk, cand=bool(cand),
                 reps=tuple(sorted(reps.items())))


# -- a stage -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StagePlan:
    index: int
    count: int
    lo: int
    hi: int
    n_layers: int
    cin: Carry | None
    cout: Carry | None
    own_sources: tuple
    own_raw: tuple
    planes: tuple
    index_planes: tuple
    engram_owned: tuple
    half: int
    up_taps: int                     # DSpark taps computed before this stage (only --no-drafts allows them)
    drafts: bool

    @property
    def first(self) -> bool:
        return self.lo == 0

    @property
    def last(self) -> bool:
        return self.hi == self.n_layers

    @property
    def after_replay(self) -> bool:
        return self.lo > self.half

    @property
    def layers(self) -> tuple:
        return (self.lo, self.hi)

    def summary(self) -> str:
        return (f"stage {self.index} [{self.lo},{self.hi}) planes {list(self.planes)} (indexer {list(self.index_planes)})"
                f" engram {list(self.engram_owned)}; in: {self.cin.describe() if self.cin else 'embed'}; "
                f"out: {self.cout.describe() if self.cout else 'head'}")


def stage_plan(mv: ModelView, index: int, ranges: Sequence[Sequence[int]], *, drafts: bool) -> StagePlan:
    lo, hi = (int(x) for x in ranges[index])
    nxt = int(ranges[index + 1][0]) if index + 1 < len(ranges) else mv.n_layers
    cin, cout = carry_into(mv, lo), carry_into(mv, nxt)
    own = tuple(s for s in mv.kv_sources if lo <= s < hi)
    own_raw = tuple(s for s in own if mv.ratios[s] > 1)
    rep_in = cin.rep_map if cin is not None else {}
    planes = tuple(sorted(set(own) | set(rep_in)))
    index_planes = tuple(sorted({s for s in own if s in mv.index_sources} | {s for s, v in rep_in.items() if v}))
    P = StagePlan(index=index, count=len(ranges), lo=lo, hi=hi, n_layers=mv.n_layers, cin=cin, cout=cout,
                  own_sources=own, own_raw=own_raw, planes=planes, index_planes=index_planes,
                  engram_owned=tuple(e for e in mv.engram_layers if lo <= e < hi), half=mv.half,
                  up_taps=sum(1 for t in mv.taps if t < lo), drafts=bool(drafts))
    if cout is not None:                               # what this stage sends on, it holds (spec 5.1)
        for s, with_index in cout.reps:
            if s not in P.planes:
                raise PlanError(f"stage [{lo},{hi}) forwards plane {s} it does not hold")
            if with_index and s not in P.index_planes:
                raise PlanError(f"stage [{lo},{hi}) forwards the indexer keys of {s} it does not hold")
    if P.after_replay and own:
        raise PlanError(f"stage [{lo},{hi}) after the replay layer owns kv sources {list(own)}")
    return P


def plan_chain(mv: ModelView, ranges: Sequence[Sequence[int]], *, drafts: bool) -> list[StagePlan]:
    """Every stage's plan; refuses a chain the rules do not allow (spec 5.2 items 1-2)."""

    mv.check()
    if not ranges:
        raise PlanError("no stages")
    at = 0
    for i, (a, b) in enumerate(ranges):
        a, b = int(a), int(b)
        if a != at:
            raise PlanError(f"stage {i} starts at layer {a}, expected {at} (contiguous from 0)")
        if b <= a:
            raise PlanError(f"stage {i}: an empty range [{a}, {b})")
        at = b
    if at != mv.n_layers:
        raise PlanError(f"the chain ends at layer {at}, the model has {mv.n_layers}")
    leader_lo = int(ranges[-1][0])
    if drafts:
        if mv.dspark_block and len(mv.taps) != 3:
            raise PlanError(f"DSpark's main_proj takes 3 taps, the config names {len(mv.taps)}")
        if mv.taps and leader_lo > min(mv.taps):
            raise PlanError(f"with drafts the leader must hold every DSpark tap layer {list(mv.taps)}: its range "
                            f"starts at {leader_lo} (cut at {min(mv.taps)} or before, or --no-drafts)")
    return [stage_plan(mv, i, ranges, drafts=drafts) for i in range(len(ranges))]


def chain_digest(plans: Sequence[StagePlan]) -> str:
    return hashlib.sha256(json.dumps([asdict(p) for p in plans], sort_keys=True, default=list).encode()).hexdigest()[:16]


# -- prompt chunks (spec 6.2) ----------------------------------------------------------------------------------------
def chunk_in(P: StagePlan, start: int, n: int, replay: int | None) -> tuple[str, int, int, int]:
    """(kind, start, rows, floor) of the ACT entering stage P for the chunk [start, start + n): "chunk" with the rows
    the stage computes (a stage after the replay layer gets the chunk already truncated at the replay row), or "kv"
    (the chunk is encoder-only there: the replica rows alone). Every stage computes it alike from the prompt."""

    if replay is None or not P.after_replay:
        return "chunk", start, n, 0
    first = max(start, replay) - start                     # F/model.py:1343
    if first >= n:
        return "kv", start + n, 0, replay
    return "chunk", start + first, n - first, replay


def replay_row(prompt_len: int, window: int, replay_on: bool) -> int | None:
    """F/engine.py:394."""
    return max(0, prompt_len - window) if replay_on else None


# -- ACT parts (spec 6.4, 7.1, 9.3) ------------------------------------------------------------------------------------
def _ceil(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def groups(start0: int, n0: int, ratio: int) -> tuple[int, int]:
    """The compressed rows a chunk [start0, start0 + n0) writes of a kv source (F/model.py:883-900)."""
    return start0 // ratio, (start0 + n0) // ratio


def rep_parts(mv: ModelView, s: int, with_index: bool, rows: int) -> list[tuple[str, str, list[int]]]:
    out = [(f"c{s}", "u8", [rows, mv.head_dim // 2]), (f"cs{s}", "u8", [rows, mv.head_dim // 16])]
    if with_index:
        out += [(f"k{s}", "u8", [rows, mv.idx_dim // 2]), (f"ks{s}", "u8", [rows, mv.idx_dim // 32])]
    return out


def f1_parts(mv: ModelView, rows: int) -> list[tuple[str, str, list[int]]]:
    return [("pend", "f32", [1, rows, mv.dim]), ("post", "f32", [rows, mv.hc]), ("comb", "f32", [rows, mv.hc, mv.hc])]


def topk_width(mv: ModelView, carry: Carry, end: int) -> int:
    return min(mv.idx_topk, int(end) // carry.topk[1])


def cand_width(mv: ModelView, entries: int) -> int:
    return _ceil(entries, mv.cand_block) if mv.cand_block else 0


def chunk_parts(mv: ModelView, carry: Carry, start0: int, n0: int, s2: int, n2: int, *, f1: bool = False):
    """A prompt chunk's ACT into the stage after ``carry``'s cut: h and pre of the rows it computes [s2, s2 + n2), what
    the suffix reads of earlier layers, the replica rows of the whole chunk."""

    e = s2 + n2
    out = [("h", "bf16", [n2, mv.hc, mv.dim]), ("pre", "f32", [n2, mv.hc])]
    if carry.topk is not None:
        out.append(("topk", "i64", [n2, topk_width(mv, carry, e)]))
    if carry.cand:
        out.append(("cand", "u8", [n2, cand_width(mv, e // mv.ratios[mv.cand_source])]))
    for s, with_index in carry.reps:
        g0, g1 = groups(start0, n0, mv.ratios[s])
        out += rep_parts(mv, s, with_index, g1 - g0)
    if f1:
        out += f1_parts(mv, n2)
    return out


def kv_parts(mv: ModelView, carry: Carry, start0: int, n0: int):
    """An encoder-only chunk's ACT (after the replay layer): the replica rows alone."""

    out = []
    for s, with_index in carry.reps:
        g0, g1 = groups(start0, n0, mv.ratios[s])
        out += rep_parts(mv, s, with_index, g1 - g0)
    return out


def cand_form(mv: ModelView, bucket: int, *, idx_on: bool, cand_only_on: bool) -> str:
    """Which form the candidate pool takes in a round of this bucket: "cblk" (block ids) or "cand" (block mask), as
    the candidate-source layer picks it (F/rounds.py:339-375)."""

    nb = bucket // mv.ratios[mv.cand_source]
    kk = min(mv.idx_topk, nb)
    if idx_on and kk > 0 and kk & (kk - 1) == 0 and cand_only_on and nb > mv.cand_blocks * mv.cand_block:
        return "cblk"
    return "cand"


def round_parts(mv: ModelView, carry: Carry, R: int, bucket: int, form: str, *, f1: bool = False):
    """A decode round's ACT: one row a window row (spec 9.3)."""

    out = [("h", "bf16", [R, mv.hc, mv.dim]), ("pre", "f32", [R, mv.hc])]
    if carry.topk is not None:
        out.append(("topk", "i64", [R, min(mv.idx_topk, bucket // carry.topk[1])]))
    if carry.cand:
        nbk = cand_width(mv, bucket // mv.ratios[mv.cand_source])
        out.append(("cblk", "i32", [R, min(mv.cand_blocks, nbk)]) if form == "cblk" else ("cand", "u8", [R, nbk]))
    for s, with_index in carry.reps:
        out += rep_parts(mv, s, with_index, R)
    if f1:
        out += f1_parts(mv, R)
    return out


def buckets(cap: int, bucket_min: int = 1024) -> list[int]:
    """Every bucket graph.bucket_for can return for a pool of ``cap`` positions (F/graph.py:24-28)."""

    out, b = set(), bucket_min
    top = max(cap, bucket_min)
    while b < top:
        out.add(min(b, top))
        b *= 2
    out.add(top)
    return sorted(out)


DT_BYTES = {"bf16": 2, "f32": 4, "i64": 8, "i32": 4, "u8": 1}


def parts_bytes(parts) -> int:
    return sum(math.prod(s) * DT_BYTES[d] for _, d, s in parts)


# -- the cut table (spec 5.3 / 5.4) ------------------------------------------------------------------------------------
def classify(mv: ModelView, c: int, *, drafts: bool = True) -> str:
    cls = []
    if c in mv.engram_layers:
        cls.append("before-engram(natural)")
    if c in mv.kv_sources:
        cls.append("group-boundary")
    elif mv.ratios[c] and mv.ratios[c - 1]:
        cls.append(f"inside-group(src {src_of(mv, c)}, r{mv.ratios[c]})")
    if c - 1 in mv.engram_layers:
        cls.append("after-engram")
    if c == mv.half:
        cls.append("replay-layer")
    elif c > mv.half:
        cls.append("after-replay")
    if drafts and mv.taps and c > min(mv.taps):
        cls.append("REFUSED(splits the taps)")
    return " ".join(cls)


def rules_table(mv: ModelView) -> list[str]:
    out = [f"n_layers {mv.n_layers} kv {list(mv.kv_sources)} index {list(mv.index_sources)} cand {mv.cand_source} "
           f"engram {list(mv.engram_layers)} taps {list(mv.taps)} replay layer {mv.half}"]
    for c in range(1, mv.n_layers):
        k = carry_into(mv, c)
        out.append(f"  cut before {c:2d}: {classify(mv, c):60s} {k.describe()}")
    return out


# -- three-node placements (spec 16.1) ---------------------------------------------------------------------------------
def stage_weight_bytes(per: dict, lo: int, hi: int, *, first: bool, last: bool, drafter: bool, vision: bool) -> int:
    """Weights of [lo, hi) from per-layer byte counts (wf/hf/per_mia.json: "L00".., "top:embed", "top:head", ...)."""

    n = sum(int(per[f"L{i:02d}"]) for i in range(lo, hi))
    if first or (last and drafter):
        n += int(per.get("top:embed", 0))
    if last:
        n += int(per.get("top:norm", 0)) + int(per.get("top:head", 0))
        if drafter:
            n += int(per.get("top:mtp", 0))
        if vision:
            n += sum(int(per.get(k, 0)) for k in ("top:vision", "top:aligner", "top:image_start", "top:image_end",
                                                   "top:image_newline"))
    return n


def fit(mv: ModelView, per: dict, budgets_gib: Sequence[float], *, extra_gib: Sequence[float] = (),
        disk_gib: Sequence[float] = (), engram_gib: float = 94.56, drafter: bool = True, vision: bool = False,
        top: int = 12) -> list[dict]:
    """Every (c1, c2) of a three-stage chain, the ones whose weights + ``extra_gib`` (work space and caches) fit each
    stage's measured budget, best (the largest smallest slack) first. ``disk_gib``: each node's free ext4 space for
    its shards and the Engram tables it owns (one table each, ``engram_gib``)."""

    L = mv.n_layers
    out = []
    extra = list(extra_gib) or [0.0] * len(budgets_gib)
    for c1 in range(1, L - 1):
        for c2 in range(c1 + 1, L):
            ranges = [(0, c1), (c1, c2), (c2, L)]
            try:
                plan_chain(mv, ranges, drafts=drafter)
            except PlanError:
                continue
            w = [stage_weight_bytes(per, a, b, first=a == 0, last=b == L, drafter=drafter, vision=vision) / GIB
                 for a, b in ranges]
            slack = [budgets_gib[i] - w[i] - extra[i] for i in range(3)]
            if min(slack) < 0:
                continue
            rec = {"cuts": [c1, c2], "weights_gib": [round(x, 2) for x in w], "slack_gib": [round(x, 2) for x in slack]}
            if disk_gib:
                need = [w[i] + engram_gib * sum(1 for e in mv.engram_layers if ranges[i][0] <= e < ranges[i][1])
                        for i in range(3)]
                rec["disk_gib"] = [round(x, 1) for x in need]
                if any(need[i] > disk_gib[i] for i in range(3)):
                    continue
            rec["classes"] = [classify(mv, c1, drafts=drafter), classify(mv, c2, drafts=drafter)]
            out.append(rec)
    out.sort(key=lambda r: -min(r["slack_gib"]))
    return out[:top]


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.ds41.plan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("rules", "fit", "chain"))
    ap.add_argument("--config", required=True, help="config.json (or a model folder)")
    ap.add_argument("--ranges", default="", help="chain: A:B,B:C,...")
    ap.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--bytes", default="", help="fit: per-layer bytes JSON (wf/hf/per_mia.json)")
    ap.add_argument("--budgets", default="", help="fit: each node's measured budget, GiB (stage 0, 1, leader)")
    ap.add_argument("--extra", default="", help="fit: each stage's work space + caches beside its weights, GiB")
    ap.add_argument("--disk", default="", help="fit: each node's free ext4 GiB")
    ap.add_argument("--vision", action="store_true")
    a = ap.parse_args(argv)
    p = Path(a.config)
    mv = ModelView.read(p) if p.is_dir() else ModelView.from_config(json.loads(p.read_text(encoding="utf-8")))
    if a.what == "rules":
        print("\n".join(rules_table(mv)))
        return 0
    if a.what == "chain":
        ranges = [tuple(int(x) for x in r.split(":")) for r in a.ranges.split(",") if r]
        try:
            plans = plan_chain(mv, ranges, drafts=not a.no_drafts)
        except PlanError as exc:
            print(f"refused: {exc}")
            return 1
        for P in plans:
            print(P.summary())
        print(f"plan digest {chain_digest(plans)}")
        return 0
    per = json.loads(Path(a.bytes).read_text(encoding="utf-8"))
    for rec in fit(mv, per, _floats(a.budgets), extra_gib=_floats(a.extra), disk_gib=_floats(a.disk),
                   drafter=not a.no_drafts, vision=a.vision):
        print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
