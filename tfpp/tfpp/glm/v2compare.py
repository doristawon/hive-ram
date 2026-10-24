"""Compare tfpp-glm v2 traces and bench outputs (spec v2 section 15.2; no torch).

  python -m tfpp.glm.v2compare strict  REF.jsonl GOT.jsonl [--ignore-taps] [--show N]
  python -m tfpp.glm.v2compare stages  LEADER.jsonl FOLLOWER.jsonl [FOLLOWER.jsonl ...]
  python -m tfpp.glm.v2compare streams REF.jsonl GOT.jsonl [--rows]
  python -m tfpp.glm.v2compare tokens  REF.jsonl GOT.jsonl
  python -m tfpp.glm.v2compare bench   P1.json V2.json [--prefix ID,ID]

``strict``: the leaders of two runs of the same lockstep schedule on the same pool plan (a one-process reference and a
chain, or two cuts): the records the worker thread wrote must be equal in order (messages, admissions, chunks, windows
with every logits row's hash, keeps, pools), the client records equal tag by tag. Timings are not compared. A leader
trace with other pinned parameters (``meta2``) is refused (exit 2). ``stages``: inside one run, every stage applied
the leader's messages (sequence number, checksum, ops) and held the same pool at every idle point. ``streams``: each
request's reply by tag, and with ``--rows`` every committed row (position, token, logits hash) by tag: valid across
schedules. ``tokens``: the replies by tag only. ``bench``: a v1 ``--bench-out`` against a v2 one, by (id, rep).
Exit 0: identical; 1: they differ; 2: not comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKER_KINDS = ("meta2", "iter", "msg", "admit", "fill", "pieces", "round", "keeps", "pool", "idle")
CLIENT_KINDS = ("req", "done", "fail")
SKIP_KEYS = {"wall_ms", "ms", "time", "prefill_s", "decode_s", "stages_ms", "encode_s"}
BULKY = ("logits", "taps", "tokens", "heads", "pool", "extents", "kept", "lanes")


def load(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                raise SystemExit(f"{path}:{i}: not JSON") from None
    return out


def clean(r: dict, ignore_taps: bool = False) -> dict:
    r = {k: v for k, v in r.items() if k not in SKIP_KEYS}
    if ignore_taps:
        r.pop("taps", None)
    if r.get("t") == "done" and isinstance(r.get("stats"), dict):
        r["stats"] = {k: v for k, v in r["stats"].items() if k not in SKIP_KEYS}
    return r


def first_difference(a: dict, b: dict) -> str:
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            x, y = a.get(key), b.get(key)
            if isinstance(x, list) and isinstance(y, list):
                i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
                return f"{key}[{i}]: {x[i] if i < len(x) else '(end)'} vs {y[i] if i < len(y) else '(end)'} " \
                       f"(lengths {len(x)} / {len(y)})"
            return f"{key}: {x!r} vs {y!r}"
    return "(equal)"


def meta2(trace: list[dict]) -> dict | None:
    return next((r for r in trace if r.get("t") == "meta2"), None)


def short(r: dict) -> str:
    return json.dumps({k: v for k, v in r.items() if k not in BULKY}, separators=(",", ":"))[:220]


# -- strict ---------------------------------------------------------------------------------------------------------------
def strict(ref: list[dict], got: list[dict], *, ignore_taps: bool = False, show: int = 3) -> tuple[int, list[str]]:
    out: list[str] = []
    ma, mb = meta2(ref), meta2(got)
    if ma is None or mb is None:
        return 2, ["refused: a trace without a meta2 record (not a v2 leader trace)"]
    if clean(ma) != clean(mb):
        return 2, ["refused: the runs were pinned differently (meta2): " + first_difference(clean(ma), clean(mb))]
    wa = [clean(r, ignore_taps) for r in ref if r.get("t") in WORKER_KINDS]
    wb = [clean(r, ignore_taps) for r in got if r.get("t") in WORKER_KINDS]
    bad = 0
    at = next((i for i, (x, y) in enumerate(zip(wa, wb)) if x != y), None)
    if at is None and len(wa) != len(wb):
        at = min(len(wa), len(wb))
    if at is None:
        out.append(f"worker records: IDENTICAL ({len(wa)} records)")
    else:
        bad += 1
        x = wa[at] if at < len(wa) else {"t": "(none)"}
        y = wb[at] if at < len(wb) else {"t": "(none)"}
        what = first_difference(x, y) if x.get("t") == y.get("t") else f"record kinds {x.get('t')} vs {y.get('t')}"
        out.append(f"worker records: DIFFER at record {at} ({x.get('t')}"
                   f"{', mseq ' + str(x['mseq']) if 'mseq' in x else ''}{', rid ' + str(x['rid']) if 'rid' in x else ''}"
                   f"): {what}")
        for r in (wa[max(0, at - show):at] if show else []):
            out.append(f"    before: {short(r)}")
    ca, cb = by_tag(ref, ignore_taps), by_tag(got, ignore_taps)
    same = 0
    for tag in sorted(set(ca) | set(cb)):
        if ca.get(tag) == cb.get(tag):
            same += 1
            continue
        bad += 1
        x, y = ca.get(tag), cb.get(tag)
        if x is None or y is None:
            out.append(f"request {tag}: only in the {'reference' if y is None else 'second'} trace")
            continue
        kinds = sorted(set(x) | set(y))
        k = next((k for k in kinds if x.get(k) != y.get(k)), kinds[0])
        out.append(f"request {tag}: DIFFERS in {k}: " + first_difference(x.get(k) or {}, y.get(k) or {}))
    out.append(f"{same}/{len(set(ca) | set(cb))} requests identical")
    out.append("IDENTICAL" if not bad else "DIFFERS")
    return (0 if not bad else 1), out


def by_tag(trace: list[dict], ignore_taps: bool = False) -> dict[str, dict]:
    tags: dict[str, dict] = {}
    for r in trace:
        if r.get("t") in CLIENT_KINDS:
            tags.setdefault(r.get("tag"), {})[r["t"]] = clean(r, ignore_taps)
    return tags


# -- stages ---------------------------------------------------------------------------------------------------------------
def stages(leader: list[dict], followers: list[tuple[str, list[dict]]]) -> tuple[int, list[str]]:
    out: list[str] = []
    lm = [(r["mseq"], r["digest"], r["sha"], r["ops"]) for r in leader if r.get("t") == "msg"]
    li = [(r["mseq"], r["pool"]) for r in leader if r.get("t") == "idle"]
    if not lm:
        return 2, ["refused: the leader's trace holds no msg records"]
    bad = 0
    for name, fol in followers:
        fm = [(r["mseq"], r["digest"], r["sha"], r["ops"]) for r in fol if r.get("t") == "msg"]
        start = next((i for i, m in enumerate(fm) if m[0] == lm[0][0]), None)
        if start is None:
            out.append(f"{name}: no message {lm[0][0]} (the leader's first traced one)")
            bad += 1
            continue
        mine = fm[start:start + len(lm)]
        at = next((i for i, (x, y) in enumerate(zip(lm, mine)) if x != y), None)
        if at is None and len(mine) != len(lm):
            at = len(mine)
        if at is not None:
            bad += 1
            x = lm[at] if at < len(lm) else None
            y = mine[at] if at < len(mine) else None
            out.append(f"{name}: message sequence DIFFERS at {at}: leader {x and x[:3]} ops {x and x[3]}, "
                       f"stage {y and y[:3]} ops {y and y[3]}")
        else:
            out.append(f"{name}: {len(lm)} messages equal (sequence number, checksum, ops)")
        fi = [(r["mseq"], r["pool"]) for r in fol if r.get("t") == "idle" and r["mseq"] >= lm[0][0]]
        if fi != li:
            bad += 1
            k = next((i for i, (x, y) in enumerate(zip(li, fi)) if x != y), min(len(li), len(fi)))
            out.append(f"{name}: idle pools DIFFER at idle point {k} ({len(li)} on the leader, {len(fi)} here)"
                       + (f": {first_difference(li[k][1], fi[k][1])}" if k < min(len(li), len(fi)) else ""))
        else:
            out.append(f"{name}: {len(li)} idle pools equal")
    out.append("IDENTICAL" if not bad else "DIFFERS")
    return (0 if not bad else 1), out


# -- streams / tokens -----------------------------------------------------------------------------------------------------
def tokens_by_tag(trace: list[dict]) -> dict[str, list[int]]:
    return {r["tag"]: r["tokens"] for r in trace if r.get("t") == "done"}


def committed_rows(trace: list[dict]) -> dict[str, dict[int, tuple]]:
    """Per tag, position -> (token, logits row hash) of every row a window committed (the first ``keep`` rows)."""

    sid_tag = {r["sid"]: r.get("tag") for r in trace if r.get("t") == "admit"}
    rounds: dict[int, dict] = {}
    rows: dict[str, dict[int, tuple]] = {}
    for r in trace:
        if r.get("t") == "round":
            rounds[r["rid"]] = r
        elif r.get("t") == "keeps" and r["rid"] in rounds:
            rd = rounds.pop(r["rid"])
            at = 0
            for k, (slot, pos, n) in enumerate(rd["segs"]):
                tag = sid_tag.get(rd["sids"][k])
                for i in range(int(r["keeps"][k])):
                    rows.setdefault(tag, {})[pos + i] = (rd["tokens"][at + i], tuple(rd["logits"][at + i]))
                at += n
    return rows


def streams(ref: list[dict], got: list[dict], *, rows: bool = False) -> tuple[int, list[str]]:
    out: list[str] = []
    ta, tb = tokens_by_tag(ref), tokens_by_tag(got)
    bad = 0
    for tag in sorted(set(ta) | set(tb)):
        x, y = ta.get(tag), tb.get(tag)
        if x == y:
            continue
        bad += 1
        if x is None or y is None:
            out.append(f"{tag}: only in the {'reference' if y is None else 'second'} trace")
        else:
            i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            out.append(f"{tag}: tokens differ from token {i} (lengths {len(x)} / {len(y)})")
    out.append(f"{len(set(ta) | set(tb)) - bad}/{len(set(ta) | set(tb))} replies identical")
    if rows:
        ra, rb = committed_rows(ref), committed_rows(got)
        nrow = 0
        for tag in sorted(set(ra) | set(rb)):
            x, y = ra.get(tag, {}), rb.get(tag, {})
            common = sorted(set(x) & set(y))
            nrow += len(common)
            diff = [p for p in common if x[p] != y[p]]
            if diff:
                bad += 1
                p = diff[0]
                out.append(f"{tag}: committed row at position {p} differs: token {x[p][0]} vs {y[p][0]}, "
                           f"logits {'equal' if x[p][1] == y[p][1] else 'differ'} ({len(diff)} positions differ of "
                           f"{len(common)} both runs committed)")
        out.append(f"{nrow} committed rows compared across {len(set(ra) | set(rb))} requests")
    out.append("IDENTICAL" if not bad else "DIFFERS")
    return (0 if not bad else 1), out


def tokens(ref: list[dict], got: list[dict]) -> tuple[int, list[str]]:
    return streams(ref, got, rows=False)


# -- bench ----------------------------------------------------------------------------------------------------------------
def bench(a: list[dict], b: list[dict], prefix_ids=("stop-after-5",)) -> tuple[int, list[str]]:
    """v1 bench-out (the first file: the reference) against v2's, by (id, rep): every entry of the second file must
    reply with the reference's tokens (a free run's passes carry rep + 1000 * pass: they are all compared with the
    reference's rep); for the ids that end early on a stop either side may be the prefix of the other."""

    def key(r):
        return (r["id"], r.get("rep", 0) % 1000)

    ref = {key(r): r for r in a}
    out, bad, seen, equal = [], 0, set(), 0
    for r in b:
        k = key(r)
        seen.add(k)
        x = ref.get(k)
        name = f"{k[0]}#{r.get('rep', 0)}"
        if x is None:
            bad += 1
            out.append(f"{name}: not in the reference")
            continue
        if "error" in x and "error" in r:                  # refused alike (e.g. a prompt past the context): the texts
            equal += 1                                     # carry each engine's own limit, so they are not compared
            continue
        if "error" in x or "error" in r:
            bad += 1
            out.append(f"{name}: failed on one side only ({x.get('error') or r.get('error')})")
            continue
        tx, ty = x["tokens"], r["tokens"]
        if tx == ty:
            equal += 1
            continue
        n = min(len(tx), len(ty))
        if k[0] in prefix_ids and tx[:n] == ty[:n]:
            equal += 1
            out.append(f"{name}: prefix-equal ({len(tx)} / {len(ty)} tokens, a stop)")
            continue
        bad += 1
        i = next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), n)
        out.append(f"{name}: tokens differ from token {i} (lengths {len(tx)} / {len(ty)})")
    for k in sorted(set(ref) - seen):
        bad += 1
        out.append(f"{k[0]}#{k[1]}: only in the reference")
    out.append(f"{equal}/{len(b)} replies equal the reference's")
    out.append("IDENTICAL" if not bad else "DIFFERS")
    return (0 if not bad else 1), out


# -- CLI ------------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.glm.v2compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("strict", "stages", "streams", "tokens", "bench"))
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ignore-taps", action="store_true")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--rows", action="store_true", help="streams: also every committed row (position, token, logits)")
    ap.add_argument("--prefix", default="stop-after-5", help="bench: ids whose replies may be a prefix of each other")
    a = ap.parse_args(argv)
    if a.mode == "stages":
        if len(a.files) < 2:
            raise SystemExit("stages: LEADER.jsonl FOLLOWER.jsonl [...]")
        code, lines = stages(load(a.files[0]), [(Path(f).name, load(f)) for f in a.files[1:]])
    else:
        if len(a.files) != 2:
            raise SystemExit(f"{a.mode}: two files")
        if a.mode == "bench":
            ja, jb = (json.loads(Path(f).read_text(encoding="utf-8")) for f in a.files)
            code, lines = bench(ja, jb, tuple(x for x in a.prefix.split(",") if x))
        else:
            ra, rb = load(a.files[0]), load(a.files[1])
            code, lines = {"strict": lambda: strict(ra, rb, ignore_taps=a.ignore_taps, show=a.show),
                           "streams": lambda: streams(ra, rb, rows=a.rows),
                           "tokens": lambda: tokens(ra, rb)}[a.mode]()
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
