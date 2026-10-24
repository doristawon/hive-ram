"""Compare two traces and name the first place they part (spec 15.6; no torch).

  python -m tfpp.ds41.compare ref.jsonl got.jsonl [--tokens-only | --by-position] [--show N]
                              [--allow KEY,KEY ...] [--ignore-meta]

Default: request by request (a ``req`` record opens one), every record in order -- admissions, prompt chunks (rows,
logits and taps hashes), rounds (rows, tokens, every logits row's hash, every taps row's hash), kept drafts, the reply's
tokens and its rounds / drafted / accepted. For sequential prompt sets (the bench): both runs see the same history.
``--tokens-only``: each request's reply tokens alone, matched by prompt (and its occurrence).
``--by-position``: concurrent requests (T1-multi), whose rounds group streams by arrival: every computed row keyed by
(its stream's prompt and occurrence, the window's first position, the row's index and the window's tokens up to it --
the row's whole input), equal logits and taps wherever both runs computed the same row, and equal replies.
Exit 0: identical; 1: they differ; 2: not comparable.
Before any record is compared the two runs' ``meta`` records are: each trace opens with what its run was made under
(trace.run_meta: the modules' settings and the env digest, the model / config / selection identities, the versions of
torch, Triton, the fork and tfpp-ds41, context, parallel, drafts, graphs). Runs made under different settings cannot be
told apart from a real difference, so they are refused (exit 2, the differing keys named). ``--allow`` names keys (dotted,
a prefix covers what is below it: ``settings.depth``, ``graphs``) that a test changes on purpose (T0b: ``graphs``;
T-ref-graphs: ``graphs,settings.graphs_env,settings.env``); ``--ignore-meta`` skips the check. A trace without a ``run``
record (made before this check) is compared with a warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tfpp.glm.compare import first_difference, load  # noqa: F401  (the same JSONL reader and difference text)

COMPARED = ("req", "admit", "chunk", "step", "keep", "done", "fail")
SKIP_KEYS = {"wall_ms", "time", "ms"}


def records(trace: list[dict]) -> list[dict]:
    out = []
    for r in trace:
        if r.get("t") not in COMPARED:
            continue
        out.append({k: v for k, v in r.items() if k not in SKIP_KEYS})
    return out


def run_meta_of(trace: list[dict]) -> dict | None:
    for r in trace:
        if r.get("t") == "meta":
            return r.get("run")
    return None


def _flat(d, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def meta_differences(ref: list[dict], got: list[dict], allow=()) -> list[str] | None:
    """The keys of the two runs' ``run`` metas that differ and are not allowed (None: a trace has no meta)."""

    a, b = run_meta_of(ref), run_meta_of(got)
    if a is None or b is None:
        return None
    fa, fb = _flat(a), _flat(b)
    allow = tuple(x.strip() for x in allow if x.strip())
    out = []
    for key in sorted(set(fa) | set(fb)):
        if fa.get(key, "(absent)") == fb.get(key, "(absent)"):
            continue
        if any(key == x or key.startswith(x + ".") for x in allow):
            continue
        out.append(f"{key}: {str(fa.get(key, '(absent)'))[:100]} vs {str(fb.get(key, '(absent)'))[:100]}")
    return out


def requests(recs: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = [[]]
    for r in recs:
        if r["t"] == "req":
            groups.append([])
        groups[-1].append(r)
    return groups if groups[0] else groups[1:]


def compare_records(ref: list[dict], got: list[dict], *, show: int = 3, out=print) -> int:
    ra, rb = requests(records(ref)), requests(records(got))
    same, first = 0, None
    for n, (ga, gb) in enumerate(zip(ra, rb), 1):
        diff_at = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
        if diff_at is None and len(ga) != len(gb):
            diff_at = min(len(ga), len(gb))
        head = ga[0] if ga and ga[0]["t"] == "req" else {}
        ta = next((r["tokens"] for r in ga if r["t"] == "done"), None)
        tb = next((r["tokens"] for r in gb if r["t"] == "done"), None)
        if diff_at is None:
            same += 1
            out(f"request {n}: IDENTICAL ({len(ga)} records, {len(ta or [])} tokens, prompt {head.get('prompt_len')} "
                f"draft {head.get('draft')})")
            continue
        tok = ""
        if ta is not None and tb is not None:
            t = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                     None if len(ta) == len(tb) else min(len(ta), len(tb)))
            tok = "tokens identical" if t is None else f"tokens differ from token {t}"
        x = ga[diff_at] if diff_at < len(ga) else {"t": "(none)"}
        y = gb[diff_at] if diff_at < len(gb) else {"t": "(none)"}
        what = first_difference(x, y) if x["t"] == y["t"] else f"record kinds {x['t']} vs {y['t']}"
        where = ""
        if x["t"] == "step":
            where = f", round {x.get('round')} rows {x.get('rows')}"
        elif x["t"] == "chunk":
            where = f", sid {x.get('sid')} start {x.get('start')}"
        out(f"request {n}: DIFFERS at record {diff_at} ({x['t']}{where}): {what}; {tok}")
        for r in (ga[max(0, diff_at - show):diff_at] if show else []):
            out(f"    before: {json.dumps({k: r[k] for k in r if k not in ('logits', 'taps')})[:200]}")
        if first is None:
            first = (n, diff_at)
    if len(ra) != len(rb):
        out(f"request counts differ: {len(ra)} vs {len(rb)}")
    total = max(len(ra), len(rb))
    out(f"{same}/{total} requests identical" + (f"; first difference: request {first[0]}, record {first[1]}"
                                                 if first else ""))
    return 0 if same == total and len(ra) == len(rb) else 1


def replies(trace: list[dict]) -> dict:
    """(prompt sha, occurrence) -> the reply's tokens (or its error)."""

    occ: Counter = Counter()
    out = {}
    for r in trace:
        if r.get("t") in ("done", "fail"):
            k = r.get("prompt_sha")
            out[(k, occ[k])] = r.get("tokens") if r["t"] == "done" else ("error", r.get("error"))
            occ[k] += 1
    return out


def compare_tokens(ref: list[dict], got: list[dict], *, out=print) -> int:
    a, b = replies(ref), replies(got)
    bad = 0
    for key in sorted(set(a) | set(b), key=str):
        if a.get(key) != b.get(key):
            bad += 1
            x, y = a.get(key), b.get(key)
            if isinstance(x, list) and isinstance(y, list):
                i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
                out(f"reply {key[0]}#{key[1]}: DIFFERS from token {i} (lengths {len(x)} / {len(y)})")
            else:
                out(f"reply {key[0]}#{key[1]}: {str(x)[:80]} vs {str(y)[:80]}")
    out(f"{len(set(a) | set(b)) - bad}/{len(set(a) | set(b))} replies identical")
    return 0 if bad == 0 and a else 1


def rows_by_input(trace: list[dict]) -> dict:
    """(stream key, window start, row index, window tokens up to the row) -> (logits hash, taps hash)."""

    occ: Counter = Counter()
    key_of: dict = {}
    table: dict = {}
    for r in trace:
        t = r.get("t")
        if t == "admit":
            k = r["prompt_sha"]
            key_of[r["sid"]] = f"{k}#{occ[k]}"
            occ[k] += 1
        elif t == "chunk":
            table[(key_of.get(r["sid"]), "chunk", r["start"], r["rows"])] = (r.get("logits"), r.get("taps"))
        elif t == "step":
            at = 0
            taps = r.get("taps") or [None] * len(r["tokens"])
            for sid, _slot, p0, R in r["rows"]:
                window = r["tokens"][at:at + R]
                for i in range(R):
                    table[(key_of.get(sid), p0, i, tuple(window[:i + 1]))] = (r["logits"][at + i], taps[at + i])
                at += R
    return table


def compare_positions(ref: list[dict], got: list[dict], *, out=print) -> int:
    a, b = rows_by_input(ref), rows_by_input(got)
    common = set(a) & set(b)
    bad = [k for k in common if a[k] != b[k]]
    for k in sorted(bad, key=str)[:10]:
        out(f"row {k[0]} at {k[1]}+{k[2] if len(k) > 2 else ''}: {a[k]} vs {b[k]}")
    out(f"{len(common)} rows computed by both runs ({len(a)} / {len(b)} in each), {len(bad)} differ")
    rc = compare_tokens(ref, got, out=out)
    return 1 if bad or rc or not common else 0


def compare(ref: list[dict], got: list[dict], *, mode: str = "records", show: int = 3, out=print) -> int:
    if mode == "tokens":
        return compare_tokens(ref, got, out=out)
    if mode == "position":
        return compare_positions(ref, got, out=out)
    return compare_records(ref, got, show=show, out=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.ds41.compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("got")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tokens-only", action="store_true")
    g.add_argument("--by-position", action="store_true")
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--allow", default="", help="comma-separated run-meta keys that may differ (a test's own change)")
    ap.add_argument("--ignore-meta", action="store_true", help="do not compare the runs' settings")
    a = ap.parse_args(argv)
    mode = "tokens" if a.tokens_only else "position" if a.by_position else "records"
    ref, got = load(Path(a.ref)), load(Path(a.got))
    if not a.ignore_meta:
        diffs = meta_differences(ref, got, a.allow.split(","))
        if diffs is None:
            print("warning: a trace has no run meta (settings, versions): the runs' settings are not compared")
        elif diffs:
            print(f"NOT COMPARABLE: the two runs were made under different settings ({len(diffs)} keys):")
            for d in diffs[:40]:
                print(f"    {d}")
            print("(--allow KEY for a difference the test makes on purpose; --ignore-meta to skip this check)")
            return 2
    return compare(ref, got, mode=mode, show=a.show)


if __name__ == "__main__":
    sys.exit(main())
