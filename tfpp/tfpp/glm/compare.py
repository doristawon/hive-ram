"""Compare two leader traces record by record and name the first place they part (spec 13.3; no torch).

  python -m tfpp.glm.compare ref.jsonl got.jsonl [--ignore-taps] [--tokens-only] [--show N]

Per request: the tokens and the reply's stats; per prompt chunk: rows and hashes; per verify window: position,
rows, tokens (pending + drafts), every logits row's hash, the taps' hashes; per window: the rows kept. Timings are
not compared. Exit 0: identical; 1: they differ (the first difference is printed with its context), a request did
not end (no ``done`` or ``fail`` record: a run that stopped mid-request is never a pass), or there is nothing to
compare (no request in either trace).
--tokens-only compares each request's reply tokens alone: for runs whose kept-prompt byte cap binds (TF_GLM_CACHE_GIB
pressure), where which states survive depends on the cut (each leader counts its own layers' bytes), so a request may
resume in one run and prefill fresh in the other; MIA's resume-equals-fresh invariant still holds for the tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMPARED = ("req", "chunk", "step", "keep", "done", "fail")
SKIP_KEYS = {"wall_ms", "ms", "time", "prefill_s", "decode_s", "stages_ms", "encode_s"}


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


def records(trace: list[dict], ignore_taps: bool = False, tokens_only: bool = False) -> list[dict]:
    out = []
    for r in trace:
        if r.get("t") not in COMPARED:
            continue
        if tokens_only:
            if r["t"] in ("req", "done", "fail"):
                out.append({k: r.get(k) for k in ("t", "n", "prompt_sha", "draft", "policy", "tokens", "error")
                            if k in r})
            continue
        r = {k: v for k, v in r.items() if k not in SKIP_KEYS}
        if ignore_taps:
            r.pop("taps", None)
        if r.get("t") == "done" and isinstance(r.get("stats"), dict):
            r["stats"] = {k: v for k, v in r["stats"].items() if k not in SKIP_KEYS}
        out.append(r)
    return out


def requests(recs: list[dict]) -> list[list[dict]]:
    """Records grouped by request (a ``req`` record opens one; anything before the first is request 0)."""

    groups: list[list[dict]] = [[]]
    for r in recs:
        if r["t"] == "req":
            groups.append([])
        groups[-1].append(r)
    return groups if groups[0] else groups[1:]


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


def compare(ref: list[dict], got: list[dict], *, ignore_taps: bool = False, tokens_only: bool = False,
            show: int = 3) -> int:
    ra = requests(records(ref, ignore_taps, tokens_only))
    rb = requests(records(got, ignore_taps, tokens_only))
    same = 0
    first = None
    for n, (ga, gb) in enumerate(zip(ra, rb), 1):
        diff_at = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
        if diff_at is None and len(ga) != len(gb):
            diff_at = min(len(ga), len(gb))
        head = ga[0] if ga and ga[0]["t"] == "req" else {}
        tokens_a = next((r["tokens"] for r in ga if r["t"] == "done"), None)
        tokens_b = next((r["tokens"] for r in gb if r["t"] == "done"), None)
        if diff_at is None and head and not ended(ga):
            print(f"request {n}: INCOMPLETE ({len(ga)} records equal so far, no done or fail record in either trace: "
                  "both runs stopped mid-request)")
            if first is None:
                first = (n, len(ga))
            continue
        if diff_at is None:
            same += 1
            print(f"request {n}: IDENTICAL ({len(ga)} records, {len(tokens_a or [])} tokens, "
                  f"prompt {head.get('prompt_len')} draft {head.get('draft')})")
            continue
        tok = ""
        if tokens_a is not None and tokens_b is not None:
            t = next((i for i, (x, y) in enumerate(zip(tokens_a, tokens_b)) if x != y),
                     None if len(tokens_a) == len(tokens_b) else min(len(tokens_a), len(tokens_b)))
            tok = "tokens identical" if t is None else f"tokens differ from token {t}"
        x = ga[diff_at] if diff_at < len(ga) else {"t": "(none)"}
        y = gb[diff_at] if diff_at < len(gb) else {"t": "(none)"}
        what = first_difference(x, y) if x["t"] == y["t"] else f"record kinds {x['t']} vs {y['t']}"
        print(f"request {n}: DIFFERS at record {diff_at} ({x['t']}"
              f"{', pos ' + str(x.get('pos', x.get('start'))) if x['t'] in ('step', 'chunk', 'keep') else ''}): "
              f"{what}; {tok}")
        for r in (ga[max(0, diff_at - show):diff_at] if show else []):
            print(f"    before: {json.dumps({k: r[k] for k in r if k not in ('logits', 'taps')})[:200]}")
        if first is None:
            first = (n, diff_at)
    if len(ra) != len(rb):
        print(f"request counts differ: {len(ra)} vs {len(rb)}")
    total = max(len(ra), len(rb))
    if total == 0:
        print("0/0 requests: neither trace holds a request, nothing was compared")
        return 1
    print(f"{same}/{total} requests identical" + (f"; first difference: request {first[0]}, record {first[1]}"
                                                  if first else ""))
    return 0 if same == total and len(ra) == len(rb) else 1


def ended(group: list[dict]) -> bool:
    """A request's records end in its ``done`` or ``fail`` record (a leader that stopped mid-request leaves neither)."""

    return any(r["t"] in ("done", "fail") for r in group)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.glm.compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("got")
    ap.add_argument("--ignore-taps", action="store_true", help="leave the DFlash2 taps' hashes out")
    ap.add_argument("--tokens-only", action="store_true", help="each request's reply tokens alone (kept-prompt "
                    "pressure runs: which states survive a byte cap depends on the cut)")
    ap.add_argument("--show", type=int, default=3, help="records printed before a difference")
    a = ap.parse_args(argv)
    return compare(load(a.ref), load(a.got), ignore_taps=a.ignore_taps, tokens_only=a.tokens_only, show=a.show)


if __name__ == "__main__":
    sys.exit(main())
