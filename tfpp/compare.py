# SPDX-License-Identifier: Apache-2.0
"""Compare two tfpp_run outputs token by token: python compare.py ref.json pp.json

Exit status: 0 every prompt identical; 1 some differ; 2 not comparable (an unreadable or empty file, or different
prompt lists: run both with the same --repeat).
"""

from __future__ import annotations

import json
import sys


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def comparable(ref, got) -> str | None:
    """Why the two result lists cannot be compared prompt by prompt, or None."""

    if not isinstance(ref, list) or not isinstance(got, list) or not ref:
        return "each file must hold a non-empty list of results"
    if not all(isinstance(r, dict) and "tokens" in r for r in ref + got):
        return "an entry without tokens"
    if len(ref) != len(got) or any(r.get("prompt") != g.get("prompt") for r, g in zip(ref, got)):
        return (f"{len(ref)} vs {len(got)} results, prompts {[r.get('prompt') for r in ref]} vs "
                f"{[g.get('prompt') for g in got]} (run both with the same --repeat)")
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python compare.py ref.json pp.json", file=sys.stderr)
        return 2
    try:
        ref, got = load(argv[1]), load(argv[2])
    except (OSError, ValueError) as exc:
        print(f"not comparable: {exc}", file=sys.stderr)
        return 2
    why = comparable(ref, got)
    if why:
        print(f"not comparable: {why}", file=sys.stderr)
        return 2
    same = 0
    for r, g in zip(ref, got):
        a, b = r["tokens"], g["tokens"]
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        equal = a == b
        same += equal
        print(f"prompt {r['prompt']}: {'IDENTICAL' if equal else f'differs at token {n}'} ({len(a)} vs {len(b)} tokens) | "
              f"decode {r['decode_tps']:.1f} -> {g['decode_tps']:.1f} tok/s, prefill {r['prefill_tps']:.0f} -> "
              f"{g['prefill_tps']:.0f} tok/s, accepted {r['accepted']}/{r['drafted']} -> {g['accepted']}/{g['drafted']}")
    print(f"{same}/{len(ref)} identical")
    return 0 if same == len(ref) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
