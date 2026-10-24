"""tfpp-ds41's run modes in tfpp_run.py's terms (a thin front for ``tfpp.ds41.serve``).

  local  : python -m tfpp.ds41.run local  --model DIR --layers 0:L [serve's leader flags]   one stage (T0a)
  first  : python -m tfpp.ds41.run first  --model DIR --layers 0:B --down H:P              stage 0 (a follower)
  middle : python -m tfpp.ds41.run middle --model DIR --layers A:B --listen H:P --down H:P a middle stage
  last   : python -m tfpp.ds41.run last   --model DIR --layers A:L --listen H:P            the leader: head, DSpark,
           sampling, HTTP (or --bench); ``leader`` is the same mode

Every other flag goes to ``python -m tfpp.ds41.serve`` unchanged (its --help). The unmodified fork's one-process
reference is ``python -m tfpp.ds41.ref``.
"""

from __future__ import annotations

import os
import sys

MODES = {"local": "leader", "last": "leader", "leader": "leader", "first": "follower", "middle": "follower"}


def translate(argv: list[str]) -> list[str]:
    if not argv or argv[0] not in MODES:
        raise SystemExit(f"usage: python -m tfpp.ds41.run {{{'|'.join(MODES)}}} [serve flags]\n\n{__doc__}")
    mode, rest = argv[0], list(argv[1:])
    if "--role" in rest:
        raise SystemExit("the mode sets --role")
    if mode == "local" and ("--listen" in rest or "--down" in rest):
        raise SystemExit("local runs one stage: no --listen / --down")
    if mode in ("local", "first"):
        for i, x in enumerate(rest):
            if x == "--layers" and i + 1 < len(rest) and not rest[i + 1].startswith("0:"):
                raise SystemExit(f"{mode}: --layers 0:B")
    if mode == "middle" and ("--listen" not in rest or "--down" not in rest):
        raise SystemExit("middle: --listen HOST:PORT and --down HOST:PORT")
    return ["--role", MODES[mode], *rest]


def main(argv: list[str] | None = None) -> int:
    from .serve import main as serve_main

    return serve_main(translate(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
