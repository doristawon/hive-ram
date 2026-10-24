"""tfpp-glm's run modes in tfpp_run.py's terms (a thin front for ``tfpp.glm.serve``).

  local  : python -m tfpp.glm.run local  --model DIR --layers 0:L [serve's leader flags]   one process (reference)
  first  : python -m tfpp.glm.run first  --model DIR --layers 0:B --down HOST:PORT          stage 0 (a follower)
  middle : python -m tfpp.glm.run middle --model DIR --layers A:B --listen H:P --down H:P   a middle stage (follower)
  leader : python -m tfpp.glm.run leader --model DIR --layers A:END --listen H:P             the last stage: head,
           sampling, drafts, HTTP (or --bench); ``last`` is the same mode (in tfpp-glm the leader is the last stage)

Every other flag goes to ``python -m tfpp.glm.serve`` unchanged (see its --help).
"""

from __future__ import annotations

import os
import sys

MODES = {"local": "leader", "leader": "leader", "last": "leader", "first": "follower", "middle": "follower"}


def translate(argv: list[str]) -> list[str]:
    if not argv or argv[0] not in MODES:
        raise SystemExit(f"usage: python -m tfpp.glm.run {{{'|'.join(MODES)}}} [serve flags]\n\n{__doc__}")
    mode, rest = argv[0], list(argv[1:])
    if "--role" in rest:
        raise SystemExit("the mode sets --role")
    if mode == "local" and ("--listen" in rest or "--down" in rest):
        raise SystemExit("local runs one process: no --listen / --down")
    if mode == "first":
        for i, x in enumerate(rest):
            if x == "--layers" and i + 1 < len(rest) and not rest[i + 1].startswith("0:"):
                raise SystemExit("first: --layers 0:B")
    if mode == "middle" and "--listen" not in rest:
        raise SystemExit("middle: --listen HOST:PORT (the stage before connects to it)")
    return ["--role", MODES[mode], *rest]


def main(argv: list[str] | None = None) -> int:
    from .serve import main as serve_main

    return serve_main(translate(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
