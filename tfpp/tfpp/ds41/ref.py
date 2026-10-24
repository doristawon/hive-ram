"""The reference (spec 3.5, 15.6 T-ref): the unmodified fork's DsEngine at world=1 (``--parallel P``: MultiDecoder,
RoundRunner, the round graphs, as the fork serves it on DGX Spark) on one process, one change only: ``weights.load``
is swapped for ``load.load_full`` (a module attribute, read when DsEngine.__init__ runs: F/engine.py:54, 65), so the
shortened model's renamed shards load. Then the same self-check, trace hooks, oracle and bench as the chain's leader.

  python -m tfpp.ds41.ref --model DIR --engram DIR --context N --parallel P (--mtp-drafts K | --no-drafts)
          --bench PROMPTS.jsonl --trace OUT.jsonl [--bench-out O.json] [--oracle REF.json] [--no-graphs] [--experts K]
          [--bench-max-prompt N]
--no-graphs sets TF_DS_GRAPHS=0 before the fork is imported (F/engine.py:26); --experts K TFPP_DS41_EXPERTS=K (the
T-X0 proxy, load_full reads it).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tfpp.ds41.ref", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engram", required=True, help="a folder with every Engram layer's table (the test tables: both;"
                    " always explicit: DsEngine would otherwise glob the model's parent folder, F/engine.py:590-597)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--parallel", type=int, default=2)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mtp-drafts", type=int, default=5)
    g.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--bench-out", default="")
    ap.add_argument("--bench-max-prompt", type=int, default=0)
    ap.add_argument("--trace", default="")
    ap.add_argument("--oracle", default="")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--no-self-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.parallel < 2:
        raise SystemExit("the reference runs the --parallel path (MultiDecoder): --parallel 2 or more (F/engine.py:120)")
    if a.no_graphs:
        os.environ["TF_DS_GRAPHS"] = "0"
    if a.experts:
        have = os.environ.get("TFPP_DS41_EXPERTS")
        if have and int(have) != a.experts:
            raise SystemExit(f"TFPP_DS41_EXPERTS={have} and --experts {a.experts}")
        os.environ["TFPP_DS41_EXPERTS"] = str(a.experts)
    from . import env, pins

    try:
        env.check_env("reference")
        pins.check_tree()
        pins.check_seams()
        env.check_device()
    except (env.EnvError, pins.TreeMismatch) as exc:
        print(f"[tfpp-ds41] {exc}", flush=True)
        return 1
    from tensorfold.families.deepseek_v41 import fp32_gemms
    from tensorfold.families.deepseek_v41.cuda import engine as E, weights as W

    from . import bench, load, oracle, trace
    from .engine import self_check

    W.load = load.load_full                                  # the one swap (spec 3.5)
    drafts = 0 if a.no_drafts else int(a.mtp_drafts)
    t0 = time.perf_counter()
    eng = E.DsEngine(Path(a.model), rank=0, world=1, master="", port=0, drafts=drafts, context=a.context,
                     engram_dir=a.engram, parallel=a.parallel)
    fp32_gemms()
    print(f"[tfpp-ds41] reference ready in {time.perf_counter() - t0:.0f}s (world 1, --parallel {a.parallel}, "
          f"drafts {drafts}, graphs {E.GRAPHS})", flush=True)
    code = 0
    tracer = None
    try:
        if not a.no_self_check:
            self_check(eng)
        tracer = trace.make(a.trace or None, "reference", (0, eng.w.cfg.n_layers),
                            trace.run_meta(a.model, context=int(a.context), parallel=int(a.parallel), drafts=drafts,
                                           graphs=bool(E.GRAPHS)) if a.trace else None)
        if tracer is not None:
            tracer.install(eng)
        if a.oracle:
            oracle.install(eng, oracle.load_oracle(a.oracle))
        bench.run(eng, bench.load_prompts(a.bench), model_dir=a.model, out_path=a.bench_out or None,
                  context=a.context, max_prompt=a.bench_max_prompt or None)
    except BaseException as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"[tfpp-ds41] reference: {type(exc).__name__}: {exc}", flush=True)
        code = 1
    finally:
        if tracer is not None:
            tracer.close()
    return code


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
