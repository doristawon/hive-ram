"""tfpp-glm v2: the leader's entry for ``--parallel 2..8`` (spec v2 sections 6.2, 12).

``serve.check_args`` and ``serve._leader`` hand over to this module when ``--parallel`` is above 1 (two whitelisted
statements in serve.py); with ``--parallel 1`` nothing here is imported. The followers need no flag: they take P from GO.

  python -m tfpp.glm.serve --role leader --layers A:END --parallel 4 --drafter DIR [--pool-rows N --kept-entries E]
         [--mbench SCHEDULE.json [--bench-out OUT.json]] [--bench PROMPTS.jsonl] [--trace OUT.jsonl]   [serve's other flags]

HTTP (the default) serves ``GlmApp`` with ``concurrent=True``: the server hands every request to ``generate``, MIA's
scheduler orders them. ``--mbench`` runs a schedule (v2bench: lockstep for the equivalence tests, free for throughput)
instead; ``--bench`` runs the prompt set one request at a time through the same scheduler.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def check_args_v2(a: argparse.Namespace) -> None:
    """serve.check_args' checks for P > 1, after v1's own earlier ones."""

    if not 2 <= a.parallel <= 8:
        raise SystemExit(f"--parallel {a.parallel}: 1 to 8 requests at once")
    if not a.drafter and not a.no_drafts:
        raise SystemExit("--parallel needs DFlash2: --drafter DIR (concurrent requests draft with DFlash2 only), "
                         "or --no-drafts for the serial reference")
    if a.mbench and a.bench:
        raise SystemExit("--mbench or --bench, not both")
    if a.bench_out and not (a.bench or a.mbench):
        raise SystemExit("--bench-out needs --bench or --mbench")
    if a.pool_rows is not None and (a.pool_rows < 2048 or a.pool_rows % 2048):
        raise SystemExit(f"--pool-rows {a.pool_rows}: a multiple of 2048, at least 2048")
    if a.kept_entries is not None and a.kept_entries < 1:
        raise SystemExit(f"--kept-entries {a.kept_entries}: at least 1")
    if a.mbench:
        from . import v2bench

        try:
            v2bench.load_schedule(a.mbench)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--mbench {a.mbench}: {exc}") from None


def leader_v2(a: argparse.Namespace) -> int:
    """``serve._leader`` for P > 1."""

    from . import startup, v2trace
    from .tail import exit_code
    from .v2tail import PipeMultiEngine
    from tensorfold.families.glm5_next.cuda.engine import DEFAULT_POLICY

    first, end = a.layers
    link = None
    if first > 0:
        link = startup.leader_listen(a.listen, io_timeout=a.io_timeout, accept_timeout=a.connect_timeout)
    eng = None
    try:
        eng = PipeMultiEngine(Path(a.model), layers=(first, end), truncate=a.truncate, link=link,
                              policy=a.policy or DEFAULT_POLICY, drafter=a.drafter or None, context=a.context,
                              serial_only=a.no_drafts, prefill_rows=a.prefill_rows, vision=a.vision,
                              vision_urls=a.vision_urls, parallel=a.parallel, graphs=not a.no_graphs,
                              tap_layers=a.tap_layers, reserve_gib=a.reserve_gib, budget_gib=a.budget_gib,
                              io_timeout=a.io_timeout, startup_timeout=a.connect_timeout, pure=a.pure,
                              warmup=not a.no_warmup, pool_rows=a.pool_rows, kept_entries=a.kept_entries)
    except BaseException as exc:
        print(f"[tfpp-glm] leader: startup failed: {type(exc).__name__}: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        startup.leader_fail(link, "startup", exc)    # the followers stop too (exit 75)
        return exit_code(exc) if link is not None else 1
    tracer = v2trace.make(a.trace, "leader", a.layers)
    if tracer is not None:
        v2trace.install_leader(tracer, eng)
    code = 0
    try:
        if a.mbench:
            from . import v2bench

            v2bench.run_mbench(eng, a, tracer=tracer)
        elif a.bench:
            from . import bench

            bench.run(eng, bench.load_prompts(a.bench), model_dir=a.model, out_path=a.bench_out or None)
        else:
            from .serve import _serve_http

            _serve_http(eng, a)
    except BaseException as exc:  # noqa: BLE001
        if not isinstance(exc, KeyboardInterrupt):
            print(f"[tfpp-glm] leader: {type(exc).__name__}: {exc}", flush=True)
            import traceback

            traceback.print_exc()
            code = exit_code(exc)
    finally:
        eng.quit()                                    # QUIT down the chain (nothing once it broke)
        if tracer is not None:
            tracer.close()
    if eng.broken:
        from .proto import FAIL_EXIT

        return FAIL_EXIT
    return code
