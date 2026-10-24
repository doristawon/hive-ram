"""tfpp-glm's entry point: one stage of the chain (spec 12.3).

  leader:   python -m tfpp.glm.serve --role leader --model DIR --layers A:END [--listen HOST:PORT (A > 0)]
            [--drafter DIR | --no-drafts] [--context N] [--parallel 1] [--prefill-rows 2048] [--no-graphs]
            [--name ...] [--host 0.0.0.0] [--port 8888] [--max-tokens 16384] [--thinking] [--vision] [--vision-urls]
            [--reserve-gib 3] [--budget-gib X] [--io-timeout 600] [--trace OUT.jsonl]
            [--bench PROMPTS.jsonl [--bench-out OUT.json]]    no HTTP: run the set, then QUIT the chain
            [--pure]                                          A = 0 only: the package's own _compute (reference)
            [--truncate L --tap-layers a,b,c,d,e]             shortened test models only
  follower: python -m tfpp.glm.serve --role follower --model DIR --layers A:B --down HOST:PORT
            [--listen HOST:PORT (A > 0)] [--reserve-gib 3] [--budget-gib X] [--no-prefetch-next]
            [--io-timeout 600] [--connect-timeout 3600] [--trace OUT.jsonl]

``--role leader --layers 0:L`` alone is the one-process reference (N=1). The followers take P, the context and the
prompt rows from the leader (GO, AGREE). Exit codes: 0 QUIT, 75 the chain broke (restart every stage), 1 other.
The parser runs without torch (tests); everything else is imported after env.check_env.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _layers(text: str) -> tuple[int, int]:
    from .proto import parse_layers

    try:
        return parse_layers(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _taps(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"--tap-layers {text!r}: comma separated layer numbers") from None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tfpp.glm.serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("leader", "follower"), required=True)
    ap.add_argument("--model", required=True, help="the checkpoint folder (config, tokenizer, this stage's shards)")
    ap.add_argument("--layers", type=_layers, required=True, help="A:B, this stage's layers [A, B)")
    ap.add_argument("--listen", default="", help="HOST:PORT the stage before this one connects to (A > 0)")
    ap.add_argument("--reserve-gib", type=float, default=float(os.environ.get("TFPP_RESERVE_GIB", "") or 3.0),
                    help="GPU memory left free past the caches (2.5 on the 64 GB node)")
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="the caches' and buffers' budget outright (several stages on one GPU)")
    ap.add_argument("--io-timeout", type=float, default=600.0, help="seconds a request waits for a frame")
    ap.add_argument("--trace", default="", help="JSONL trace of every request (compare with tfpp.glm.compare)")
    # follower
    ap.add_argument("--down", default="", help="follower: HOST:PORT of the next stage (towards the leader)")
    ap.add_argument("--connect-timeout", type=float, default=3600.0)
    ap.add_argument("--no-prefetch-next", action="store_true", help="load one layer at a time (less host memory)")
    # leader: engine
    ap.add_argument("--drafter", default="", help="DFlash2's folder (incoai/GLM-5.3-Flash-DFlash2@bf582e4e)")
    ap.add_argument("--no-drafts", action="store_true", help="serial decoding only (the reference)")
    ap.add_argument("--policy", default="", help="the default draft policy (MIA's DEFAULT_POLICY: auto)")
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--parallel", type=int, default=1, help="v1: 1")
    ap.add_argument("--prefill-rows", type=int, default=None)
    ap.add_argument("--no-graphs", action="store_true", help="every window eagerly (T0b)")
    ap.add_argument("--pure", action="store_true", help="A = 0, no followers: the package's own _compute")
    ap.add_argument("--truncate", type=int, default=None, help="test only: the chain ends after L layers")
    ap.add_argument("--tap-layers", type=_taps, default=None, help="test only: DFlash2's taps after these layers")
    ap.add_argument("--no-warmup", action="store_true", help="skip the startup self-test (debugging only)")
    # leader: HTTP (TF/cli_args.py:27-56)
    ap.add_argument("--name", default="", help="the model id clients ask for (as in the reference /v1/models)")
    ap.add_argument("--alias", action="append", default=[])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default=None)
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--vision-urls", action="store_true")
    ap.add_argument("--bench", default="", help="a prompt set (JSONL) to run instead of serving HTTP")
    ap.add_argument("--bench-out", default="", help="the bench's replies (JSON)")
    return ap


def check_args(a: argparse.Namespace) -> None:
    first, end = a.layers
    if a.role == "follower":
        if not a.down:
            raise SystemExit("a follower needs --down HOST:PORT (the next stage towards the leader)")
        if first > 0 and not a.listen:
            raise SystemExit("a follower past layer 0 needs --listen HOST:PORT")
        for flag in ("drafter", "bench", "pure", "truncate", "tap_layers", "vision"):
            if getattr(a, flag):
                raise SystemExit(f"--{flag.replace('_', '-')} is the leader's")
        return
    if a.down:
        raise SystemExit("--down is a follower's: the leader is the chain's last stage")
    if first > 0 and not a.listen:
        raise SystemExit("a leader past layer 0 listens for the stage before it: --listen HOST:PORT")
    if a.pure and first != 0:
        raise SystemExit("--pure: --layers 0:END (one process)")
    if a.tap_layers and a.truncate is None:
        raise SystemExit("--tap-layers is for shortened test models (--truncate L)")
    if a.truncate is not None and end != a.truncate:
        raise SystemExit(f"--truncate {a.truncate}: the leader's layers end there (--layers A:{a.truncate})")
    if a.drafter and a.no_drafts:
        raise SystemExit("--drafter or --no-drafts, not both")
    if a.parallel != 1:
        raise SystemExit("tfpp-glm v1 serves --parallel 1 (v2: spec section 14)")
    if a.bench_out and not a.bench:
        raise SystemExit("--bench-out needs --bench")


def _leader(a: argparse.Namespace) -> int:
    from . import startup, trace
    from .tail import PipeGlmEngine, exit_code
    from tensorfold.families.glm5_next.cuda.engine import DEFAULT_POLICY

    first, end = a.layers
    link = None
    if first > 0:
        link = startup.leader_listen(a.listen, io_timeout=a.io_timeout, accept_timeout=a.connect_timeout)
    eng = None
    try:
        eng = PipeGlmEngine(Path(a.model), layers=(first, end), truncate=a.truncate, link=link,
                            policy=a.policy or DEFAULT_POLICY, drafter=a.drafter or None, context=a.context,
                            serial_only=a.no_drafts, prefill_rows=a.prefill_rows, vision=a.vision,
                            vision_urls=a.vision_urls, parallel=a.parallel, graphs=not a.no_graphs,
                            tap_layers=a.tap_layers, reserve_gib=a.reserve_gib, budget_gib=a.budget_gib,
                            io_timeout=a.io_timeout, startup_timeout=a.connect_timeout, pure=a.pure,
                            warmup=not a.no_warmup)
    except BaseException as exc:
        print(f"[tfpp-glm] leader: startup failed: {type(exc).__name__}: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        startup.leader_fail(link, "startup", exc)    # the followers stop too (exit 75)
        return exit_code(exc) if link is not None else 1
    tracer = trace.make(a.trace, "leader", a.layers)
    if tracer is not None:
        tracer.install_leader(eng)
    code = 0
    try:
        if a.bench:
            from . import bench

            bench.run(eng, bench.load_prompts(a.bench), model_dir=a.model, out_path=a.bench_out or None)
        else:
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


def _serve_http(eng, a: argparse.Namespace) -> None:
    from tensorfold.cli import _generation_config
    from tensorfold.cuda.server import serve
    from tensorfold.families.glm5_next.cuda.app import GlmApp

    model_dir = Path(a.model)
    sampling = _generation_config(model_dir)
    for key, value in (("temperature", a.temperature), ("top_p", a.top_p), ("top_k", a.top_k), ("min_p", a.min_p)):
        if value is not None:
            sampling[key] = value
    served = a.name or model_dir.name
    app = GlmApp(eng, model_dir, served, default_thinking=bool(a.thinking), sampling=sampling,
                 max_tokens=int(a.max_tokens), context_window=a.context, reasoning_effort=a.reasoning_effort,
                 thinking_budget=int(a.thinking_budget), aliases=list(a.alias))
    shown = "greedy" if float(sampling.get("temperature", 1.0)) <= 0 else ", ".join(f"{k} {v}" for k, v in sampling.items())
    print(f"[tfpp-glm] serving {served} at http://{a.host}:{a.port}/v1 (sampling: {shown}; drafts: "
          f"{'off' if a.no_drafts else 'on'}; context: {app.effective_context_window})", flush=True)
    serve(app, a.host, int(a.port))


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    check_args(a)
    from . import env, pins

    try:
        env.check_env(a.role)
        pins.check_tree()
        env.check_device()
    except (env.EnvError, pins.TreeMismatch) as exc:
        print(f"[tfpp-glm] {exc}", flush=True)
        return 1
    if a.role == "follower":
        from . import follower, trace

        tracer = trace.make(a.trace, "follower", a.layers)
        if tracer is not None:
            tracer.install_common()
        try:
            return follower.run(a, tracer)
        finally:
            if tracer is not None:
                tracer.close()
    return _leader(a)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)          # no interpreter teardown (CUDA graphs, pinned buffers, a KeepWarm thread)
