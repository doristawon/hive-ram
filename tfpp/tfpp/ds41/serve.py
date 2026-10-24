"""tfpp-ds41's entry point: one stage of the chain (spec 14.1).

  leader:   python -m tfpp.ds41.serve --role leader --model DIR --layers A:L [--listen H:P (A > 0)] [--engram DIR]
            [--context N] [--parallel P] [--mtp-drafts K | --no-drafts] [--no-graphs] [--experts K]
            [--name ... --host 0.0.0.0 --port 18891 --max-tokens N --thinking|--no-thinking --temperature ...
             --top-p ... --top-k ...] [--reserve-gib 3] [--budget-gib X] [--io-timeout 600] [--trace OUT.jsonl]
            [--bench P.jsonl [--bench-out O.json] [--oracle REF.json] [--bench-max-prompt N]]   no HTTP; QUIT after
  follower: python -m tfpp.ds41.serve --role follower --model DIR --layers A:B --down H:P [--listen H:P] [--engram DIR]
            [--budget-gib X] [--reserve-gib 3] [--io-timeout 600] [--connect-timeout 3600] [--trace OUT.jsonl]

``--role leader --layers 0:L`` alone is a one-stage chain (T0a: the stage classes without a cut). The followers take
the context, --parallel, the drafts, the chunk and --no-graphs from the leader (GO), the pool from AGREE. Exit codes:
0 QUIT, 75 the chain broke (restart every stage), 1 other. The parser runs without torch (tests); everything else is
imported after env.check_env.
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tfpp.ds41.serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=("leader", "follower"), required=True)
    ap.add_argument("--model", required=True, help="the checkpoint folder (config, tokenizer, this stage's shards)")
    ap.add_argument("--layers", type=_layers, required=True, help="A:B, this stage's layers [A, B)")
    ap.add_argument("--listen", default="", help="HOST:PORT the stage before this one connects to (A > 0)")
    ap.add_argument("--engram", default="", help="a folder holding exactly the Engram tables of the layers this "
                    "stage owns")
    ap.add_argument("--reserve-gib", type=float, default=float(os.environ.get("TFPP_RESERVE_GIB", "") or 3.0))
    ap.add_argument("--budget-gib", type=float, default=None, help="the stage's budget outright (several stages on "
                    "one GPU)")
    ap.add_argument("--io-timeout", type=float, default=600.0)
    ap.add_argument("--connect-timeout", type=float, default=3600.0)
    ap.add_argument("--trace", default="")
    # follower
    ap.add_argument("--down", default="", help="follower: HOST:PORT of the next stage (towards the leader)")
    # leader: engine
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--parallel", type=int, default=2)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mtp-drafts", type=int, default=5)
    g.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--no-graphs", action="store_true", help="every round eagerly (T0b)")
    ap.add_argument("--experts", type=int, default=0, help="test proxy: the first K routed experts a layer (T-X0)")
    ap.add_argument("--no-self-check", action="store_true", help="skip spec 11.4's self-check (debugging only)")
    # leader: HTTP (TF/cli.py:296-323)
    ap.add_argument("--name", default="")
    ap.add_argument("--alias", action="append", default=[])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18891)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    # leader: bench
    ap.add_argument("--bench", default="", help="a prompt set (JSONL) to run instead of serving HTTP")
    ap.add_argument("--bench-out", default="")
    ap.add_argument("--bench-max-prompt", type=int, default=0)
    ap.add_argument("--oracle", default="", help="T1-oracle: a serial run's --bench-out (the drafts' tokens)")
    return ap


def check_args(a: argparse.Namespace) -> None:
    first, end = a.layers
    if a.role == "follower":
        if not a.down:
            raise SystemExit("a follower needs --down HOST:PORT (the next stage towards the leader)")
        if first > 0 and not a.listen:
            raise SystemExit("a follower past layer 0 needs --listen HOST:PORT")
        for flag in ("bench", "oracle", "bench_out"):
            if getattr(a, flag):
                raise SystemExit(f"--{flag.replace('_', '-')} is the leader's")
        return
    if a.down:
        raise SystemExit("--down is a follower's: the leader is the chain's last stage")
    if first > 0 and not a.listen:
        raise SystemExit("a leader past layer 0 listens for the stage before it: --listen HOST:PORT")
    if a.parallel < 1:
        raise SystemExit("--parallel 1 or more")
    if not a.no_drafts and not 1 <= a.mtp_drafts <= 15:
        raise SystemExit("--mtp-drafts 1 .. 15 (a verify window of drafts + 1 rows within the 16-row rounds)")
    if (a.bench_out or a.oracle or a.bench_max_prompt) and not a.bench:
        raise SystemExit("--bench-out / --oracle / --bench-max-prompt need --bench")
    if a.oracle and a.no_drafts:
        raise SystemExit("--oracle replaces the drafts: not with --no-drafts")


def log(msg: str) -> None:
    print(f"[tfpp-ds41] {msg}", flush=True)


def _leader(a: argparse.Namespace) -> int:
    from tensorfold.families.deepseek_v41.cuda import engine as E

    from . import capacity, env, plan as PL, proto, startup, trace
    from .engine import PipeDsEngine
    from .warm_gate import GatedKeepWarm
    from .wire import LinkError

    first, end = a.layers
    link = None
    eng = None
    try:
        if first > 0:
            link = startup.leader_listen(a.listen, io_timeout=a.io_timeout, accept_timeout=a.connect_timeout)
        mv = PL.ModelView.read(a.model)
        drafts = 0 if a.no_drafts else int(a.mtp_drafts)
        experts, carry = env.experts(), env.cut_post()
        extra = startup.stage_extra(a.model, a.engram or None, mv)
        own = startup.stage_info((first, end), "leader", a.model, experts=experts, carry=carry, extra=extra)
        infos = startup.leader_hello(link, own, timeout=a.connect_timeout)
        problems = proto.compare_stages(infos)
        ranges = [list(i.layers) for i in infos]
        try:
            plans = PL.plan_chain(mv, ranges, drafts=drafts > 0)
        except PL.PlanError as exc:
            problems.append(str(exc))
            plans = None
        problems += proto.check_engram(ranges, mv.engram_layers, [i.extra for i in infos])
        if problems:
            raise RuntimeError("HELLO: the chain does not fit together:\n  " + "\n  ".join(problems))
        P = plans[-1]
        for p in plans:
            log(p.summary())
        go = {"stages": ranges, "n_layers": mv.n_layers, "context": int(a.context), "parallel": int(a.parallel),
              "drafts": drafts, "chunk": int(E.PREFILL_CHUNK), "replay": bool(E.REPLAY), "experts": experts,
              "cut_post": carry, "plan_digest": PL.chain_digest(plans), "graphs": not a.no_graphs}
        if link is not None:
            link.send(proto.GO, go)
        gate = GatedKeepWarm()
        eng = PipeDsEngine(a.model, plan=P, mv=mv, role="leader", up=link, drafts=drafts, context=a.context,
                           parallel=a.parallel, graphs=not a.no_graphs, experts=experts, engram_dir=a.engram or None,
                           gate=gate, io_timeout=a.io_timeout)
        cap_positions = int(a.context) + drafts + 1 + 8
        f1 = startup.f1_chunk_now(carry, mv)
        need = capacity.stage_need(mv, P, cap_positions, a.parallel, drafter=drafts > 0, chunk=E.PREFILL_CHUNK, f1=f1)
        rx, _ = capacity.stage_staging(mv, P, E.PREFILL_CHUNK, cap_positions, f1=f1)
        budget = capacity.measure(reserve_gib=a.reserve_gib, budget_gib=a.budget_gib, staging=need["staging"])
        own_cap = {"layers": [first, end], "pool": need["pool"], "workspace": need["workspace"],
                   "staging": need["staging"], "need": need["need"], "weights": eng.weights_bytes, **budget}
        caps = startup.leader_collect(link, proto.CAP, "caps", own_cap, timeout=a.connect_timeout)
        problems = capacity.check_caps(caps)
        if problems:
            raise RuntimeError("CAP: a stage cannot hold its share:\n  " + "\n  ".join(problems))
        agree = {"cap": cap_positions, "slots": int(a.parallel), "context": int(a.context), "drafts": drafts}
        if link is not None:
            link.send(proto.AGREE, agree)
            link.set_staging(recv_bytes=rx)
        eng.staging_bytes = rx
        eng.build(agree)
        mine = eng.built_info()
        mine["declared"] = a.budget_gib is not None
        built = startup.leader_collect(link, proto.READY, "built", mine, timeout=a.connect_timeout)
        problems = capacity.check_built(built)
        if problems:
            raise RuntimeError("READY: the stages built differently:\n  " + "\n  ".join(problems))
        eng.start()
        if not a.no_self_check:
            eng.self_check()
        if eng.ops is not None:
            eng.ops.armed = True                         # TFPP_FAULT counts the requests after the self-check
    except BaseException as exc:  # noqa: BLE001
        log(f"leader: startup failed: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        startup.leader_fail(link, "startup", exc)        # the followers stop too (exit 75)
        broken = isinstance(exc, (LinkError, proto.ProtocolError)) and link is not None
        return proto.FAIL_EXIT if broken else 1
    tracer = trace.make(a.trace or None, "leader", a.layers,
                        trace.run_meta(a.model, context=int(a.context), parallel=int(a.parallel), drafts=drafts,
                                       graphs=bool(E.GRAPHS) and not a.no_graphs) if a.trace else None)
    if tracer is not None:
        tracer.install(eng)
    code = 0
    try:
        if a.oracle:
            from . import oracle

            oracle.install(eng, oracle.load_oracle(a.oracle), log=log)
        if a.bench:
            from . import bench

            bench.run(eng, bench.load_prompts(a.bench), model_dir=a.model, out_path=a.bench_out or None,
                      context=a.context, max_prompt=a.bench_max_prompt or None)
        else:
            _serve_http(eng, a)
    except BaseException as exc:  # noqa: BLE001
        if not isinstance(exc, KeyboardInterrupt):
            log(f"leader: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            code = proto.FAIL_EXIT if isinstance(exc, (LinkError, proto.ProtocolError)) else 1
    finally:
        eng.quit()                                       # QUIT up the chain (nothing once it broke)
        if tracer is not None:
            tracer.close()
    if eng.broken:
        return proto.FAIL_EXIT
    return code


def _serve_http(eng, a: argparse.Namespace) -> None:
    from types import SimpleNamespace

    from tensorfold.cli import _generation_config
    from tensorfold.cuda.server import serve
    from tensorfold.families.deepseek_v41.cuda.app import DsApp
    from tensorfold.server.authentication import configure

    model_dir = Path(a.model)
    sampling = _generation_config(model_dir)
    for key, value in (("temperature", a.temperature), ("top_p", a.top_p), ("top_k", a.top_k), ("min_p", a.min_p)):
        if value is not None:
            sampling[key] = value
    served = a.name or model_dir.name
    app = DsApp(eng, model_dir, served, default_thinking=bool(a.thinking), sampling=sampling,
                max_tokens=int(a.max_tokens), context_window=a.context, reasoning_effort=a.reasoning_effort,
                thinking_budget=int(a.thinking_budget), aliases=list(a.alias))
    # the API key: TF_API_KEY / TF_API_KEY_FILE (the CUDA server's own gate checks it on every route but /health)
    app.auth = configure(SimpleNamespace(host=a.host, api_key=[], api_key_file=None, metrics_open=False))
    shown = "greedy" if float(sampling.get("temperature", 1.0)) <= 0 else ", ".join(f"{k} {v}" for k, v in
                                                                                    sampling.items())
    log(f"serving {served} at http://{a.host}:{a.port}/v1 (sampling: {shown}; drafts: "
        f"{'off' if a.no_drafts else a.mtp_drafts}; --parallel {a.parallel}; context: {app.effective_context_window})")
    serve(app, a.host, int(a.port))


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    check_args(a)
    if a.experts:
        have = os.environ.get("TFPP_DS41_EXPERTS")
        if have and int(have) != a.experts:
            raise SystemExit(f"TFPP_DS41_EXPERTS={have} and --experts {a.experts}")
        os.environ["TFPP_DS41_EXPERTS"] = str(a.experts)        # before check_env: the env digest takes it
    from . import env, pins

    try:
        env.check_env(a.role)
        pins.check_tree()
        pins.check_seams()
        env.check_device()
    except (env.EnvError, pins.TreeMismatch) as exc:
        log(str(exc))
        return 1
    if a.role == "follower":
        from . import follower, trace

        tracer = trace.make(a.trace or None, "follower", a.layers, trace.run_meta(a.model) if a.trace else None)
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
    os._exit(code)          # no interpreter teardown (CUDA graphs, pinned buffers, the KeepWarm thread)
