"""tfpp-glm v2: the multi-stream test harness (spec v2 sections 14.3, 15.3) -- a deterministic lockstep schedule, a free
(concurrent) run and an HTTP client. No torch: the classes here take the scheduler and the engine as arguments.

  leader:  --parallel 4 --mbench SCHEDULE.json [--bench-out OUT.json] [--trace OUT.jsonl]     run the schedule, QUIT
  client:  python -m tfpp.glm.v2bench http --url http://127.0.0.1:8888 [--concurrency 4] [--rounds 1] [--out OUT.json]

LOCKSTEP. MIA's scheduler decides everything from what is in its queue at the start of an iteration; a request thread that
is a millisecond late changes the chunks, the windows, the slots and the seal numbers. So the harness takes the iteration
entry over (``sch._iteration`` as an instance attribute: ``GlmScheduler._loop`` looks it up on every pass, G/multi.py:2662)
and lets a request in only at the iteration its schedule names: each request has a client thread waiting at a gate; the
hook opens the gate of every request whose ``at`` is due, waits until that request is in the scheduler's queue (the
engine's tagging ``put`` tells it: ``eng.v2landed``), then runs the scheduler's own iteration. When the scheduler is idle
and the next arrival is later, the iteration number jumps there. The same schedule on the same pool plan makes
the same sequence of messages, ids, slots, windows and (greedy or keyed) samples on a one-process run and on a chain.
A client-side stop (``stop_after``) becomes a stream-side cap on the worker (``eng.v2cancel``): deterministic too.

FREE. ``concurrency`` clients take the requests in turn (staggered by ``stagger_ms``), ``rounds`` passes of the set:
timings and aggregate throughput, replies compared by token against the lockstep run (level A: any timing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import bench as v1bench

HERE = Path(__file__).resolve().parent


# -- lockstep -------------------------------------------------------------------------------------------------------------
class Landings:
    """One Event a request tag: set when that request is in the scheduler's queue (or has failed before it got there).
    Only tags a harness registered (``Lockstep`` does) have one: ``signal`` of any other tag, every HTTP request's, is a
    no-op, so a long-running server collects nothing here; ``forget`` drops a finished harness' tags."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}

    def event(self, tag: str) -> threading.Event:
        with self._lock:
            ev = self._events.get(tag)
            if ev is None:
                ev = self._events[tag] = threading.Event()
            return ev

    def register(self, tag: str) -> threading.Event:
        return self.event(tag)

    def signal(self, tag: str) -> None:
        with self._lock:
            ev = self._events.get(tag)
        if ev is not None:
            ev.set()

    def forget(self, tag: str) -> None:
        with self._lock:
            self._events.pop(tag, None)


@dataclass
class Req:
    tag: str
    at: int                                   # the iteration whose start lets this request in
    run: Callable[[], Any]                    # what its client thread does (``engine.generate`` and the bookkeeping)
    cancel: int | None = None                 # informational: the stream-side cap the caller registered for this tag


class Lockstep:
    """The deterministic iteration hook (see the module docstring). ``sch``: a ``GlmScheduler`` (or anything with
    ``_iteration``, ``waiting``, ``held``, ``loading``, ``ready``, ``decoder.live()`` and a ``_Flush`` class);
    ``landed``: the engine's ``Landings``; ``write``: a callable for the ``iter`` records (the tracer's)."""

    def __init__(self, sch, reqs: list[Req], *, landed: Landings, write: Callable[[dict], None] | None = None,
                 wait_s: float = 120.0) -> None:
        tags = [r.tag for r in reqs]
        if len(set(tags)) != len(tags):
            raise ValueError("lockstep: request tags must be unique")
        self.sch = sch
        self.reqs = list(reqs)
        self.pending = sorted(reqs, key=lambda r: r.at)          # a stable sort: equal arrivals keep their order
        self.landed = landed
        self.write = write or (lambda rec: None)
        self.wait_s = float(wait_s)
        self.n = 0
        self.gate = {r.tag: threading.Event() for r in reqs}
        for r in reqs:
            landed.register(r.tag)
        self.results: dict[str, tuple[bool, Any]] = {}
        self.threads: list[threading.Thread] = []
        self.error: BaseException | None = None
        self.parked = threading.Event()       # the worker wrote its last ``iter`` record and waits for work

    def _idle(self) -> bool:
        sch = self.sch
        return (not sch.decoder.live() and sch.held is None and not getattr(sch, "loading", None)
                and not getattr(sch, "ready", None) and sch.waiting.empty())

    def _hook(self) -> None:
        """Replaces ``sch._iteration`` (worker thread)."""

        released: list[str] = []
        self.parked.clear()
        try:
            if self.pending and self._idle():
                self.n = max(self.n, self.pending[0].at)           # nothing can happen before the next arrival
            while self.pending and self.pending[0].at <= self.n:
                r = self.pending.pop(0)
                self.gate[r.tag].set()
                if not self.landed.event(r.tag).wait(self.wait_s):
                    raise TimeoutError(f"lockstep: request {r.tag!r} did not reach the scheduler's queue in "
                                       f"{self.wait_s:g}s")
                released.append(r.tag)
        except BaseException as exc:
            self.error = exc
            raise
        self.write({"t": "iter", "n": self.n, "released": released})
        if not self.pending and self._idle():
            self.parked.set()                                      # nothing more to write until new work comes
        try:
            type(self.sch)._iteration(self.sch)                    # the scheduler's own iteration
        finally:
            self.n += 1

    def _client(self, r: Req) -> None:
        self.gate[r.tag].wait()
        try:
            self.results[r.tag] = (True, r.run())
        except BaseException as exc:  # noqa: BLE001
            self.results[r.tag] = (False, f"{type(exc).__name__}: {exc}")
        finally:
            self.landed.signal(r.tag)                               # (also when it failed before the queue)

    def start(self) -> None:
        for r in self.reqs:
            t = threading.Thread(target=self._client, args=(r,), name=f"lockstep-{r.tag}", daemon=True)
            t.start()
            self.threads.append(t)
        self.sch._iteration = self._hook
        # a worker that idles in the original iteration (blocked on the queue) comes out of it on this item, which
        # changes no state (GlmScheduler._admit passes it by: G/multi.py:2586-2587), and picks the hook up next
        self.sch.waiting.put((self.sch._Flush, threading.Event()))

    def stop(self) -> None:
        self.sch.__dict__.pop("_iteration", None)

    def run(self, timeout: float = 3600.0, park_s: float = 30.0) -> dict[str, tuple[bool, Any]]:
        """Start, wait for every client, then for the worker to come to rest (its last ``iter`` record written, so a
        trace closed after this has the same tail every run), stop. Returns ``{tag: (ok, result or error text)}``."""

        deadline = time.monotonic() + timeout
        self.start()
        try:
            for t in self.threads:
                t.join(max(0.0, deadline - time.monotonic()))
            alive = [t.name for t in self.threads if t.is_alive()]
            if alive:
                raise TimeoutError(f"lockstep: {len(alive)} requests did not finish in {timeout:g}s: {alive[:5]}"
                                   + (f" (the hook: {self.error!r})" if self.error else ""))
            self.parked.wait(max(0.0, min(float(park_s), deadline - time.monotonic())))
        finally:
            self.stop()
            for r in self.reqs:
                self.landed.forget(r.tag)
        return self.results


# -- schedules ------------------------------------------------------------------------------------------------------------
def find_file(name: str, near: Path | None = None) -> Path:
    """A path as given, else beside the schedule file, else in this package's folder or its ``schedules``."""

    p = Path(name)
    for cand in ([p] if p.is_absolute() else [p] + ([near / p] if near else []) + [HERE / p, HERE / "schedules" / p]):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"{name}: not found (looked beside the schedule and in {HERE})")


def load_schedule(path: str | Path) -> dict:
    try:
        p = find_file(str(path))
    except FileNotFoundError:
        p = find_file(str(path) + ".json")                 # a bare name: AS1
    s = json.loads(p.read_text(encoding="utf-8"))
    mode = s.get("mode")
    if mode not in ("lockstep", "free"):
        raise ValueError(f"{p}: mode is lockstep or free, not {mode!r}")
    s.setdefault("name", p.stem)
    s.setdefault("prompts", "bench_prompts.jsonl")
    s["_dir"] = str(p.parent)
    if mode == "free":
        s["concurrency"] = int(s.get("concurrency", 1))
        s["stagger_ms"] = float(s.get("stagger_ms", 50))
        s["rounds"] = int(s.get("rounds", 1))
        if s["concurrency"] < 1 or s["rounds"] < 1:
            raise ValueError(f"{p}: concurrency and rounds are at least 1")
    else:
        a = s.get("arrivals", 0)
        if not (isinstance(a, int) or (isinstance(a, list) and all(isinstance(x, int) and x >= 0 for x in a))):
            raise ValueError(f"{p}: arrivals is an iteration number or a list of them")
    return s


def expand(items: list[dict], sched: dict) -> list[dict]:
    """The schedule's requests: ``{tag, id, rep, at, item}`` in the prompt set's order (``only`` picks ids; ``arrivals``
    is one iteration for all or one a prompt; a prompt's ``repeat`` makes that many requests arriving together)."""

    only = sched.get("only")
    sel = [it for it in items if not only or it["id"] in only]
    if only and len(sel) != len(set(only)):
        missing = sorted(set(only) - {it["id"] for it in sel})
        raise ValueError(f"schedule {sched.get('name')}: prompts {missing} are not in the set")
    arrivals = sched.get("arrivals", 0)
    if isinstance(arrivals, list) and len(arrivals) != len(sel):
        raise ValueError(f"schedule {sched.get('name')}: {len(arrivals)} arrivals for {len(sel)} prompts")
    out = []
    for i, it in enumerate(sel):
        at = arrivals[i] if isinstance(arrivals, list) else int(arrivals)
        reps = int(it.get("repeat", 1))
        for rep in range(reps):
            out.append({"tag": it["id"] if reps == 1 else f"{it['id']}#{rep}", "id": it["id"], "rep": rep, "at": at,
                        "item": it})
    return out


# -- running a request ---------------------------------------------------------------------------------------------------
def make_run(eng, ids: list[int], r: dict, tag: str, *, capped: bool) -> Callable[[], dict]:
    """The client thread's work for one request: ``engine.generate`` like ``bench.run`` does it, the timings added.
    ``capped``: the stop-after is the worker's (lockstep), not the client's."""

    item = r["item"]
    stop_after = int(item.get("stop_after", 0) or 0)

    def run() -> dict:
        got: list[int] = []
        first = [None]
        t = time.perf_counter()

        def on_tokens(toks):
            if first[0] is None:
                first[0] = time.perf_counter() - t
            got.extend(int(x) for x in toks)
            return bool(stop_after) and not capped and len(got) >= stop_after

        eng.request.policy = item.get("policy")
        eng.request.stop_eos = not bool(item.get("ignore_eos", False))
        eng.request.trace_tag = tag
        stats = eng.generate(ids, int(item.get("max_tokens", 128)), v1bench.sampling_of(item), on_tokens,
                             draft=bool(item.get("draft", True)))
        return {"id": r["id"], "rep": r["rep"], "tag": tag, "at": r["at"], "prompt_tokens": len(ids), "tokens": got,
                "token_sha": hashlib.sha256(json.dumps(got).encode()).hexdigest()[:16],
                "wall_s": round(time.perf_counter() - t, 3),
                "ttft_s": None if first[0] is None else round(first[0], 3), "stats": stats}

    return run


def run_lockstep(eng, requests: list[dict], ids: dict[str, list[int]], *, write=None, timeout: float = 3600.0) -> list:
    reqs = []
    for r in requests:
        cap = int(r["item"].get("stop_after", 0) or 0) or None
        if cap:
            eng.v2cancel[r["tag"]] = cap
        reqs.append(Req(r["tag"], r["at"], make_run(eng, ids[r["tag"]], r, r["tag"], capped=True), cancel=cap))
    lock = Lockstep(eng.scheduler, reqs, landed=eng.v2landed, write=write)
    results = lock.run(timeout)
    return [ok_or_error(results[r["tag"]], r) for r in requests]


def run_free(eng, requests: list[dict], ids: dict[str, list[int]], *, concurrency: int, stagger_ms: float,
             rounds: int) -> tuple[list, float]:
    import queue

    work: queue.Queue = queue.Queue()
    for rnd in range(rounds):
        for r in requests:
            work.put((rnd, r))
    out: list[Any] = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    def worker(k: int) -> None:
        time.sleep(k * stagger_ms / 1000.0)
        while True:
            try:
                rnd, r = work.get_nowait()
            except queue.Empty:
                return
            tag = f"{r['tag']}@{rnd}"
            again = dict(r, tag=tag, rep=r["rep"] + 1000 * rnd)       # (v2compare bench reads rep % 1000)
            try:
                res = (True, make_run(eng, ids[r["tag"]], again, tag, capped=False)())
            except BaseException as exc:  # noqa: BLE001
                res = (False, f"{type(exc).__name__}: {exc}")
            with lock:
                out.append(ok_or_error(res, again))

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, time.perf_counter() - t0


def ok_or_error(res: tuple[bool, Any], r: dict) -> dict:
    ok, val = res
    if ok:
        return val
    return {"id": r["id"], "rep": r["rep"], "tag": r["tag"], "at": r.get("at"), "tokens": [], "error": val}


def summarize(results: list[dict], wall_s: float) -> dict:
    """Aggregate and per-stream throughput, TTFT and the spread of the token intervals (a request's decode time over
    its tokens)."""

    done = [r for r in results if "error" not in r and r.get("tokens")]
    toks = sum(len(r["tokens"]) for r in done)
    per = [(len(r["tokens"]) - 1) / max(r["wall_s"] - (r["ttft_s"] or 0.0), 1e-9) for r in done
           if len(r["tokens"]) > 1 and r.get("ttft_s") is not None]
    gaps = sorted((r["wall_s"] - r["ttft_s"]) / (len(r["tokens"]) - 1) for r in done
                  if len(r["tokens"]) > 1 and r.get("ttft_s") is not None)

    def q(p: float):
        return round(gaps[min(len(gaps) - 1, int(p * len(gaps)))] * 1e3, 1) if gaps else None

    return {"requests": len(results), "failed": len(results) - len(done), "tokens": toks, "wall_s": round(wall_s, 3),
            "aggregate_tok_s": round(toks / wall_s, 2) if wall_s > 0 else None,
            "stream_tok_s_mean": round(statistics.fmean(per), 2) if per else None,
            "ttft_s_mean": round(statistics.fmean([r["ttft_s"] for r in done if r.get("ttft_s") is not None]), 3)
            if any(r.get("ttft_s") is not None for r in done) else None,
            "token_interval_ms_p50": q(0.5), "token_interval_ms_p95": q(0.95)}


def run_mbench(eng, a, *, tracer=None, log=print) -> list[dict]:
    """``--mbench``: the schedule's requests through the leader's scheduler, the replies written to ``--bench-out``
    (a list shaped like ``bench.run``'s, one entry a request, plus ``tag`` and the timings)."""

    sched = load_schedule(a.mbench)
    prompts = find_file(sched["prompts"], Path(sched["_dir"]))
    items = v1bench.load_prompts(prompts)
    requests = expand(items, sched)
    renderer = v1bench.Renderer(a.model)
    ids: dict[str, list[int]] = {}
    for r in requests:
        ids[r["tag"]] = renderer.ids(r["item"])
    log(f"[tfpp-glm] mbench {sched['name']} ({sched['mode']}): {len(requests)} requests from {prompts.name}")
    write = tracer.write if tracer is not None else None
    t0 = time.perf_counter()
    if sched["mode"] == "lockstep":
        results = run_lockstep(eng, requests, ids, write=write)
        free = None
    else:
        results, _ = run_free(eng, requests, ids, concurrency=sched["concurrency"], stagger_ms=sched["stagger_ms"],
                              rounds=sched["rounds"])
        free = {"concurrency": sched["concurrency"], "rounds": sched["rounds"]}
    wall = time.perf_counter() - t0
    if getattr(eng, "broken", None):
        raise RuntimeError(f"the chain broke during the schedule: {eng.broken}")
    summary = summarize(results, wall)
    if free:
        summary.update(free)
    for r in results:
        if "error" in r:
            log(f"[tfpp-glm] mbench {r['tag']}: failed ({r['error']})")
        else:
            log(f"[tfpp-glm] mbench {r['tag']}: {r['prompt_tokens']} prompt + {len(r['tokens'])} tokens in "
                f"{r['wall_s']:.2f}s, ttft {r['ttft_s']}")
    log(f"[tfpp-glm] mbench {sched['name']}: {json.dumps(summary)}")
    if getattr(a, "bench_out", ""):
        Path(a.bench_out).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        Path(str(a.bench_out) + ".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return results


# -- an HTTP client -------------------------------------------------------------------------------------------------------
def http_request(base: str, model: str, item: dict, timeout: float) -> dict:
    """One streaming chat request; TTFT at the first content delta, tokens from the usage block (else the deltas)."""

    messages = list(item.get("messages") or [{"role": "user", "content": item["text"] * int(item.get("text_repeat", 1))}])
    if item.get("system"):
        messages = [{"role": "system", "content": item["system"] * int(item.get("system_repeat", 1))}] + messages
    body = {"model": model, "messages": messages, "max_tokens": int(item.get("max_tokens", 128)), "stream": True,
            "temperature": float(item.get("temperature", 0.0) or 0.0),
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": bool(item.get("thinking", False))}}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, deltas, usage, text = None, 0, None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                try:
                    obj = json.loads(line[5:])
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
                    if piece:
                        deltas += 1
                        text.append(piece)
                        if first is None:
                            first = time.perf_counter() - t0
    except (urllib.error.URLError, OSError) as exc:
        return {"id": item["id"], "error": f"{type(exc).__name__}: {exc}"}
    wall = time.perf_counter() - t0
    n = int(usage["completion_tokens"]) if usage and "completion_tokens" in usage else deltas
    return {"id": item["id"], "tokens": n, "wall_s": round(wall, 3), "ttft_s": None if first is None else round(first, 3),
            "text_sha": hashlib.sha256("".join(text).encode()).hexdigest()[:16]}


def main_http(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m tfpp.glm.v2bench http", description="stream the prompt set to a server")
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="")
    ap.add_argument("--prompts", default="bench_prompts.jsonl")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--stagger-ms", type=float, default=50.0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--only", default="", help="comma separated prompt ids")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    items = v1bench.load_prompts(find_file(a.prompts))
    if a.only:
        want = set(a.only.split(","))
        items = [i for i in items if i["id"] in want]
    model = a.model
    if not model:
        with urllib.request.urlopen(a.url.rstrip("/") + "/v1/models", timeout=30) as resp:
            model = json.load(resp)["data"][0]["id"]
    import queue

    work: queue.Queue = queue.Queue()
    for rnd in range(a.rounds):
        for it in items:
            work.put((rnd, it))
    out: list[dict] = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    def worker(k: int) -> None:
        time.sleep(k * a.stagger_ms / 1000.0)
        while True:
            try:
                rnd, it = work.get_nowait()
            except queue.Empty:
                return
            res = http_request(a.url, model, it, a.timeout)
            res["round"] = rnd
            with lock:
                out.append(res)

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(max(1, a.concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    done = [r for r in out if "error" not in r]
    toks = sum(r["tokens"] for r in done)
    summary = {"requests": len(out), "failed": len(out) - len(done), "tokens": toks, "wall_s": round(wall, 3),
               "aggregate_tok_s": round(toks / wall, 2) if wall > 0 else None,
               "ttft_s_mean": round(statistics.fmean([r["ttft_s"] for r in done if r.get("ttft_s") is not None]), 3)
               if any(r.get("ttft_s") is not None for r in done) else None}
    print(json.dumps(summary))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "results": out}, indent=1), encoding="utf-8")
    return 0 if not summary["failed"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "http":
        return main_http(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
