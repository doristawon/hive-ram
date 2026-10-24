"""Traces for the equivalence tests (spec 15.6): what each request did, chunk by chunk and round by round, as exact
device hashes. The same hooks go on the chain's leader and on the reference (ref.py: the unmodified DsEngine), as
instance attributes, after the warm-up:

  engine.generate         req (prompt sha, length, options), done (the reply's tokens; rounds / drafted / accepted)
  multi._admit            admit (sid, slot, extent)
  multi._fill             which stream's chunk the next model.forward runs
  engine.model.forward    chunk (sid, start, rows, the returned logits' hash, the taps' hash)
  multi.runner.forward    step (round, bucket, rows [sid, slot, P, R], tokens, every logits row's hash, taps rows --
                          the taps only on an engine with a drafter: nothing else reads them)
  multi._round            keep (each stream's kept drafts)
Followers record ``act`` (every part of every ACT they send, hashed) to find the stage where two runs part. Hashing
syncs the GPU (tracing only); compare.py reads the files.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

TIMING_KEYS = ("prefill_s", "decode_s", "tokens_per_second", "draft_ms", "verify_ms", "absorb_ms", "ms", "wall_ms",
               "started", "finished")

_INT = None


def _ints(t):
    import torch

    global _INT
    if _INT is None:
        _INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return t.contiguous().view(_INT[t.element_size()])


def _weights(n: int, device):
    import torch

    return torch.arange(n, device=device, dtype=torch.int64) % 65521 + 1


def devhash(t) -> list[int]:
    """[sum, weighted sum] of a tensor's raw words as integers, on its device (copied from G/part.py:117-125)."""

    import torch

    if t.dtype == torch.bool:
        t = t.view(torch.uint8)
    v = _ints(t).reshape(-1).to(torch.int64)
    if v.numel() == 0:
        return [0, 0]
    s = torch.stack([v.sum(), (v * _weights(v.numel(), v.device)).sum()])
    return [int(x) for x in s.tolist()]


def devhash_rows(t) -> list[list[int]]:
    """``devhash`` of each row of a 2-D tensor, one host sync (copied from G/part.py:128-138)."""

    import torch

    if t.dtype == torch.bool:
        t = t.view(torch.uint8)
    if t.dim() != 2:
        t = t.reshape(t.shape[0], -1)
    v = _ints(t).to(torch.int64)
    if v.numel() == 0:
        return [[0, 0] for _ in range(t.shape[0])]
    wgt = _weights(v.shape[1], v.device)
    s = torch.stack([v.sum(dim=1), (v * wgt).sum(dim=1)], dim=1)
    return [[int(a), int(b)] for a, b in s.tolist()]


def prompt_sha(ids) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:16]


def run_meta(model_dir, *, context=None, parallel=None, drafts=None, graphs=None) -> dict:
    """What a run was made under, for compare.py's first check: runs whose settings differ are not comparable. The settings
    vector (the modules' values, the env digest, the model/config/selection identities, experts, cut_post), the versions
    (python, torch, CUDA, Triton, the fork's and tfpp-ds41's sources) and the run's shape. Call it after the fork's modules
    are imported (env.check_env first)."""

    from . import env, pins

    meta = {"settings": env.settings_vector(model_dir, experts_k=env.experts(), carry=env.cut_post()),
            "versions": env.versions(pins.EXPECT_PKG_SHA)}
    for key, value in (("context", context), ("parallel", parallel), ("drafts", drafts), ("graphs", graphs)):
        if value is not None:
            meta[key] = value
    return meta


class Tracer:
    def __init__(self, path: str | Path, role: str, layers, meta: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
        self.lock = threading.Lock()
        self.on = True
        self.req = 0
        self.rounds = 0
        self.cur_sid = None
        rec = {"t": "meta", "role": role, "layers": list(layers), "time": time.time()}
        if meta is not None:
            rec["run"] = meta
        self.write(rec)

    def write(self, rec: dict) -> None:
        if not self.on:
            return
        with self.lock:
            self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.f.flush()

    def close(self) -> None:
        with self.lock:
            self.f.close()

    # -- followers -------------------------------------------------------------------------------------------------
    def act(self, kind: str, fields: dict, parts) -> None:
        rec = {"t": "act", "kind": kind}
        rec.update({k: v for k, v in fields.items() if k in ("start0", "n0", "start", "n", "floor", "round", "R", "b")})
        rec["parts"] = [[name, devhash(t)] for name, t in parts]
        self.write(rec)

    # -- the leader / the reference ----------------------------------------------------------------------------
    def install(self, eng) -> None:
        """Hooks on one engine (a PipeDsEngine or the fork's DsEngine) after its warm-up."""

        tracer, mu, model = self, eng.multi, eng.model
        if mu is None:
            raise RuntimeError("tracing needs the --parallel path (MultiDecoder): --parallel 2 or more")

        orig_admit = mu._admit

        def _admit(s, index, positions, reuse=None):
            out = orig_admit(s, index, positions, reuse)
            tracer.write({"t": "admit", "sid": int(s.sid), "slot": int(s.st.index), "base": int(s.base),
                          "size": int(s.size), "prompt_sha": prompt_sha(s.prompt), "prompt_len": len(s.prompt)})
            return out

        mu._admit = _admit
        orig_fill = mu._fill

        def _fill(s):
            tracer.cur_sid = int(s.sid)
            try:
                return orig_fill(s)
            finally:
                tracer.cur_sid = None

        mu._fill = _fill
        orig_fwd = model.forward

        def forward(sc, ids, start, all_logits=False, taps=None, host_ids=None, replay=None, image=None):
            out = orig_fwd(sc, ids, start, all_logits=all_logits, taps=taps, host_ids=host_ids, replay=replay,
                           image=image)
            if tracer.on and tracer.cur_sid is not None:
                import torch

                rec = {"t": "chunk", "sid": tracer.cur_sid, "start": int(start), "rows": int(ids.shape[0]),
                       "logits": devhash(out) if out is not None else None}
                if taps:
                    rec["taps"] = devhash(torch.cat(taps, -1))
                tracer.write(rec)
            return out

        model.forward = forward
        runner = mu.runner
        orig_rf = runner.forward
        # the taps are the drafter's input and nothing else reads them: without a drafter they are not part of a reply,
        # and a --no-drafts leader that starts past a tap layer holds zeros for the upstream taps (decode._seed_taps)
        # where the one-process reference computes them, so a round's taps are recorded only when the engine drafts
        with_taps = getattr(eng, "drafter", None) is not None

        def rforward(windows, replay=True):
            t = time.perf_counter()
            logits, taps = orig_rf(windows, replay)
            if tracer.on and replay and logits is not None:
                tracer.rounds += 1
                by_slot = {s.st.index: s.sid for s in mu.streams.values() if s.st is not None}
                rows = [[int(by_slot.get(slot, -1)), int(slot), int(p0), len(toks)] for slot, _, _, p0, toks, _
                        in windows]
                R = sum(r[3] for r in rows)
                rec = {"t": "step", "round": tracer.rounds, "b": int(runner.bucket(max(r[2] + r[3] for r in rows))),
                       "rows": rows, "tokens": [int(x) for *_, toks, _ in windows for x in toks],
                       "logits": devhash_rows(logits[:R]), "wall_ms": round((time.perf_counter() - t) * 1e3, 3)}
                if taps is not None and with_taps:
                    rec["taps"] = devhash_rows(taps[:R])
                tracer.write(rec)
            return logits, taps

        runner.forward = rforward
        orig_round = mu._round

        def _round(live, k):
            before = {s.sid: s.st.sc.length for s in live}
            done = orig_round(live, k)
            if tracer.on:
                kept = [[int(s.sid), int(s.st.sc.length - before[s.sid] - 1)] for s in live
                        if s.st is not None and s.st.sc is not None]
                tracer.write({"t": "keep", "round": tracer.rounds, "k": int(k), "kept": kept})
            return done

        mu._round = _round
        gen = eng.generate

        def generate(prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None, vision=None, **kw):
            with tracer.lock:
                tracer.req += 1
                n = tracer.req
            got: list[int] = []
            ids = [int(t) for t in prompt]
            tracer.write({"t": "req", "n": n, "prompt_len": len(ids), "prompt_sha": prompt_sha(ids),
                          "max_tokens": int(max_tokens), "draft": bool(draft),
                          "stop_eos": bool(getattr(eng.request, "stop_eos", True)),
                          "sampling": None if sampling is None else [sampling.seed, sampling.temperature,
                                                                     sampling.top_k, sampling.top_p, sampling.min_p]})

            def on(toks):
                got.extend(int(x) for x in toks)
                return on_tokens(toks) if on_tokens is not None else False

            try:
                stats = gen(prompt, max_tokens, sampling, on, draft, constraint, vision, **kw)
            except BaseException as exc:
                tracer.write({"t": "fail", "n": n, "prompt_sha": prompt_sha(ids),
                              "error": f"{type(exc).__name__}: {exc}"[:500]})
                raise
            tracer.write({"t": "done", "n": n, "prompt_sha": prompt_sha(ids), "tokens": got,
                          "stats": {k: v for k, v in (stats or {}).items() if k in ("rounds", "drafted", "accepted")}})
            return stats

        eng.generate = generate


def make(path: str | None, role: str, layers, meta: dict | None = None):
    return Tracer(path, role, layers, meta) if path else None
