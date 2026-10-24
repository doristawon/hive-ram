"""Traces for the equivalence tests (spec 13.3): what each request did, round by round, as exact device hashes.

Leader records (compared by ``compare``): ``req`` (a request's prompt and options), ``chunk`` (a prompt chunk: start,
rows, the head row's hash, the taps' hashes), ``step`` (a verify window: position, rows, its tokens = the pending
token and the drafts, every logits row's hash, the taps' hashes), ``keep`` (the rows a window kept), ``done``
(the reply's tokens and the stats that are not timings). Followers record ``chunk`` / ``act`` (the streams they send
on), to find the stage where two runs part. Hooks wrap ``TailEngine.forward`` and ``decode.prefill_chunk`` outside
everything else, so a CUDA graph replay is traced as well as an eager window; hashing syncs the GPU (tracing only).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

from . import part

TIMING_KEYS = ("prefill_s", "decode_s", "stages_ms", "encode_s", "ms", "wall_ms")


class Tracer:
    def __init__(self, path: str | Path, role: str, layers) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
        self.lock = threading.Lock()
        self.on = True
        self.req = 0
        self.write({"t": "meta", "role": role, "layers": list(layers), "time": time.time()})

    def write(self, rec: dict) -> None:
        if not self.on:
            return
        with self.lock:
            self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.f.flush()

    def close(self) -> None:
        with self.lock:
            self.f.close()

    # -- stage-local helpers -------------------------------------------------------------------------------------
    @staticmethod
    def _taps(b, R: int) -> list:
        S = part.STAGE
        slots = range(len(b.taps)) if S is None else S.valid_taps
        return [[s, part.devhash(b.taps[s][:R])] for s in slots]

    def act(self, kind: str, pos: int, R: int, b) -> None:
        self.write({"t": "act", "kind": kind, "pos": pos, "R": R, "x": part.devhash(b.x[:R]),
                    "taps": [[s, part.devhash(b.taps[s][:R])] for s in part.STAGE.carry_out]})

    # -- hooks ---------------------------------------------------------------------------------------------------
    def install_common(self) -> None:
        from tensorfold.families.glm5_next.cuda import decode as D

        orig = D.prefill_chunk
        tracer = self

        def prefill_chunk(e, prompt, start, R, **kw):
            out = orig(e, prompt, start, R, **kw)
            if tracer.on:
                tracer.write({"t": "chunk", "start": int(start), "R": int(R),
                              "head": part.devhash(out) if out is not None else None,
                              "taps": tracer._taps(e.pbuf, R)})
            return out

        D.prefill_chunk = prefill_chunk

    def install_leader(self, eng) -> None:
        """After the seams (outermost): the windows, the keeps, and each request's frame of records."""

        from tensorfold.families.glm5_next.cuda import decode as D, drafter_choice as DC

        from .tail import TailEngine

        self.install_common()
        tracer = self
        fwd = TailEngine.forward

        def forward(e, tokens):
            t = time.perf_counter()
            out = fwd(e, tokens)
            if tracer.on and e is eng.e:
                R = len(tokens)
                tracer.write({"t": "step", "pos": int(e.st.pos), "R": R, "window": [int(x) for x in tokens],
                              "logits": part.devhash_rows(out[:R]), "taps": tracer._taps(e.buf, R),
                              "wall_ms": round((time.perf_counter() - t) * 1e3, 3)})
            return out

        TailEngine.forward = forward
        commit = D.commit

        def traced_commit(w, st, b, R, keep):
            if tracer.on and not b.prefill and b is eng.e.buf:
                tracer.write({"t": "keep", "pos": int(st.pos), "R": int(R), "keep": int(keep)})
            return commit(w, st, b, R, keep)

        D.commit = DC.commit = traced_commit
        gen = eng.generate

        def generate(prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None, *, vision=None,
                     background=False):
            tracer.req += 1
            got: list[int] = []
            ids = [int(t) for t in prompt]
            tracer.write({"t": "req", "n": tracer.req, "prompt_len": len(ids),
                          "prompt_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
                          "max_tokens": int(max_tokens), "draft": bool(draft),
                          "policy": getattr(eng.request, "policy", None),
                          "sampling": None if sampling is None else [sampling.seed, sampling.temperature,
                                                                     sampling.top_k, sampling.top_p, sampling.min_p],
                          "constraint": constraint is not None, "vision": vision is not None})

            def on(toks):
                got.extend(int(x) for x in toks)
                return on_tokens(toks) if on_tokens is not None else False

            try:
                stats = gen(prompt, max_tokens, sampling, on, draft, constraint, vision=vision, background=background)
            except BaseException as exc:
                tracer.write({"t": "fail", "n": tracer.req, "error": f"{type(exc).__name__}: {exc}"[:500]})
                raise
            tracer.write({"t": "done", "n": tracer.req, "tokens": got,
                          "stats": {k: v for k, v in (stats or {}).items() if k not in TIMING_KEYS}})
            return stats

        eng.generate = generate


def make(path: str | None, role: str, layers):
    return Tracer(path, role, layers) if path else None
