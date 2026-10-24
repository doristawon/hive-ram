"""tfpp-glm v2 traces for the equivalence tests (spec v2 section 15.1): what a --parallel leader did, message by message,
as exact device hashes. ``v1``'s ``trace.Tracer`` is the writer (a JSONL file, a lock); this module adds the records and
the hooks. Hooks wrap the leader's seams from outside (instance attributes and module attributes), so nothing in MIA or
in tfpp-glm's v1 modules changes, and hashing syncs the GPU (tracing only).

Leader records, written by the worker thread (compared in order by ``v2compare strict``):
  meta2   the pinned parameters (a comparison across other values is refused)
  iter    the lockstep hook's iteration number and the requests it let in
  msg     one sealed message: its sequence number, checksum, and its ops (name, payload length)
  admit   a stream placed: slot, extent, kept prompt resumed from, policy code, cached tokens, tag
  fill    a prompt chunk: slot, start, rows, the head row's hash, the taps' hashes
  pieces  a multi-prompt chunk: its pieces [slot, pos, rows], each head's hash, the taps' hashes
  round   a verify window: the message that carried it, its segments [slot, pos, rows], the stream ids, the window's
          token ids, every logits row's hash, the taps' hashes
  keeps   the rows each segment keeps and where it ends
  pool    the pool, kept prompts and lanes at the end of an iteration
  idle    the pool when the leader tells the stages to idle (a follower writes the same when it applies IDLE)
Client records (compared by tag): ``req`` (a request's prompt and options), ``done`` (the reply's tokens and the stats
that are not timings), ``fail``. A follower writes ``msg``, ``idle`` and ``act`` (the rows it sends on).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading

from tensorfold.families.glm5_next.cuda import decode as D, multi as M, multi_prefill

from . import part, v2proto
from .trace import TIMING_KEYS, Tracer

WORKER_KINDS = ("meta2", "iter", "msg", "admit", "fill", "pieces", "round", "keeps", "pool", "idle")
CLIENT_KINDS = ("req", "done", "fail")


def make(path: str | None, role: str, layers):
    return Tracer(path, role, layers) if path else None


def _sha(values) -> str:
    return hashlib.sha256(json.dumps([int(v) for v in values]).encode()).hexdigest()[:16]


def pool_summary(multi) -> dict:
    """The decoder's placement state (extents, kept prompts, lanes), the same on every stage at an idle point."""

    pool = multi.pool
    return {"free": int(pool.free_rows()),
            "extents": [[int(x.eid), int(x.base), int(x.size), None if x.owner is None else int(x.owner)]
                        for x in pool.extents],
            "kept": [[int(c.kid), len(c.ids), int(c.extent.eid)] for c in multi.kept],
            "lanes": [[int(l.sid), int(l.slot), int(l.st.pos), bool(l.decoding)] for l in multi.lanes.values()]}


def install_leader(tracer: Tracer, eng) -> None:
    """After the engine is built (the warm-up is not traced): the leader's hooks, outermost."""

    multi = eng.multi
    verify = multi.verify
    plan = eng.v2plan
    tracer.write({"t": "meta2", "v2": v2proto.V2_VERSION, "parallel": int(eng.parallel), "pool_rows": int(plan.rows),
                  "entries": int(plan.entries), "limit": int(plan.limit), "multi_window": int(eng.multi_window),
                  "fill_rows": int(multi.fill_rows), "prefill_rows": int(eng.e.prefill_rows),
                  "end": int(eng.stage.layers[1]), "drafts": not eng.serial_only, "policy": eng.policy,
                  "share": float(multi.share), "group": bool(multi.group), "shared": int(eng.shared)})
    # -- messages: one record a flush, an idle record when the stages are told to wait
    share = eng._share

    def _share(values):
        if tracer.on:
            ops = v2proto.describe_ops(values, M.OPS)
            tracer.write({"t": "msg", "mseq": int(values[-2]), "digest": int(values[-1]), "ops": ops,
                          "sha": _sha(values)})
            if any(o[0] == "IDLE" for o in ops):
                tracer.write({"t": "idle", "mseq": int(values[-2]), "pool": pool_summary(multi)})
        return share(values)

    eng._share = _share
    # -- admissions
    admitted = multi._admitted

    def _admitted(a, s, feed, constraint, spec=""):
        admitted(a, s, feed, constraint, spec)
        if tracer.on:
            glm = getattr(s, "glm", None) or {}
            tracer.write({"t": "admit", "tag": glm.get("tfpp_tag"), "sid": int(a["sid"]), "slot": int(a["slot"]),
                          "base": int(a["base"]), "size": int(a["size"]), "kid": int(a["kid"]),
                          "copy": int(a["copy"]), "count": int(a["count"]), "stop_eos": bool(a["stop_eos"]),
                          "draft": bool(a["draft"]), "code": [int(x) for x in a["code"]],
                          "prompt_len": len(a["prompt"]), "cached": int(s.cached),
                          "shared": [int(x) for x in a["shared"]], "images": len(a["positions"])})

    multi._admitted = _admitted
    # -- prompt chunks (module attributes: MIA's _fill / _fill_many look them up at call time)
    chunk = D.prefill_chunk

    def prefill_chunk(e, prompt, start, R, **kw):
        slot, pos = int(e.st.slot), int(e.st.pos)
        out = chunk(e, prompt, start, R, **kw)
        if tracer.on and e is eng.e:
            tracer.write({"t": "fill", "slot": slot, "pos": pos, "start": int(start), "R": int(R),
                          "head": part.devhash(out) if out is not None else None, "taps": tracer._taps(e.pbuf, R)})
        return out

    D.prefill_chunk = prefill_chunk
    pieces_fn = multi_prefill.prefill_pieces

    def prefill_pieces(e, pieces):
        segs = [[int(p.st.slot), int(p.st.pos), int(p.rows)] for p in pieces]
        heads = pieces_fn(e, pieces)
        if tracer.on and e is eng.e:
            R = sum(s[2] for s in segs)
            tracer.write({"t": "pieces", "segs": segs, "R": R,
                          "heads": [part.devhash(h) if h is not None else None for h in heads],
                          "taps": tracer._taps(e.pbuf, R)})
        return heads

    multi_prefill.prefill_pieces = prefill_pieces
    # -- verify windows and their keeps (instance attributes: multi.forward_streams calls verify.forward)
    forward, commit = verify.forward, verify.commit

    def v_forward(segments):
        v = forward(segments)
        if tracer.on:
            slot_sid = {l.slot: l.sid for l in multi.lanes.values()}
            segs = [[int(s.st.slot), int(s.st.pos), len(s.tokens)] for s in segments]
            rows = []
            for lg, s in zip(v.logits, segments):
                rows += part.devhash_rows(lg[:len(s.tokens)])
            R = sum(x[2] for x in segs)
            tracer.write({"t": "round", "rid": (multi.sent - 1) % M.SEAL_MOD, "segs": segs,
                          "sids": [slot_sid.get(x[0]) for x in segs],
                          "tokens": [int(t) for s in segments for t in s.tokens], "logits": rows,
                          "taps": tracer._taps(verify.b, R)})
        return v

    def v_commit(segments, keeps):
        if tracer.on:
            tracer.write({"t": "keeps", "rid": (multi.sent - 1) % M.SEAL_MOD, "keeps": [int(k) for k in keeps],
                          "end": [int(s.st.pos) + int(k) for s, k in zip(segments, keeps)]})
        return commit(segments, keeps)

    verify.forward, verify.commit = v_forward, v_commit
    # -- the iteration's end
    rnd = multi.round

    def round_():
        done = rnd()
        if tracer.on:
            tracer.write({"t": "pool", **pool_summary(multi)})
        return done

    multi.round = round_
    # -- the client side: a record a request, tagged like the scheduler's queue tags it
    counter = itertools.count(1)
    gen = eng.generate
    lock = threading.Lock()

    def generate(prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None, *, vision=None,
                 background=False):
        own_tag = getattr(eng.request, "trace_tag", None)
        with lock:
            tag = own_tag or f"req{next(counter)}"
        eng.request.trace_tag = tag
        got: list[int] = []
        ids = [int(t) for t in prompt]
        tracer.write({"t": "req", "tag": tag, "prompt_len": len(ids),
                      "prompt_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
                      "max_tokens": int(max_tokens), "draft": bool(draft),
                      "policy": getattr(eng.request, "policy", None),
                      "sampling": None if sampling is None else [sampling.seed, sampling.temperature,
                                                                 sampling.top_k, sampling.top_p, sampling.min_p],
                      "constraint": constraint is not None, "vision": vision is not None,
                      "background": bool(background)})

        def on(toks):
            got.extend(int(x) for x in toks)
            return on_tokens(toks) if on_tokens is not None else False

        try:
            stats = gen(prompt, max_tokens, sampling, on, draft, constraint, vision=vision, background=background)
        except BaseException as exc:
            tracer.write({"t": "fail", "tag": tag, "error": f"{type(exc).__name__}: {exc}"[:500]})
            raise
        finally:
            if own_tag is None:
                eng.request.trace_tag = None
        tracer.write({"t": "done", "tag": tag, "tokens": got,
                      "stats": {k: v for k, v in (stats or {}).items() if k not in TIMING_KEYS}})
        return stats

    eng.generate = generate


def install_follower(tracer: Tracer, glm, multi) -> None:
    """A follower's records: the messages it applies, the pool at its idle points, the rows it sends on."""

    share = glm._share
    last = {"mseq": None}

    def _share(values):
        out = share(values)
        if tracer.on and values is None:
            ops = v2proto.describe_ops(out, M.OPS)
            last["mseq"] = int(out[-2])
            tracer.write({"t": "msg", "mseq": int(out[-2]), "digest": int(out[-1]), "ops": ops, "sha": _sha(out)})
        return out

    glm._share = _share
    apply = multi.apply

    def apply_(msg):
        apply(msg)
        if tracer.on and any(o[0] == "IDLE" for o in v2proto.describe_ops(list(msg) + [0, 0], M.OPS)):
            tracer.write({"t": "idle", "mseq": last["mseq"], "pool": pool_summary(multi)})

    multi.apply = apply_

    def act_rows(fields, R, b):
        if tracer.on:
            tracer.write({"t": "act", "fields": fields, "R": int(R), "x": part.devhash(b.x[:R]),
                          "taps": [[s, part.devhash(b.taps[s][:R])] for s in part.STAGE.carry_out]})

    tracer.act_rows = act_rows
