# SPDX-License-Identifier: Apache-2.0
# Portions adapted from TensorFold (Apache-2.0); modified by the hive-ram contributors.
"""Rank roles: the leader (rank 0) drives requests; middle ranks pass rows on; the last rank heads, samples and drafts.

A round: the leader runs its layers on [pending] + drafts and sends the streams down the chain; the last rank samples,
keeps the drafts that match, commits, drafts the next ones with the MTP head and replies; every rank commits the
same number of rows. Prompt chunks flow the same way and commit whole.

Startup: the leader's ``hello`` goes down the chain and every rank answers with its record (layers, checkpoint, cache,
windows, MTP head); the leader checks them as one pipeline (``proto.check_chain``) before the first request. Every
frame a rank receives is checked before it is used (``proto.check_frame`` / ``proto.check_reply``). A rank that fails
sends an ``error`` frame to both neighbours (best effort) and stops; the leader raises it as ``RemoteError`` naming
the rank. A Leader whose request failed part-way refuses further requests: the ranks' caches are in an unknown state.

Between requests a follower waits for the next frame without KeepWarm and without a timeout; while a request is live
(from its reset to the reply that ends it) it waits under KeepWarm and within the link's io_timeout.
"""

from __future__ import annotations

import inspect
import os
import time

import torch

from tensorfold.families.qwen4_exp.cuda import decode as dec
from tensorfold.families.qwen4_exp.cuda.forward import commit
from tensorfold.families.qwen4_exp.cuda.mtp import mtp_forward

from . import proto
from .part import PartGraphs, forward_part
from .proto import LinkError, ProtocolError, RemoteError
from .warm import KeepWarm

TIMING = os.environ.get("TFPP_TIMING") == "1"


class Clock:
    """Per-phase wall time (with a device sync at each mark when TFPP_TIMING=1)."""

    def __init__(self) -> None:
        self.acc: dict[str, float] = {}
        self.t = time.perf_counter()

    def mark(self, name: str) -> None:
        if not TIMING:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + now - self.t
        self.t = now

    def reset(self) -> None:
        self.acc, self.t = {}, time.perf_counter()

    def report(self, rounds: int) -> dict:
        return {k: round(v * 1000 / max(1, rounds), 3) for k, v in self.acc.items()}


def _graphs_blocked(w) -> str | None:
    """Why this rank must not capture decode graphs, or None. As TensorFold's Engine: experts that read their plan
    on the host (NVFP4) cannot be captured. And KeepWarm's thread calls CUDA while a graph may be captured lazily,
    which only a thread-local capture (TensorFold 0.6.1's) tolerates."""

    if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
        return "this MoE reads its plan's item list on the host, which a capture rejects"
    from tensorfold.families.qwen4_exp.cuda.graphs import Graphs

    try:
        src = inspect.getsource(Graphs._capture)
    except (OSError, TypeError):
        return None
    if "thread_local" not in src:
        return "this TensorFold captures in the global mode, which the keep-warm thread's CUDA calls would break"
    return None


def engine(w, depth: int, capacity: int, *, graphs: bool = False, first: bool, last: bool):
    """This rank's engine; ``first``/``last``: whether it embeds the tokens / applies the head (the decode graphs
    are built for that)."""

    e = dec.Engine(w, capacity=capacity, max_rows=max(8, depth + 1), graphs=False, kv_dtype="bf16")
    if graphs:
        why = _graphs_blocked(w)
        if why:
            print(f"[tfpp] CUDA graphs off: {why}; decode runs eagerly (correct, slower)", flush=True)
        else:
            e.graphs = PartGraphs(e, first=first, last=last, max_rows=e.rows)
            n = e.graphs.warm(depth + 1)
            print(f"[tfpp] {n} decode graphs captured", flush=True)
    return e


def _tensorfold_version() -> str | None:
    try:
        import tensorfold

        return str(getattr(tensorfold, "__version__", None))
    except Exception:  # noqa: BLE001
        return None


def rank_record(e, role: str, depth: int) -> dict:
    """What HELLO tells the leader about this rank (``proto.check_chain`` compares the records)."""

    w = e.w
    first, end = w.meta.get("pp_range", (0, w.cfg.layers))
    return {"proto": proto.PROTO, "role": role, "first": int(first), "end": int(end), "total": int(w.cfg.layers),
            "capacity": int(e.capacity), "prefill_rows": int(e.prefill_rows), "rows": int(e.rows),
            "depth": int(depth), "mtp": w.mtp is not None, "around_one": bool(getattr(w, "around_one", True)),
            "config": w.meta.get("pp_config"), "tensorfold": _tensorfold_version(),
            "graphs": isinstance(e.graphs, PartGraphs)}


def _where(e) -> str:
    first, end = e.w.meta.get("pp_range", (0, e.w.cfg.layers))
    return f"layers {first}:{end}"


def _limits(e) -> proto.Limits:
    c = e.w.cfg
    return proto.Limits(vocab=int(c.vocab), rows=int(e.rows), prefill_rows=int(e.prefill_rows),
                        width=int(c.streams * c.hidden))


def _down_reply(down, up, warm, kind: str, *, vocab: int, rows: int | None = None, depth: int | None = None,
                timeout=None) -> dict:
    """The next rank's reply to ``kind``, checked. An error frame goes on upstream (``up``) and raises."""

    kw = {} if timeout is None else {"timeout": timeout}
    if warm is None:
        r, t = down.recv(**kw)
    else:
        with warm:
            r, t = down.recv(**kw)
    if t is not None:
        raise ProtocolError(f"a {kind} reply with a tensor")
    if r.get("op") == "error" and up is not None:
        up.fail(r.get("where") or "a later rank", r.get("msg") or "")
    return proto.check_reply(r, kind, vocab=vocab, rows=rows, depth=depth)


class Leader:
    def __init__(self, e, down, *, depth: int = 0, timeout: float | None = None) -> None:
        """``depth``: the drafts this leader will ask for (HELLO checks every rank can take them); ``timeout``: how
        long to wait for the chain's HELLO reply (None: the link's io_timeout)."""

        self.e, self.down = e, down
        self.warm = KeepWarm()
        self.broken = False
        down.max_payload = 0                 # replies never carry a tensor
        self.ranks = self._hello(depth, timeout)
        self.min_rows = min(r["rows"] for r in self.ranks)

    def _hello(self, depth: int, timeout) -> list[dict]:
        own = rank_record(self.e, "leader", depth)
        self.down.send({"op": "hello", "proto": proto.PROTO})
        r = _down_reply(self.down, None, None, "hello", vocab=self.e.w.cfg.vocab, timeout=timeout)
        ranks = [own] + list(r["ranks"])
        problems = proto.check_chain(ranks, depth)
        if problems:
            msg = "the ranks do not form one pipeline:\n  " + "\n  ".join(problems)
            self.down.fail("leader", msg, drain_s=0)
            raise ValueError(msg)
        print("[tfpp] chain: " + " -> ".join(f"{r['role']} {r['first']}:{r['end']}" for r in ranks)
              + f" of {own['total']} layers", flush=True)
        return ranks

    def _send(self, header: dict, h=None) -> None:
        try:
            self.down.send(header, h)
        except LinkError as exc:             # the next rank may have said why before it went
            if self.down.readable(0.5):
                try:
                    r, _ = self.down.recv(timeout=1.0)
                except (LinkError, OSError):
                    r = {}
                if r.get("op") == "error":
                    raise RemoteError(f"{r.get('where') or 'a later rank'}: {r.get('msg')}") from exc
            raise LinkError(f"{exc}; a later rank stopped, see its log") from exc

    def _early_reply(self) -> None:
        """Between prompt chunks nothing comes back unless a later rank failed."""

        if self.down.readable(0.0):
            r, _ = self.down.recv()
            proto.check_reply(r, "reset", vocab=self.e.w.cfg.vocab)      # an error frame raises RemoteError
            raise ProtocolError(f"an unexpected reply in the middle of a prompt: {r!r}")

    @torch.no_grad()
    def generate(self, prompt: list[int], count: int, depth: int, confidence: float) -> dict:
        e = self.e
        if self.broken:
            raise RuntimeError("an earlier request failed part-way and the ranks' caches are in an unknown state; "
                               "restart the pipeline")
        proto.check_request(prompt, count, depth, confidence, vocab=e.w.cfg.vocab, capacity=e.capacity,
                            rows=self.min_rows)
        try:
            return self._generate(prompt, count, depth, confidence)
        except BaseException:
            self.broken = True
            raise

    def _generate(self, prompt: list[int], count: int, depth: int, confidence: float) -> dict:
        e, w, st = self.e, self.e.w, self.e.st
        eos = set(w.cfg.eos)
        vocab = w.cfg.vocab
        self._send({"op": "reset"})
        e.reset()
        _down_reply(self.down, None, self.warm, "reset", vocab=vocab)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start, r = 0, None
        while start < len(prompt):
            end = min(start + e.prefill_rows, len(prompt))
            chunk, final = prompt[start:end], end == len(prompt)
            h = forward_part(e, chunk, prefill=True, first=True, last=False)
            self._send({"op": "prefill", "tokens": chunk, "next": prompt[start + 1:end + 1], "final": final,
                        "depth": depth, "conf": confidence, "remaining": count - 1}, h)
            commit(w, st, e.pbuf, len(chunk), len(chunk))
            start = end                     # the next chunk computes while the peers take this one
            if not final:
                self._early_reply()
        r = _down_reply(self.down, None, self.warm, "prefill", vocab=vocab, depth=depth)
        t1 = time.perf_counter()
        out, drafts = [r["first"]], r["drafts"]
        rounds = drafted = accepted = 0
        clock = Clock()
        while len(out) < count and out[-1] not in eos:
            tokens = [out[-1]] + drafts
            R = len(tokens)
            clock.mark("idle")
            h = forward_part(e, tokens, prefill=False, first=True, last=False)
            clock.mark("compute")
            self._send({"op": "step", "tokens": tokens, "remaining": count - len(out), "depth": depth,
                        "conf": confidence}, h)
            clock.mark("send")
            r = _down_reply(self.down, None, self.warm, "step", vocab=vocab, rows=R, depth=depth)
            clock.mark("wait")
            keep = r["keep"]
            commit(w, st, e.buf, R, keep)
            clock.mark("commit")
            out.extend(r["sampled"])
            drafts = r["drafts"]
            rounds += 1
            drafted += R - 1
            accepted += keep - 1
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        out = out[:count]
        return {"tokens": out, "prefill_s": t1 - t0, "decode_s": t2 - t1, "rounds": rounds, "drafted": drafted,
                "accepted": accepted, "prompt_tokens": len(prompt),
                "decode_tps": (len(out) - 1) / (t2 - t1) if t2 > t1 else 0.0,
                "prefill_tps": len(prompt) / (t1 - t0) if t1 > t0 else 0.0,
                "ms_per_round": clock.report(rounds), "peer_ms_per_round": r.get("timing")}

    def quit(self) -> None:
        """Tell the chain to stop (best effort: a chain that already stopped is not an error here)."""

        try:
            self.down.send({"op": "quit"})
        except (LinkError, OSError):
            pass


@torch.no_grad()
def follow(e, up, down=None, *, depth: int = 0) -> None:
    """A middle rank (``down`` set) or the last rank (``down`` None) until the leader quits. ``depth``: this rank's
    --depth, reported in HELLO."""

    where = _where(e)
    try:
        _follow(e, up, down, depth, where)
    except RemoteError:
        raise                                # the rank that failed has told both ways already
    except BaseException as exc:
        msg = f"{type(exc).__name__}: {exc}"
        up.fail(where, msg)
        if down is not None:
            down.fail(where, msg, drain_s=0)
        raise


def _follow(e, up, down, depth: int, where: str) -> None:
    w, st = e.w, e.st
    last = down is None
    eos = set(w.cfg.eos)
    lim = _limits(e)
    vocab = lim.vocab
    up.max_payload = max(lim.prefill_rows, lim.rows) * lim.width * 2     # the largest frame: a chunk or window
    if down is not None:
        down.max_payload = 0                 # replies never carry a tensor
    clock, steps = Clock(), 0
    warm = KeepWarm()
    hello = busy = False
    while True:
        clock.mark("idle")
        if busy:
            with warm:
                hdr, h = up.recv()
        else:
            hdr, h = up.recv(timeout=None)   # idle: as long as the leader takes to send the next request
        clock.mark("recv")
        op = proto.check_frame(hdr, None if h is None else tuple(h.shape), lim, hello_done=hello)
        if op == "error":
            if down is not None:
                down.fail(hdr.get("where") or "the leader", hdr.get("msg") or "", drain_s=0)
            raise RemoteError(f"{hdr.get('where') or 'the leader'}: {hdr.get('msg')}")
        if op == "hello":
            own = rank_record(e, "last" if last else "middle", depth)
            ranks = [own]
            if down is not None:
                down.send(hdr)
                ranks += _down_reply(down, up, None, "hello", vocab=vocab)["ranks"]
            up.send({"op": "hello", "ranks": ranks})
            hello = True
            continue
        if op == "quit":
            if down is not None:
                down.send(hdr)
            return
        if op == "reset":
            clock.reset()
            steps = 0
            e.reset()
            if down is not None:
                down.send(hdr)
                _down_reply(down, up, warm, "reset", vocab=vocab)
            up.send({"ok": 1})
            busy = True
            continue
        tokens = hdr["tokens"]
        R = len(tokens)
        if op == "prefill":
            final = hdr["final"]
            got = forward_part(e, tokens, prefill=True, first=False, last=last, h_in=h, logits=final)
            if not last:
                down.send(hdr, got)
                commit(w, st, e.pbuf, R, R)
                if final:
                    reply = _down_reply(down, up, warm, "prefill", vocab=vocab, depth=hdr["depth"])
                    up.send(reply)
                    busy = not reply.get("done", False)
                continue
            logits = got.clone() if final else None
            e.last_streams = e.pbuf.streams[R - 1:R].clone()
            nxt = hdr["next"]
            if w.mtp is not None and nxt:
                mtp_forward(w, st, e.pbuf, nxt, e.pbuf.streams[:len(nxt)], logits=False)
                st.set_mtp_len(st.mtp_len + len(nxt))
            commit(w, st, e.pbuf, R, R)
            if not final:
                continue
            first = e.sample(logits, [st.pos], None)[0]
            n = min(hdr["depth"], hdr["remaining"])
            drafts = (dec.draft(e, e.last_streams, [first], st.pos + 1, n, None, hdr["conf"])
                      if n > 0 and w.mtp is not None and first not in eos else [])
            done = proto.done_after_prefill(hdr["remaining"], first, eos)
            up.send({"first": first, "drafts": drafts, "done": done})
            busy = not done
            continue
        # op == "step"
        steps += 1
        got = forward_part(e, tokens, prefill=False, first=False, last=last, h_in=h)
        clock.mark("compute")
        if not last:
            down.send(hdr, got)
            reply = _down_reply(down, up, warm, "step", vocab=vocab, rows=R, depth=hdr["depth"])
            clock.mark("downstream")
            commit(w, st, e.buf, R, reply["keep"])
            clock.mark("commit")
            up.send(reply)
            busy = not reply.get("done", False)
            continue
        sampled = e.sample(got[:R], [st.pos + 1 + r for r in range(R)], None)
        clock.mark("sample")
        keep = 1
        for i, d in enumerate(tokens[1:]):
            if sampled[i] != d or sampled[i] in eos:
                break
            keep += 1
        commit(w, st, e.buf, R, keep)
        clock.mark("commit")
        kept = sampled[:keep]
        n = min(hdr["depth"], hdr["remaining"] - keep)
        drafts = (dec.draft(e, e.buf.streams[:keep], kept, st.pos + 1, n, None, hdr["conf"])
                  if n > 0 and w.mtp is not None and kept[-1] not in eos else [])
        clock.mark("draft")
        done = proto.done_after_step(hdr["remaining"], kept, eos)
        reply = {"keep": keep, "sampled": kept, "drafts": drafts, "done": done}
        if TIMING:
            reply["timing"] = clock.report(steps)
        up.send(reply)
        busy = not done


@torch.no_grad()
def local_generate(e, prompt: list[int], count: int, depth: int, confidence: float) -> dict:
    """TensorFold's own single-GPU path, the reference the pipeline must equal."""

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first = dec.prefill(e, prompt, None, mtp=depth > 0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if depth > 0:
        res = dec.mtp_decode(e, first, count, None, depth=depth, confidence=confidence, stop_eos=True)
    else:
        res = dec.serial_decode(e, first, count, None, stop_eos=True)
    t2 = time.perf_counter()
    return {"tokens": res.tokens, "prefill_s": t1 - t0, "decode_s": t2 - t1, "rounds": res.rounds,
            "drafted": res.drafted, "accepted": res.accepted, "prompt_tokens": len(prompt),
            "decode_tps": (len(res.tokens) - 1) / (t2 - t1), "prefill_tps": len(prompt) / (t1 - t0)}
