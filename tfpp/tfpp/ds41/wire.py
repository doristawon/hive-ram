"""tfpp-ds41's links and the cut-state transport (spec 6.4, 7.5, 9).

``DsLink`` is tfpp-glm's Link (G/wire.py:95-378: seq numbers, pinned staging, scatter-gather sends, timeouts) whose
LINK frame says ``tfpp-ds41/1``. ``StageIO`` moves a stage's ACT frames:

- a prompt chunk (``recv_chunk`` / ``send_chunk``): h, pre, the carried top-k / candidate pool and the replica rows of
  the kv-source planes the stages after the cut attend; replica rows land straight in the plane's rows (spec 6.4);
- an encoder-only chunk (``recv_kv`` / ``send_kv``): the replica rows alone;
- a decode round (``recv_round`` / ``send_round``): into and out of the round decoder's static views (StageBuffers),
  which its captured graphs read and write.

Every frame's parts are what ``plan.chunk_parts`` / ``kv_parts`` / ``round_parts`` say for the cut: the receiver
computes them itself and refuses anything else (the chain lost step: fail-stop). ``local`` (the warm-up, spec 11.3):
synthetic inputs in, nothing out. Receives wait inside the KeepWarm (the GPU keeps its clocks).
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from tfpp.glm.wire import Link, LinkError, wait_any  # noqa: F401  (re-exported for the stage modules)

from . import PROTO, plan as PL, proto

TORCH_DT = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64, "u8": torch.uint8}
DT_NAME = {v: k for k, v in TORCH_DT.items()}


class DsLink(Link):
    def handshake(self, frm: str, to: str, *, timeout: float = 600.0) -> dict:
        """Both ends send LINK and read the other's: tfpp-ds41's protocol (a tfpp-glm stage says 1: refused)."""

        self._send_frame(proto.make_header(proto.LINK, -1, {"proto": PROTO, "from": frm, "to": to}, []), [])
        f = self.recv(timeout=timeout, link_frame=True)
        if f.op != proto.LINK or f.hdr.get("proto") != PROTO:
            raise proto.ProtocolError(f"{self.name}: the peer speaks {f.hdr.get('proto')!r}, not {PROTO}")
        return f.hdr


def fill_value(name: str) -> int:
    """What an unread input view holds before a capture (spec 7.4): the fork's own "no entry" values."""

    if name in ("topk", "cblk"):
        return -1                    # F/rounds.py:60, 144, 214
    if name == "cand":
        return 1                     # every block a candidate
    return 0


def rep_views(sc, s: int, with_index: bool, g0: int, g1: int) -> dict:
    """A kv-source plane's rows [g0, g1) of a stream's view (packed FP4: codes and scales, F/model.py:754-762)."""

    out = {f"c{s}": sc.comp[s][0][g0:g1], f"cs{s}": sc.comp[s][1][g0:g1]}
    if with_index:
        out[f"k{s}"] = sc.index_k[s][0][g0:g1]
        out[f"ks{s}"] = sc.index_k[s][1][g0:g1]
    return out


class StageIO:
    def __init__(self, plan, mv, up: Link | None, down: Link | None, *, warm=None, timeout: float = 600.0,
                 tracer=None, f1_chunk: bool = False, f1_round: bool = False) -> None:
        self.P, self.mv = plan, mv
        self.up, self.down = up, down
        self.warm = warm
        self.timeout = float(timeout)
        self.tracer = tracer
        self.f1_chunk, self.f1_round = bool(f1_chunk), bool(f1_round)
        self.local = False                  # warm-up: synthetic inputs, nothing sent
        self.ops = None                     # the leader's PipeOps: counts the ACT frames owed
        self.recv_ms = 0.0
        self.acts_in = self.acts_out = 0

    # -- receiving ----------------------------------------------------------------------------------------------
    def _recv(self, into: dict, kind: str, fields: dict, parts) -> None:
        if self.up is None:
            raise proto.ProtocolError(f"an ACT {kind} expected on a stage without a stage before it")
        t = time.perf_counter()
        with (self.warm if self.warm is not None else nullcontext()):
            f = self.up.recv(timeout=self.timeout, into=into)
        self.recv_ms += (time.perf_counter() - t) * 1e3
        if f.op == proto.ERROR:
            raise LinkError(f"stage {f.hdr.get('stage')} failed in {f.hdr.get('where')}: {f.hdr.get('msg')}")
        h = f.hdr
        if f.op != proto.ACT or h.get("kind") != kind or any(h.get(k) != v for k, v in fields.items()):
            got = {k: h.get(k) for k in ("kind", *fields)}
            raise proto.ProtocolError(f"{self.up.name}: expected ACT {kind} {fields}, got {f.op} {got} (the chain "
                                      "lost step)")
        want = [[n, d, [int(x) for x in s]] for n, d, s in parts]
        if h["parts"] != want or f.host:
            raise proto.ProtocolError(f"{self.up.name}: ACT {kind} parts {[p[:2] + [p[2]] for p in h['parts']]}, "
                                      f"expected {want}")
        self.acts_in += 1
        if self.ops is not None:
            self.ops.got_act()

    def recv_chunk(self, sc, start0: int, n0: int, s1: int, n1: int, floor: int):
        """(h, pre, shared, F1 carry or None) of a prompt chunk; its replica rows written into the planes."""

        P, mv = self.P, self.mv
        cin = P.cin
        parts = PL.chunk_parts(mv, cin, start0, n0, s1, n1, f1=self.f1_chunk)
        reps: dict = {}
        for s, with_index in cin.reps:                      # replica rows: straight into the planes (spec 6.4)
            g0, g1 = PL.groups(start0, n0, mv.ratios[s])
            reps.update(rep_views(sc, s, with_index, g0, g1))
        into = {}
        for name, dt, shape in parts:
            if name not in reps:
                into[name] = torch.empty(shape, dtype=TORCH_DT[dt], device="cuda")
                if self.local:
                    into[name].fill_(fill_value(name))
        if not self.local:
            into.update(reps)
            self._recv(into, "chunk", {"start0": int(start0), "n0": int(n0), "start": int(s1), "n": int(n1),
                                       "floor": int(floor)}, parts)
        shared: dict = {}
        if cin.kv_layer is not None:
            shared["kv_layer"] = cin.kv_layer
        if cin.topk is not None:
            shared["topk"] = into["topk"]
        if cin.cand:
            shared["cand"] = into["cand"].view(torch.bool)
        carry = (into["pend"], into["post"], into["comb"]) if self.f1_chunk else None
        return into["h"], into["pre"], shared, carry

    def recv_kv(self, sc, start0: int, n0: int) -> None:
        P, mv = self.P, self.mv
        parts = PL.kv_parts(mv, P.cin, start0, n0)
        if self.local:
            return
        into = {}
        for s, with_index in P.cin.reps:
            g0, g1 = PL.groups(start0, n0, mv.ratios[s])
            into.update(rep_views(sc, s, with_index, g0, g1))
        self._recv(into, "kv", {"start0": int(start0), "n0": int(n0)}, parts)

    def recv_round(self, dec) -> None:
        """Into the decoder's static input views (its graphs read them): H2D copies on the current stream."""

        if self.local:
            return
        mv, cin = self.mv, self.P.cin
        parts = PL.round_parts(mv, cin, dec.R, dec.bucket, dec.form, f1=self.f1_round)
        self._recv(dec.vin, "round", {"round": int(dec.rr.cur_round), "R": int(dec.R), "b": int(dec.bucket),
                                      "ck": dec.form}, parts)

    # -- sending ------------------------------------------------------------------------------------------------
    def _send(self, kind: str, fields: dict, parts, tensors: dict) -> None:
        out = []
        for name, dt, shape in parts:
            t = tensors[name]
            if DT_NAME.get(t.dtype) != dt or list(t.shape) != list(shape):
                raise RuntimeError(f"tfpp-ds41: ACT {kind} part {name} is {t.dtype}{list(t.shape)}, the cut rules "
                                   f"say {dt}{list(shape)} (a stage computed another state than the plan expects)")
            out.append((name, t))
        if self.local:
            return
        self.down.send(proto.ACT, dict(fields, kind=kind), out)
        self.acts_out += 1
        if self.tracer is not None:
            self.tracer.act(kind, fields, out)

    def send_chunk(self, sc, start0: int, n0: int, s2: int, n2: int, floor: int, h, pre, shared: dict,
                   carry=None) -> None:
        P, mv = self.P, self.mv
        cout = P.cout
        parts = PL.chunk_parts(mv, cout, start0, n0, s2, n2, f1=self.f1_chunk)
        tensors = {"h": h, "pre": pre}
        if cout.topk is not None:
            tensors["topk"] = shared["topk"]
        if cout.cand:
            c = shared.get("cand")
            if c is not None:
                tensors["cand"] = c.view(torch.uint8)
            elif self.local:                # warm-up: nothing is sent, a placeholder of the right shape will do
                shape = next(s for n, _, s in parts if n == "cand")
                tensors["cand"] = torch.zeros(shape, dtype=torch.uint8, device="cuda")
            else:
                # an all-zero mask would tell the stages after the cut that no block is a candidate: they would
                # compute with an empty pool where the one-process run fails (F/model.py:1048-1090 sets the pool only
                # when the source scored keys) -- a broken invariant, so stop here instead of sending wrong state
                raise RuntimeError("tfpp-ds41: the cut carries the candidate pool but this stage computed none "
                                   f"(chunk at {int(start0)}, {int(n0)} rows)")
        for s, with_index in cout.reps:
            g0, g1 = PL.groups(start0, n0, mv.ratios[s])
            tensors.update(rep_views(sc, s, with_index, g0, g1))
        if self.f1_chunk:
            if carry is None:
                raise RuntimeError("tfpp-ds41 F1: the stage's last post was applied (no pending sublayer to carry)")
            tensors.update(pend=carry[0], post=carry[1], comb=carry[2])
        self._send("chunk", {"start0": int(start0), "n0": int(n0), "start": int(s2), "n": int(n2),
                             "floor": int(floor)}, parts, tensors)

    def send_kv(self, sc, start0: int, n0: int) -> None:
        P, mv = self.P, self.mv
        parts = PL.kv_parts(mv, P.cout, start0, n0)
        tensors = {}
        for s, with_index in P.cout.reps:
            g0, g1 = PL.groups(start0, n0, mv.ratios[s])
            tensors.update(rep_views(sc, s, with_index, g0, g1))
        self._send("kv", {"start0": int(start0), "n0": int(n0)}, parts, tensors)

    def send_round(self, dec) -> None:
        """Out of the decoder's static output views, once every stretch is enqueued (the staging copy waits for the
        stream: G/wire.py:211)."""

        mv, cout = self.mv, self.P.cout
        parts = PL.round_parts(mv, cout, dec.R, dec.bucket, dec.form, f1=self.f1_round)
        self._send("round", {"round": int(dec.rr.cur_round), "R": int(dec.R), "b": int(dec.bucket), "ck": dec.form},
                   parts, dec.vout)
