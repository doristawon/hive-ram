"""tfpp-glm v2: ACT rows in and out under v2 headers (spec v2 section 9.2).

v1's ``ActIO.recv_act`` / ``send_act`` carry ``{"kind", "pos", "R"}`` and check ``pos`` against the lane's ``st.pos``;
v2's ACT frames describe what the rows are (a lane's chunk, a multi-prompt chunk, a batched verify window, each with the
group ``g`` and the fields ``v2proto.act_*`` build) and the receiver checks every field against the value it computed
itself. The pinned staging, the non-blocking H2D copies and the KeepWarm wrapper are ``wire.ActIO``'s and ``wire.Link``'s
own (the staging is sized for ``prefill_rows``, so windows of up to 64 rows and chunks of up to 2,048 fit).
"""

from __future__ import annotations

import time
from contextlib import nullcontext

from . import proto
from .wire import LinkError


def recv_rows(io, b, R: int, fields: dict) -> None:
    """The next ACT frame from the stage before, its x rows (and the carried taps) into ``b``'s views, inside
    KeepWarm; every header field must equal ``fields`` and the parts must be the ones this stage expects."""

    if io.up is None:
        raise proto.ProtocolError("recv_rows on stage 0")
    into = io._views(b, R, io.stage.carry_in)                               # T/wire.py:412-416
    t = time.perf_counter()
    with (io.warm if io.warm is not None else nullcontext()):
        f = io.up.recv(timeout=io.timeout, into=into)
    io.recv_ms += (time.perf_counter() - t) * 1e3
    if f.op == proto.ERROR:
        raise LinkError(f"stage {f.hdr.get('stage')} failed in {f.hdr.get('where')}: {f.hdr.get('msg')}")
    h = f.hdr
    if f.op != proto.ACT or any(h.get(k) != v for k, v in fields.items()):
        raise LinkError(f"{io.up.name}: expected ACT {fields}, got {f.op} {h} (the chain lost step)")
    if {p[0] for p in h["parts"]} != set(into):
        raise LinkError(f"{io.up.name}: ACT parts {sorted(p[0] for p in h['parts'])}, expected {sorted(into)}")


def send_rows(io, b, R: int, fields: dict) -> None:
    """``b``'s x rows (and ``carry_out``'s taps) to the next stage under ``fields``."""

    parts = [("x", b.x[:R])] + [(f"tap:{s}", b.taps[s][:R]) for s in io.stage.carry_out]
    io.down.send(proto.ACT, fields, parts)
    hook = getattr(io.tracer, "act_rows", None)                             # v2trace attaches it
    if hook is not None:
        hook(fields, int(R), b)
