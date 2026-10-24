# SPDX-License-Identifier: Apache-2.0
# Portions adapted from TensorFold (Apache-2.0); modified by the hive-ram contributors.
"""One rank's share of a forward: TensorFold's ``compute`` cut at a layer boundary.

A cut writes the last MoE branch back into the streams (``finish`` does the same before the head) and the next rank
starts its first layer with nothing pending, as Flash Next's own n-gram layer does mid-model.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorfold.families.qwen4_exp.cuda import forward as fw
from tensorfold.families.qwen4_exp.cuda.graphs import Graphs


@torch.no_grad()
def compute_part(w, segs, b, *, first: bool, last: bool, h_in: torch.Tensor | None = None, logits: bool = True,
                 ends: Sequence[int] = (), context: int | None = None, preloaded: bool = False):
    """``preloaded``: a later rank's input rows are already in b.h (a graph replays from there)."""

    c = w.cfg
    R = segs[-1][2]
    if first:
        fw._embed(w, b.ids[:R], c.streams, b.h[:R])
    elif not preloaded:
        b.h[:R].copy_(h_in)
    pending = None
    for layer in w.layers:
        pending = fw.layer_forward(layer, w, segs, b, R, pending, context=context)
    if last:
        return fw.finish(w, w.mixer, b, R, pending, logits=logits, ends=ends)
    b.streams[:R].copy_(b.h[:R])
    if pending is not None:
        fw._writeback(b.streams, b, R, c, pending)
    return b.streams[:R]


@torch.no_grad()
def forward_part(e, tokens: Sequence[int], *, prefill: bool, first: bool, last: bool,
                 h_in: torch.Tensor | None = None, logits: bool = True):
    """Stage ``tokens`` (ids, n-gram rows of this rank's layers) and run this rank's layers on them."""

    if not prefill and isinstance(e.graphs, PartGraphs):
        if (e.graphs.first, e.graphs.last) != (first, last):
            raise ValueError(f"this rank's decode graphs were built for first={e.graphs.first}, last={e.graphs.last}, "
                             f"not first={first}, last={last} (driver.engine's first/last)")
        return e.graphs.part(tokens, h_in)
    b = e.pbuf if prefill else e.buf
    segs = fw.stage(e.w, b, [(e.st, list(tokens))])
    return compute_part(e.w, segs, b, first=first, last=last, h_in=h_in, logits=logits)


class PartGraphs(Graphs):
    """TensorFold's decode graphs for one rank's layers; the MTP head's graphs come from ``Graphs`` unchanged."""

    def __init__(self, e, *, first: bool, last: bool, max_rows: int = 8) -> None:
        super().__init__(e, max_rows=max_rows)
        self.first, self.last = first, last

    def _run(self, segs, context=None):
        e = self.e
        return compute_part(e.w, segs, e.buf, first=self.first, last=self.last, preloaded=True, context=context)

    @torch.no_grad()
    def part(self, tokens, h_in: torch.Tensor | None):
        e = self.e
        w, st, b = e.w, e.st, e.buf
        segs = fw.stage(w, b, [(st, list(tokens))])
        R = segs[-1][2]
        if not self.first:
            b.h[:R].copy_(h_in)
        if R > self.max_rows:
            return self._run(segs)
        context = self._bucket(st.pos + R)
        key = (R, st.cur[0] if st.cur else 0, context)
        g = self.main.get(key)
        if g is None:
            self._run(segs, context)                       # eager warm-up: compiles this launch shape
            g = self._capture(lambda: self._run(segs, context))
            self.main[key] = g
            if not self.first:                             # the warm-up ran the layers on b.h: its input again
                b.h[:R].copy_(h_in)
        g.replay()
        return b.logits[:R] if self.last else b.streams[:R]

    @torch.no_grad()
    def warm(self, rows: int | None = None) -> int:
        """Capture every decode window at both DeltaNet parities (and the MTP head's steps) before any request.

        Like TensorFold's own ``Graphs.warm`` this runs at position 0, so it captures the first context bucket
        (contexts up to 8192 tokens, or the capacity if smaller); a request that grows past it captures the next
        bucket's windows on first use, on every rank, which shows up as a one-time stall in that request."""

        e = self.e
        st = e.st
        rows = rows or self.max_rows
        saved = list(st.cur)
        before = self.captures
        zeros = torch.zeros_like(e.buf.h[:rows])
        for parity in (0, 1):
            st.cur = [parity] * len(st.cur)
            for R in range(1, rows + 1):
                self.part([0] * R, zeros[:R])
        st.cur = saved
        if e.mbuf is not None:
            for n in range(1, rows + 1):
                self.mtp_forward([0] * n, e.buf.streams[:n])
        torch.cuda.synchronize()
        e.reset()
        return self.captures - before
