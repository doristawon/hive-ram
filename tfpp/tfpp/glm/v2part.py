"""tfpp-glm v2 (--parallel 2..8): a stage's share of GLM's forward with the prompt ACT moved under v2 headers.

``part._compute_pp`` (v1) reads a prompt chunk's ACT through ``ActIO.recv_act`` with the lane's ``st.pos``. Under
--parallel two things differ (spec v2 section 9.1): a multi-prompt chunk (``multi_prefill.prefill_pieces``) calls
``compute`` with ``st=None`` (G/multi_prefill.py:172), so its ACT is described by the pieces ``[[slot, pos, rows], ...]``
(``PieceMixer.pieces``, G/multi_prefill.py:110-111; the positions are read before the commit, :183-184), and every ACT
carries the group ``g`` (always 0 in v2.0). Decode windows (the batched verify, eager or captured) never reach the I/O
branch: they run ``compute(w, None, b, R, mixer=...)`` with a non-prompt buffer (G/verify.py:121-148, 241), their ACT is
moved by ``PipeBatchedVerify`` / ``FollowerBatchedVerify`` around ``forward`` (a graph replay never calls ``_compute``).

``v1`` modules are untouched: this module installs its own ``_compute`` instead of ``part.install_common``'s.
"""

from __future__ import annotations

import torch

from . import part, v2proto, v2wire

GROUP = 0        # the group of the op being computed (v2.1 sets it; always 0 in v2.0)


def _prompt_fields(st, mixer, R: int) -> dict:
    """The fields of a prompt chunk's ACT frame: a lane's own chunk (``prefill_chunk``, ``st`` given) or a multi-prompt
    chunk (``prefill_pieces``, ``st`` None, the pieces on the mixer)."""

    if st is not None:                                                      # prefill_chunk (G/decode.py:626)
        return v2proto.act_chunk(GROUP, int(st.slot), int(st.pos), R)
    pieces = getattr(mixer, "pieces", None)                                 # PieceMixer (G/multi_prefill.py:110-111)
    if pieces is None:
        raise RuntimeError("a prompt forward with no stream state outside a multi-prompt chunk "
                           "(compute_steps, calibration): not in a pipeline")
    return v2proto.act_pieces(GROUP, [[int(p.st.slot), int(p.st.pos), int(p.rows)] for p in pieces], R)


def _compute_pp_multi(w, st, b, R, *, logits=True, nch=None, host_pos=None, sparse_np=None, inject=None,
                      head=True, mixer=None):
    """``part._compute_pp`` (T/part.py:51-82) with every prompt ACT moved by v2wire under v2 headers."""

    S, c, fw, glue = part.STAGE, w.cfg, part.fw, part.glue
    if b.split is not None:
        raise RuntimeError("tfpp-glm: TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP must be 0")
    if S.first:
        glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
        if inject is not None:
            rows, features = inject
            b.x.index_copy_(0, rows, features.repeat(1, c.streams))
    elif b.prefill:
        v2wire.recv_rows(S.io, b, R, _prompt_fields(st, mixer, R))
    elif S.scrub and not torch.cuda.is_current_stream_capturing():
        b.x[:R].zero_()
    for layer in w.layers:
        fw.layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np, mixer)
        for slot in b.tap_at.get(layer.index, ()):
            glue.stream_mean(b.x[:R], b.taps[slot][:R])
    if not S.last:
        if b.prefill:
            v2wire.send_rows(S.io, b, R, _prompt_fields(st, mixer, R))     # positions before the commit
            return None
        return b.logits[:R]
    glue.stream_mean(b.x[:R], b.hidden[:R])           # T/part.py:74-82 verbatim from here
    if not logits:
        return None
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if not head:
        return None
    if b.prefill:
        return fw.mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return fw.mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])


def install(stage) -> None:
    """This stage's description, and ``forward._compute`` -> ``_compute_pp_multi`` (every stage of a --parallel chain,
    N=1 included unless --pure). Installed after env.check_env and before any Engine is built."""

    part.bind_modules()
    part.STAGE = stage
    part.fw._compute = _compute_pp_multi              # looked up at call time (G/forward.py:1010, 1015)
