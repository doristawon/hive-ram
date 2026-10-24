"""A stage's share of GLM's forward: ``forward._compute`` (G/forward.py:1019-1041) cut at its layer range.

Between layers the only state is ``b.x[:R]`` (each layer writes its hyper-connection streams back in place, the next
starts from ``hc_pre(x)``: G/forward.py:870-887), and DFlash2's taps are the stream means after chosen layers. So a
stage that is not the first takes x (and the taps computed upstream) in; one that is not the last sends them on.

``forward.compute`` looks ``_compute`` up by its global name at call time (G/forward.py:1010, 1015) and the graphs
module binds ``compute``, not ``_compute`` (G/graphs.py:8), so replacing the module attribute reaches eager windows,
prompt chunks and every CUDA graph capture alike. I/O sits here only for prompt buffers (``b.prefill``), which are
never captured; a decode window's ACT is moved by the four callers around ``Engine.forward`` / the batched verify
(a graph replay never calls ``_compute``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

STAGE: "Stage | None" = None
ORIGINAL_COMPUTE = None
fw: Any = None             # tensorfold.families.glm5_next.cuda.forward, bound by install_common
glue: Any = None


@dataclass
class Stage:
    index: int
    count: int
    layers: tuple[int, int]
    first: bool
    last: bool
    taps: tuple[int, ...] = ()          # AGREE's tap layers in slot order (() without a drafter)
    carry_in: tuple[int, ...] = ()      # slots received from upstream: taps[s] < layers[0]
    carry_out: tuple[int, ...] = ()     # slots sent downstream: taps[s] < layers[1] (none from the last stage)
    io: Any = None                      # wire.ActIO; None for N=1
    scrub: bool = False                 # only while an Engine / BatchedVerify captures its graphs

    @property
    def own_taps(self) -> tuple[int, ...]:
        a, b = self.layers
        return tuple(s for s, t in enumerate(self.taps) if a <= t < b)

    @property
    def valid_taps(self) -> tuple[int, ...]:
        """Slots whose rows hold real values on this stage after a forward (received or computed here)."""
        return tuple(sorted(set(self.carry_in) | set(self.own_taps)))


def _compute_pp(w, st, b, R, *, logits=True, nch=None, host_pos=None, sparse_np=None, inject=None, head=True,
                mixer=None):
    S, c = STAGE, w.cfg
    if b.split is not None:
        raise RuntimeError("tfpp-glm: TF_GLM_HC_SPLIT / TF_GLM_PREFILL_OVERLAP must be 0")
    if S.first:                                                   # G/forward.py:1022-1025
        glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
        if inject is not None:
            rows, features = inject
            b.x.index_copy_(0, rows, features.repeat(1, c.streams))
    elif b.prefill:                                               # prompt buffers: always eager, never captured
        S.io.recv_act(b, R, st, "chunk")                          # x[:R] and the carried taps (inside KeepWarm)
    elif S.scrub and not torch.cuda.is_current_stream_capturing():
        b.x[:R].zero_()                                           # a capture's warm-up runs only, not the captured one
    for layer in w.layers:                                        # G/forward.py:1029-1032
        fw.layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np, mixer)
        for slot in b.tap_at.get(layer.index, ()):
            glue.stream_mean(b.x[:R], b.taps[slot][:R])
    if not S.last:
        if b.prefill:
            S.io.send_act(b, R, st, "chunk")                      # x[:R] and carry_out's taps
            return None
        return b.logits[:R]               # the [R, 1] stand-in: BatchedVerify's eager path slices it (G/verify.py:241-245)
    glue.stream_mean(b.x[:R], b.hidden[:R])                       # as G/forward.py:1033-1041 from here
    if not logits:
        return None
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if not head:
        return None
    if b.prefill:
        return fw.mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return fw.mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])


def bind_modules() -> None:
    global fw, glue, ORIGINAL_COMPUTE
    from tensorfold.families.glm5_next.cuda import forward as _fw, glue as _glue

    fw, glue = _fw, _glue
    if ORIGINAL_COMPUTE is None:
        ORIGINAL_COMPUTE = _fw._compute


def install_common(stage: Stage) -> None:
    """This stage's description, and ``forward._compute`` -> ``_compute_pp`` (every stage, N=1 included unless
    --pure). Installed after env.check_env and before any Engine is built."""

    global STAGE
    bind_modules()
    STAGE = stage
    fw._compute = _compute_pp


def scrubbing(on: bool) -> None:
    if STAGE is not None:
        STAGE.scrub = bool(on)


# -- exact device hashes (tracing only) --------------------------------------------------------------------------
_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _weights(n: int, device) -> torch.Tensor:
    return torch.arange(n, device=device, dtype=torch.int64) % 65521 + 1


def devhash(t: torch.Tensor) -> list[int]:
    """[sum, weighted sum] of a tensor's raw words as integers (exact, on its device): equal hashes mean equal bits
    with overwhelming likelihood, and no host copy of the tensor."""

    v = t.contiguous().view(_INT[t.element_size()]).reshape(-1).to(torch.int64)
    if v.numel() == 0:
        return [0, 0]
    s = torch.stack([v.sum(), (v * _weights(v.numel(), v.device)).sum()])
    return [int(x) for x in s.tolist()]


def devhash_rows(t: torch.Tensor) -> list[list[int]]:
    """``devhash`` of each row of a 2-D tensor (one host sync for all rows)."""

    if t.dim() != 2:
        t = t.reshape(t.shape[0], -1)
    v = t.contiguous().view(_INT[t.element_size()]).to(torch.int64)
    if v.numel() == 0:
        return [[0, 0] for _ in range(t.shape[0])]
    wgt = _weights(v.shape[1], v.device)
    s = torch.stack([v.sum(dim=1), (v * wgt).sum(dim=1)], dim=1)
    return [[int(a), int(b)] for a, b in s.tolist()]
