"""Image prompts' feature rows (spec 8.5). The leader encodes the pictures (G/engine.py:1490-1507); stage 0 puts the
rows into the embeddings (``_compute_pp``'s inject). MIA's ``VisionFeed`` hands a chunk's rows to the other ranks
through the all-gather (G/engine.py:1598-1623); in a pipeline the PREFILL frame carries them to stage 0 instead.

``PipeVisionFeed`` is what ``generate`` builds on the leader (``engine.VisionFeed`` swapped): the same constructor,
the same ``(index, rows)`` for a chunk when the leader is stage 0 itself (one process), None otherwise.
``LocalFeed`` is stage 0's, from the rows a PREFILL brought. ``NullFeed`` (v2) marks a lane's positions on stages
that inject nothing, so ``lane.feed is not None`` reads alike everywhere (G/multi.py:1902-1903).
"""

from __future__ import annotations

import bisect

from . import part


def _chunk(positions, rows, start: int, count: int):
    import torch

    lo = bisect.bisect_left(positions, start)
    hi = bisect.bisect_left(positions, start + count)
    if lo == hi:
        return None
    index = torch.tensor([p - start for p in positions[lo:hi]], dtype=torch.int64, device="cuda")
    return index, rows[lo:hi].contiguous().to("cuda")     # stage 0's rows stay on the host: a chunk's go up


class PipeVisionFeed:
    def __init__(self, engine, positions: list[int], rows) -> None:
        self.engine, self.positions, self.rows = engine, [int(p) for p in positions], rows
        self.width = engine.w.cfg.hidden
        self.encode_s = 0.0

    def chunk(self, start: int, count: int):
        S = part.STAGE
        if S is not None and not S.first:
            return None                        # stage 0 injects them (the PREFILL frame took them there)
        return _chunk(self.positions, self.rows, start, count)

    def between(self, begin: int, end: int):
        """(absolute positions, rows) of the image rows in prompt positions [begin, end) (the PREFILL frame's)."""

        lo = bisect.bisect_left(self.positions, begin)
        hi = bisect.bisect_left(self.positions, end)
        return self.positions[lo:hi], self.rows[lo:hi]


class LocalFeed:
    """Stage 0's feed. ``rows`` stay in host memory (a request may bring ~49k rows, 384 MiB, that stage 0's
    geometry does not count: MIA counts them on rank 0, G/engine.py:66-70); each chunk uploads only its own."""

    def __init__(self, positions: list[int], rows) -> None:
        self.positions, self.rows = [int(p) for p in positions], rows
        if len(self.positions) != int(rows.shape[0]):
            raise ValueError(f"{rows.shape[0]} image rows for {len(self.positions)} positions")

    def chunk(self, start: int, count: int):
        return _chunk(self.positions, self.rows, start, count)


class NullFeed:
    def __init__(self, positions: list[int]) -> None:
        self.positions = [int(p) for p in positions]

    def chunk(self, start: int, count: int):
        return None
