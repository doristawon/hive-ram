"""KeepWarm (W/:18-43) with a gate, so no CUDA graph capture runs beside its spinning (spec 11.1).

KeepWarm's thread launches a sleep kernel and synchronizes its side stream while a stage waits on the link. torch
captures in the global capture mode: another thread's CUDA call (the synchronize) during a capture invalidates it.
``GatedKeepWarm._run`` makes each spin (sleep + synchronize) under ``lock``; ``paused()`` takes the same lock, so a
capture that holds it starts after the spin in flight ends and no spin starts until it is done. Every capture of
tfpp-ds41 (a round decoder's, the leader's batched drafter's) runs inside ``paused()``; captures never happen inside
a ``with warm:`` block (the waits are around link reads only).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import torch

from tfpp.warm import SPIN_MS, KeepWarm


class GatedKeepWarm(KeepWarm):
    def __init__(self, spin_ms: float = SPIN_MS) -> None:
        self.lock = threading.Lock()            # before super().__init__: it starts the thread that takes it
        super().__init__(spin_ms)

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            self.waiting.wait()
            with self.lock:
                if not self.waiting.is_set():   # the wait ended while a capture held the gate
                    continue
                with torch.cuda.stream(self.stream):
                    torch.cuda._sleep(self.cycles)
                self.stream.synchronize()

    @contextmanager
    def paused(self):
        with self.lock:
            yield
