"""Keep the GPU's clocks up while a rank waits on the link.

Measured on both N1X laptops (2026-10-22): a 16-layer decode step takes 8.8 ms back to back, 15-16 ms after a
20-40 ms idle gap (the GPU lowers its clocks; a busy CPU does not help), and 9.9 ms when one spinning block runs on a
side stream through the gap. In a pipeline every rank idles while the others work, so each wait is wrapped in this.
"""

from __future__ import annotations

import os
import threading

import torch

SPIN_MS = float(os.environ.get("TFPP_WARM_MS", "1.0"))      # one spin's length; 0 turns keep-warm off


class KeepWarm:
    def __init__(self, spin_ms: float = SPIN_MS) -> None:
        self.on = spin_ms > 0
        self.waiting = threading.Event()
        if not self.on:
            return
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.Stream(device=self.device)
        self.cycles = int(spin_ms * 2.0e6)                   # ~2 GHz: about spin_ms at full clocks
        threading.Thread(target=self._run, name="tfpp-keepwarm", daemon=True).start()

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            self.waiting.wait()
            with torch.cuda.stream(self.stream):
                torch.cuda._sleep(self.cycles)
            self.stream.synchronize()

    def __enter__(self) -> "KeepWarm":
        if self.on:
            self.waiting.set()
        return self

    def __exit__(self, *exc) -> None:
        self.waiting.clear()
