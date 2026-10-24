# SPDX-License-Identifier: Apache-2.0
"""Time one rank's decode step alone (no link).

  python rank_bench.py MODEL A:B first|mid|last [ROWS] [--modes sleep@0,gpuwarm@30]

``--modes``: comma-separated MODE@GAP_MS; before each step the rank idles GAP_MS by sleeping (sleep), spinning the
CPU (cpuspin) or with one spinning GPU block on a side stream (gpuwarm).
"""

from __future__ import annotations

import argparse
import time

MODES = ("sleep", "cpuspin", "gpuwarm")


def parse_modes(text: str) -> list[tuple[str, float]]:
    out = []
    for item in text.split(","):
        mode, sep, gap = item.partition("@")
        if not sep or mode not in MODES:
            raise ValueError(f"mode {item!r}: expected MODE@GAP_MS with MODE one of {', '.join(MODES)}")
        try:
            g = float(gap)
        except ValueError:
            raise ValueError(f"mode {item!r}: the gap is not a number") from None
        if g < 0:
            raise ValueError(f"mode {item!r}: a negative gap")
        out.append((mode, g))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="time one rank's decode step alone (no link)")
    ap.add_argument("model")
    ap.add_argument("layers", help="A:B")
    ap.add_argument("role", choices=["first", "mid", "last"])
    ap.add_argument("rows", nargs="?", type=int, default=1, help="rows per step, 1 to 8")
    ap.add_argument("--modes", default="sleep@0", help="MODE@GAP_MS,... (sleep, cpuspin, gpuwarm)")
    a = ap.parse_args()

    from tfpp import proto

    try:
        lo, hi = proto.parse_layers(a.layers)
        modes = parse_modes(a.modes)
    except ValueError as exc:
        ap.error(str(exc))
    if not 1 <= a.rows <= 8:                    # the engine below is built for depth 0: 8-row windows
        ap.error(f"rows {a.rows}: expected 1 to 8")

    import torch

    from tfpp import driver
    from tfpp.load import load_range
    from tfpp.part import forward_part

    torch.cuda.set_device(0)
    w = load_range(a.model, lo, hi, mtp=False, ple_on_ssd=True)
    first, last = a.role == "first", a.role == "last"
    e = driver.engine(w, 0, 4096, graphs=True, first=first, last=last)
    rows = a.rows
    h = torch.randn((rows, w.cfg.streams * w.cfg.hidden), dtype=torch.bfloat16, device="cuda") * 0.1
    toks = [100] * rows
    side = torch.cuda.Stream(priority=0)
    for mode, gap in modes:
        ts = []
        for i in range(120):
            if gap and mode == "sleep":
                time.sleep(gap / 1000)
            elif gap and mode == "cpuspin":
                end = time.perf_counter() + gap / 1000
                while time.perf_counter() < end:
                    pass
            elif gap and mode == "gpuwarm":            # one spinning block on a side stream through the gap
                with torch.cuda.stream(side):
                    torch.cuda._sleep(int(gap * 1e6))   # ~1 GHz cycles per ms is a lower bound; it may run longer
                time.sleep(gap / 1000)
            torch.cuda.synchronize()
            t = time.perf_counter()
            forward_part(e, toks, prefill=False, first=first, last=last, h_in=None if first else h)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ts = sorted(ts[20:])
        torch.cuda.synchronize()
        print(f"layers {lo}:{hi} {a.role} rows {rows} {mode} gap {gap:g} ms: median {ts[len(ts) // 2] * 1000:.2f} ms, "
              f"p10 {ts[len(ts) // 10] * 1000:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
