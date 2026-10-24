"""Build the fork's four extensions before any stage starts, without weights (spec 13.3), and check what they link.

  python -m tfpp.ds41.prebuild         (inside toolchain.sh's environment: TORCH_EXTENSIONS_DIR keyed by the toolchain)

1. stale build locks (``<ext>/lock`` that no live process holds open; torch's FileBaton keeps its fd open while it
   builds) are removed -- a killed build leaves one and every later start waits on it forever;
2. the extensions build in order: tensorfold.cuda.exl3.linear, .experts, .prefetch, and the family's _engram_io
   (F/model.py:450-461);
3. readelf: the three CUDA extensions must NEED libcudart.so.13; anything NEEDing libcudart.so.12 (Ubuntu's CUDA 12
   runtime found by -lcudart without toolchain.sh's link: lesson 2) is refused;
4. Triton: gdc_wait / gdc_launch_dependents import (F/kernels.py:17) and a launch_pdl=True kernel runs.
Exit 0 when everything holds.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

CUDA_EXTS = ("tensorfold_exl3_linear", "tensorfold_exl3_experts", "tensorfold_exl3_l2_prefetch")


def log(msg: str) -> None:
    print(f"[tfpp-ds41 prebuild] {msg}", flush=True)


def open_paths() -> set[str]:
    """Every path a live process holds open (Linux /proc)."""

    out: set[str] = set()
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            for fd in fd_dir.iterdir():
                try:
                    out.add(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            pass
    return out


def clear_stale_locks(root: Path) -> list[str]:
    held = open_paths()
    removed = []
    for lock in root.glob("*/lock"):
        if str(lock.resolve()) not in held:
            try:
                lock.unlink()
                removed.append(str(lock))
            except OSError:
                pass
    return removed


def needed(so: Path) -> list[str]:
    try:
        out = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"readelf failed on {so}: {exc}") from None
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


def check_links(root: Path) -> list[str]:
    problems = []
    for so in sorted(root.rglob("*.so")):
        libs = needed(so)
        if any(re.fullmatch(r"libcudart\.so\.1[0-2](\..*)?", lib) for lib in libs):
            problems.append(f"{so} NEEDs a CUDA 12 runtime ({[l for l in libs if 'cudart' in l]}): toolchain.sh's "
                            f"libcudart.so link and LIBRARY_PATH; move {root} aside and rebuild")
        if so.parent.name.startswith(CUDA_EXTS) and "libcudart.so.13" not in libs:
            problems.append(f"{so} does not NEED libcudart.so.13 ({libs})")
    return problems


def triton_pdl() -> None:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    @triton.jit
    def _k(X, PDL: tl.constexpr):
        if PDL:
            gdc_wait()
            gdc_launch_dependents()
        i = tl.program_id(0)
        tl.store(X + i, i.to(tl.float32) + 1.0)

    x = torch.zeros(8, device="cuda")
    _k[(8,)](x, PDL=True, launch_pdl=True)
    torch.cuda.synchronize()
    if x.tolist() != [float(i + 1) for i in range(8)]:
        raise RuntimeError(f"a launch_pdl kernel wrote {x.tolist()}")


def main(argv: list[str] | None = None) -> int:
    from . import env, pins

    try:
        env.check_env("prebuild")
        pins.check_tree()
        env.check_device()
    except (env.EnvError, pins.TreeMismatch) as exc:
        log(str(exc))
        return 1
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or "")
    if not str(root) or not root.is_dir():
        log("TORCH_EXTENSIONS_DIR is not set to a folder (source toolchain.sh first)")
        return 1
    for lock in clear_stale_locks(root):
        log(f"removed a stale build lock: {lock}")
    import time

    import torch

    torch.cuda.set_device(0)
    from tensorfold.cuda.exl3 import experts, linear, prefetch
    from tensorfold.families.deepseek_v41.cuda import model

    for name, build in (("exl3.linear", linear._ext), ("exl3.experts", experts._ext),
                        ("exl3.prefetch", prefetch._ext), ("model._engram_io", model._engram_io)):
        t = time.perf_counter()
        build()
        log(f"{name}: ready ({time.perf_counter() - t:.0f}s)")
    problems = check_links(root)
    try:
        triton_pdl()
        log("Triton: gdc_wait / gdc_launch_dependents and launch_pdl=True work")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Triton PDL: {type(exc).__name__}: {exc}")
    for p in problems:
        log(f"REFUSED: {p}")
    if not problems:
        log(f"every extension under {root} links CUDA 13's runtime")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
