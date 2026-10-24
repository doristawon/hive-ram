"""The environment every tfpp-ds41 stage (and the reference) runs under, checked before any fork module is imported.

The fork's CUDA modules read their settings once, at import (F/model.py:29-65, 137-165; F/rounds.py:26-42;
F/kernels.py:22-27, 308-401; F/engine.py:21-42; F/multi.py:49-112; F/graph.py:20; tensorfold/cuda/exl3/*), so
``check_env`` refuses to run after one of them was imported, sets the values tfpp-ds41 needs where they are unset and
refuses the ones that contradict it (spec 13.1). ``settings_vector`` then reads what the imported modules actually took
(not the env strings) for HELLO's comparison (spec 13.2). Names holding KEY, TOKEN, SECRET or PASSWORD are never
digested or printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

FORK_CUDA = "tensorfold.families.deepseek_v41.cuda"
EARLY = (FORK_CUDA, "tensorfold.cuda.exl3")

# must hold on every stage: unset -> set to this, anything else -> refuse (spec 13.1)
ENFORCED = {
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",   # before torch's first cuBLAS handle (FO/__init__.py:56-62, 68-79)
    "TF_TP_WORLD": "1",
    "TF_DS_MEM_FLOOR_GIB": "0",                 # _memory_ceiling reads the WSL VM's meminfo (F/engine.py:169-189)
    "TF_DS_SPILL_GIB": "0",
    "TF_DS_KEEP": "0",                          # v1: no kept prompts (F/multi.py:101)
    "TF_DS_ENGRAM_SPLIT": "1",                  # =0 reads every Engram layer's rows at once (F/rounds.py:699)
    "TF_DS_KERNELS": "1",
    "TF_DS_KV": "native",                       # packed FP4 planes: the replica rows are their bytes
    "TENSORFOLD_NO_UPDATE_CHECK": "1",
    "TENSORFOLD_NO_LIVE": "1",
    "HF_HUB_OFFLINE": "1",
}
# must be unset: no communicator between stages, no rank cache (keyed by rank/world, sized by checkpoint / world)
UNSET = ("TF_DS_RANK_CACHE", "TF_COMM_BACKEND", "TF_NCCL_LIB", "TF_TP_MASTER", "TF_TP_PORT")
UNSET_PREFIXES = ("TF_DS_RDMA", "TF_RDMA_", "NCCL_")
# HELLO's env digest: every variable of these prefixes but each machine's own values (spec 13.2)
DIGEST_PREFIXES = ("TF_DS_", "TF_EXL3_", "TENSORFOLD_", "TFPP_DS41_", "TRITON_", "CUBLAS", "NVIDIA_TF32_",
                   "TORCH_ALLOW_TF32", "TF_TP_", "TF_SEED")
PER_NODE = frozenset({
    "TF_DS_ENGRAM", "TF_DS_TOKEN_MAP", "TF_DS_SPILL_DIR", "TF_DS_VISION_EXTRA",
    # Engram I/O mechanics: the same bytes (AIO is checked against preads at start, F/model.py:539-571)
    "TF_DS_ENGRAM_THREADS", "TF_DS_ENGRAM_PREFETCH", "TF_DS_ENGRAM_RANDOM", "TF_DS_ENGRAM_AIO", "TF_DS_ENGRAM_DIRECT",
    "TF_DS_ENGRAM_TOUCH_ASYNC",
    # observation only
    "TF_DS_TIMING", "TF_DS_ROUND_STATS", "TF_DS_HOST_TRACE", "TF_DS_CONF_LOG", "TF_DS_CHUNK_LOG",
    "TFPP_DS41_WORKSPACE_GIB", "TFPP_DS41_ALLOW_TRITON",
})
PATH_SUFFIXES = ("_DIR", "_PATH", "_HOME", "_ROOT", "_VENV", "_CONF", "_STATE", "_FILE")
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
TRITON_WANT = "3.7"


class EnvError(RuntimeError):
    pass


def loaded_fork_modules(modules=None) -> list[str]:
    modules = sys.modules if modules is None else modules
    return sorted(m for m in modules if any(m == p or m.startswith(p + ".") for p in EARLY))


def check_env(role: str, environ=None, modules=None, *, triton_version: str | None = None) -> dict:
    """Refuse an environment that differs from spec 13.1; returns the values it set. ``environ``, ``modules`` and
    ``triton_version`` stand in for os.environ, sys.modules and triton.__version__ (tests)."""

    env = os.environ if environ is None else environ
    problems: list[str] = []
    early = loaded_fork_modules(modules)
    if early:
        problems.append(f"{', '.join(early[:3])} already imported: their import-time settings were read before "
                        "tfpp-ds41's check (import order)")
    set_now = {}
    for key, want in ENFORCED.items():
        have = env.get(key)
        if have is None or not str(have).strip():
            env[key] = want
            set_now[key] = want
        elif str(have).strip().lower() != want:
            problems.append(f"{key}={have!r}: tfpp-ds41 runs with {want}")
    for key in list(env):
        if key in UNSET or key.startswith(UNSET_PREFIXES):
            problems.append(f"{key} is set: pipeline stages share no communicator and no rank cache (unset "
                            "TF_DS_RANK_CACHE, TF_DS_RDMA*, TF_RDMA_*, NCCL_*, TF_COMM_BACKEND, TF_NCCL_LIB, "
                            "TF_TP_MASTER, TF_TP_PORT)")
    cut = str(env.get("TFPP_DS41_CUT_POST", "post") or "post").strip().lower()
    if cut not in ("post", "carry"):
        problems.append(f"TFPP_DS41_CUT_POST={cut!r}: post (default) or carry (the F1 fallback)")
    if triton_version is None:
        try:
            import triton

            triton_version = triton.__version__
        except ImportError:
            triton_version = "missing"
    if ".".join(str(triton_version).split(".")[:2]) != TRITON_WANT and env.get("TFPP_DS41_ALLOW_TRITON") != "1":
        problems.append(f"Triton {triton_version}: tfpp-ds41's venv pins Triton {TRITON_WANT}.x (triton==3.7.1, "
                        "spec 1.4); TFPP_DS41_ALLOW_TRITON=1 to run anyway (every stage alike)")
    if problems:
        raise EnvError(f"tfpp-ds41 {role}: refusing to start:\n  " + "\n  ".join(problems))
    return set_now


def secret(key: str) -> bool:
    return any(w in key.upper() for w in SECRET_WORDS)


def digested(key: str) -> bool:
    """Whether HELLO's env digest takes this variable (every stage must have it alike)."""

    if not key.startswith(DIGEST_PREFIXES) or key in PER_NODE or secret(key):
        return False
    if key.startswith(("TRITON_", "TFPP_DS41_")) and key.endswith(PATH_SUFFIXES):
        return False
    return True


def env_items(environ=None) -> dict:
    env = os.environ if environ is None else environ
    return {k: str(v) for k, v in sorted(env.items()) if digested(k)}


def env_digest(environ=None) -> str:
    return hashlib.sha256(json.dumps(sorted(env_items(environ).items())).encode()).hexdigest()


def cut_post(environ=None) -> str:
    env = os.environ if environ is None else environ
    return str(env.get("TFPP_DS41_CUT_POST", "post") or "post").strip().lower()


def experts(environ=None) -> int:
    env = os.environ if environ is None else environ
    return int(env.get("TFPP_DS41_EXPERTS") or 0)


def _get(mod, name, default="missing"):
    v = getattr(mod, name, default)
    if isinstance(v, tuple):
        v = list(v)
    return v


def settings_vector(model_dir: str | Path, *, experts_k: int = 0, carry: str = "post") -> dict:
    """What every stage must have alike, read from the imported modules (spec 13.2)."""

    from tfpp.glm.env import file_sha, model_revision

    from tensorfold.families.deepseek_v41.cuda import dspark, engine as E, graph, kernels as K, markov, model as M, \
        multi, rounds

    from .plan import ModelView

    model_dir = Path(model_dir)
    return {
        "small_switches": dict(K.SMALL_SWITCHES),
        "rows": [K.DECODE_ROWS, K.ROUND_ROWS, _get(rounds, "MAX_ROWS"), _get(K, "HC_PF_RB"),
                 _get(K, "TOPK_FUSED_MAX")],
        "bucket_min": graph.BUCKET_MIN,
        "prefill_chunk": E.PREFILL_CHUNK,
        "replay": bool(E.REPLAY),
        "replay_floor": bool(M.REPLAY_FLOOR),
        "kv_quant": bool(M.KV_QUANT),
        "kernels": bool(M.KERNELS),
        "exact_mm": bool(M.EXACT_MM),
        "grouped": bool(M.GROUPED),
        "graphs_env": bool(E.GRAPHS),
        "warm": [bool(E.WARM), list(E.WARM_LENGTHS)],
        "adaptive": bool(E.ADAPTIVE),
        "l2": [bool(M.L2_PREFETCH), bool(M.SHARED_OVERLAP), M.L2_ATTN_MB, M.L2_MOE_MB, M.L2_NEXT_MB],
        "rounds": [bool(rounds.ENGRAM_SPLIT), bool(rounds.ENGRAM_TOUCH), bool(rounds.ONE_COPY),
                   bool(rounds.SHARED_OUT)],
        "keep": [_get(multi, "KEEP"), _get(multi, "KEEP_ENTRIES"), _get(multi, "KEEP_MARKS"),
                 _get(multi, "KEEP_SHRINK"), _get(multi, "KEEP_SHRINK_MIN")],
        "depth": [_get(multi, "DEPTH_POLICY"), _get(multi, "DEPTH_BY"), _get(multi, "EVEN_MARGIN"),
                  _get(multi, "EVEN_COHORT"), _get(multi, "DECODE_SHARE"), _get(multi, "QUICK_ROWS")],
        "round_ms": _get(multi, "ROUND_MS"),
        "draft_ms": _get(multi, "DRAFT_MS"),
        "eager_absorb": _get(multi, "EAGER_ABSORB"),
        "markov": [_get(markov, "ON"), _get(markov, "SPLIT"), _get(markov, "CACHE_ROWS"), _get(markov, "SUB")],
        "draft_fused": _get(dspark, "FUSED"),
        "env": env_digest(),
        "config_sha": file_sha(model_dir / "config.json"),
        "index_sha": file_sha(model_dir / "model.safetensors.index.json"),
        "selection_sha": file_sha(model_dir / "tfpp_selection.json"),
        "revision": selection_revision(model_dir) or model_revision(model_dir),
        "model": ModelView.read(model_dir).digest(),
        "experts": int(experts_k),
        "cut_post": carry,
    }


def selection_revision(model_dir) -> str | None:
    """A test model's source revision (testmodel build writes it into tfpp_selection.json)."""

    p = Path(model_dir) / "tfpp_selection.json"
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("revision") or "") or None
    except (OSError, ValueError):
        return None


# -- tfpp-ds41's own sources -----------------------------------------------------------------------------------------
GLM_IMPORTED = ("__init__.py", "wire.py", "proto.py", "compare.py", "pins.py", "env.py", "capacity.py")


def source_files() -> list[tuple[str, Path]]:
    """(name, path) of every source a stage runs: tfpp/ds41/*.py, tfpp/warm.py and the tfpp.glm modules imported
    (spec 1.3: they are under live test and may change; HELLO then refuses a chain built of two versions)."""

    here = Path(__file__).resolve().parent
    tf = here.parent
    out = [(f"ds41/{p.name}", p) for p in sorted(here.glob("*.py"))]
    out.append(("warm.py", tf / "warm.py"))
    out += [(f"glm/{n}", tf / "glm" / n) for n in GLM_IMPORTED]
    return out


def tfpp_sha() -> str:
    h = hashlib.sha256()
    for name, p in source_files():
        digest = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if p.is_file() else "missing"
        h.update(f"{name}\0{digest}\n".encode())
    return h.hexdigest()


def nvcc_version() -> str:
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return "missing"
    for line in out.splitlines():
        if "release" in line:
            return line.split("release", 1)[1].split(",")[0].strip()
    return "unknown"


def versions(pkg_sha: str) -> dict:
    import torch
    import triton

    import tensorfold

    cap = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else []
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": triton.__version__, "nvcc": nvcc_version(), "tensorfold": tensorfold.__version__, "cc": cap,
            "pkg_sha": pkg_sha, "tfpp_sha": tfpp_sha()}


def check_device() -> None:
    import torch

    if not torch.cuda.is_available():
        raise EnvError("no CUDA device")
    cap = tuple(torch.cuda.get_device_capability(0))
    if cap != (12, 1):
        raise EnvError(f"GPU compute capability {cap[0]}.{cap[1]}: tfpp-ds41 runs on sm_121 (N1X / GB10)")
