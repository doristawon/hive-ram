"""The environment every tfpp-glm stage runs under, checked before any MIA GLM module is imported.

Several MIA modules read their settings once, at import (G/engine.py:17, 38, 61, 65; G/multi.py:147; G/kda.py:65;
G/latent.py:702; G/__init__.py:5), so ``check_env`` refuses to run after one of them was imported, sets the values
tfpp-glm needs where they are unset, and refuses the ones that contradict it (spec 12.1). ``settings_vector`` then
reads what the imported modules actually took (not the env strings) for HELLO's comparison (spec 11.2).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

GLM_CUDA = "tensorfold.families.glm5_next.cuda"

# must hold on every stage: unset -> set to this, anything else -> refuse (spec 12.1)
ENFORCED = {
    "TF_GLM_HC_SPLIT": "0",            # world=1 raises (G/hcsplit.py:147-149)
    "TF_GLM_PREFILL_OVERLAP": "0",
    "TF_GLM_FILL_BUDGET_MS": "0",      # compute_steps / _time_prompt_layers would run prompt forwards (G/multi.py:573)
    "TF_GLM_MULTI_LONE": "0",
    "TF_GLM_MULTI_VERIFY": "batched",
    "TF_GLM_MULTI_ASYNC": "0",
    "TF_GLM_MULTI_DEPTH": "policy",
    "TF_GLM_MULTI_PROFILE": "0",
    "TF_GLM_DISPLAY_KV_MIB": "0",
    "TF_GLM_KEPT_BYTES_GIB": "0",
}
# must be unset: no communicator between pipeline stages (G/engine.py:544-557), no draft dump
UNSET = ("TF_GLM_COMM", "TF_GLM_DRAFT_DUMP")
UNSET_PREFIXES = ("TF_ROCE_", "NCCL_")
# compared in HELLO's env digest: everything of these prefixes but each machine's own values. Beside the packages'
# own switches: the process-wide numerics knobs of Triton (e.g. TRITON_DEFAULT_FP_FUSION, TRITON_F32_DEFAULT), cuBLAS
# and TF32, which would change one stage's bits with HELLO and BUILT still passing
DIGEST_PREFIXES = ("TF_GLM_", "TENSORFOLD_", "TFPP_", "TRITON_", "CUBLAS", "NVIDIA_TF32_", "TORCH_ALLOW_TF32")
PER_NODE = frozenset({"TFPP_WARM_MS", "TFPP_TRACE", "TFPP_TIMING", "TF_GLM_CACHE_GIB",
                      "TENSORFOLD_MEMORY_RESERVE_GIB", "TFPP_ALLOW_TRITON", "TFPP_FAULT", "TFPP_RESERVE_GIB"})
PER_NODE_PREFIXES = ("TFPP_GLM_",)     # the scripts' own (folders, venv)
PER_NODE_PATH_PREFIXES = ("TRITON_",)  # their folders and tool paths (TRITON_CACHE_DIR, TRITON_HOME, TRITON_PTXAS_PATH)
PATH_SUFFIXES = ("_DIR", "_PATH", "_HOME")
TRITON_PTX = "3.7"                     # the release MIA's CUDA ports reproduce (G/sparse.py:135-142)


class EnvError(RuntimeError):
    pass


def loaded_glm_modules(modules=None) -> list[str]:
    modules = sys.modules if modules is None else modules
    return sorted(m for m in modules if m == GLM_CUDA or m.startswith(GLM_CUDA + "."))


def check_env(role: str, environ=None, modules=None, *, triton_version: str | None = None) -> dict:
    """Refuse a stage whose environment differs from spec 12.1; returns the values it set. ``environ``,
    ``modules`` and ``triton_version`` stand in for os.environ, sys.modules and triton.__version__ (tests)."""

    env = os.environ if environ is None else environ
    problems: list[str] = []
    early = loaded_glm_modules(modules)
    if early:
        problems.append(f"{', '.join(early[:3])} already imported: their import-time settings were read before "
                        "tfpp-glm's check (import order)")
    set_now = {}
    for key, want in ENFORCED.items():
        have = env.get(key)
        if have is None or not str(have).strip():
            env[key] = want
            set_now[key] = want
        elif str(have).strip().lower() != want:
            problems.append(f"{key}={have!r}: tfpp-glm runs with {want}")
    for key in list(env):
        if key in UNSET or key.startswith(UNSET_PREFIXES):
            problems.append(f"{key} is set: pipeline stages share no communicator (unset TF_GLM_COMM, TF_ROCE_*, "
                            "NCCL_*, TF_GLM_DRAFT_DUMP)")
    if triton_version is None:
        try:
            import triton

            triton_version = triton.__version__
        except ImportError:
            triton_version = "missing"
    if ".".join(str(triton_version).split(".")[:2]) != TRITON_PTX and env.get("TFPP_ALLOW_TRITON") != "1":
        problems.append(f"Triton {triton_version}: MIA's CUDA ports need Triton {TRITON_PTX}.x (G/sparse.py:135-142); "
                        "without it every port falls back to Triton and prompt numerics leave the MIA package's")
    if problems:
        raise EnvError(f"tfpp-glm {role}: refusing to start:\n  " + "\n  ".join(problems))
    return set_now


def digested(key: str) -> bool:
    """Whether HELLO's env digest takes this variable (every stage must have it alike)."""

    if not key.startswith(DIGEST_PREFIXES) or key in PER_NODE or key.startswith(PER_NODE_PREFIXES):
        return False
    return not (key.startswith(PER_NODE_PATH_PREFIXES) and key.endswith(PATH_SUFFIXES))


def env_digest(environ=None) -> str:
    env = os.environ if environ is None else environ
    items = sorted((k, str(v)) for k, v in env.items() if digested(k))
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def env_items(environ=None) -> dict:
    env = os.environ if environ is None else environ
    return {k: str(v) for k, v in sorted(env.items()) if digested(k)}


def model_revision(model_dir: str | Path) -> str:
    """The checkpoint's Hub commit, so every stage reads the same repo and revision (an abliterated and an original
    checkpoint share config.json and the weight map: their sha256 alone cannot tell them apart). From what
    ``hf download --local-dir`` leaves (.cache/huggingface/download/config.json.metadata: commit, etag, time), or a
    Hub cache path (.../snapshots/<commit>/); "unknown" without either (then every stage must say unknown)."""

    model_dir = Path(model_dir)
    meta = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"
    try:
        first = meta.read_text(encoding="utf-8").splitlines()[0].strip()
        if len(first) == 40 and all(ch in "0123456789abcdef" for ch in first):
            return first
    except (OSError, IndexError):
        pass
    snap = model_dir.resolve()
    if snap.parent.name == "snapshots" and len(snap.name) == 40:
        return snap.name
    return "unknown"


def file_sha(path: str | Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"


def _f64_ints(x: float) -> list[int]:
    import struct

    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def settings_vector(model_dir: str | Path) -> dict:
    """What every stage must have alike, read from the imported modules (MIA's own comparison vector, G/engine.py:
    686-698, without capacity / spare / mtp_on / the drafter flag, which the leader decides; plus what it leaves out:
    TF_GLM_DENSE, the KDA / latent / prompt-tile / CUDA-port / L2PF switches, the env digest, the checkpoint config
    and its Hub revision)."""

    from tensorfold.cuda.geometry import PREFILL_ROWS
    from tensorfold.cuda.sampling import union_cover
    from tensorfold.families.glm5_next.cuda import LATENT, engine as E, exl3_mm, forward, kda, kv8, l2pf, latent, \
        multi, qmm, sparse
    from tensorfold.families.glm5_next.cuda.copy_drafts import CopySettings
    from tensorfold.families.glm5_next.cuda.hcsplit import SplitSettings
    from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings
    from tensorfold.families.glm5_next.cuda.sliced_fill import budget_ms
    from tensorfold.families.glm5_next.cuda.weights import Config

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    prefill_rows = int(os.environ.get("TF_GLM_PREFILL_ROWS", "") or PREFILL_ROWS)
    kvb = qmm.kvb_kind() if cfg.quant == "exl3" and LATENT else "bf16"
    copy = CopySettings.from_env(E.MAX_ROWS - 1, pad=E.WIDE_GRAPHS)
    return {
        "latent": int(LATENT),
        "prefill_rows_env": prefill_rows,
        "grid": E.prompt_grid(prefill_rows),
        "kvb": qmm.KVB_KINDS.index(kvb),
        "copy": copy.code() if copy is not None else [0, 0, 0, 0],
        "split": SplitSettings.from_env().code(),
        "draft_quant": qmm.DRAFT_QUANTS.index(qmm.draft_quant()),
        "shared_prefix": E.shared_prefix(),
        "draft_ring": int(E.DRAFT_RING),
        "multi": [("batched", "serial").index(multi.verify_kind())] + MultiSettings.from_env().code()
                 + _f64_ints(budget_ms()) + [E.multi_window()],
        "union_cover": int(union_cover()),
        "max_rows": E.MAX_ROWS,
        "wide_graphs": list(E.WIDE_GRAPHS),
        "kv": kv8.KINDS.index(kv8.kv_kind()),
        "prelaunch": int(multi.PRELAUNCH),
        "dense": qmm.dense_kind(),
        "kda_chunked": int(kda.CHUNKED),
        "sparse_onepass": int(latent.SPARSE_ONEPASS),
        "prompt_tiles": int(forward.PROMPT_TILES),
        "triton_ptx": int(sparse.triton_matches_ptx()),
        "exl3_prompt": int(exl3_mm.prompt_kernels()),
        "l2pf": l2pf.Settings.from_env().code(),
        "dflash_policy": E.DFLASH_POLICY,
        "env": env_digest(),
        "config_sha": file_sha(model_dir / "config.json"),
        "index_sha": file_sha(model_dir / "model.safetensors.index.json"),
        "revision": model_revision(model_dir),
        "quant": cfg.quant,
        "layers": cfg.layers,
    }


def tfpp_sha() -> str:
    """tfpp-glm's own sources (this package and tfpp/warm.py), hashed like pins.tree_sha."""

    here = Path(__file__).resolve().parent
    files = sorted(p for p in here.glob("*.py")) + [here.parent / "warm.py"]
    h = hashlib.sha256()
    for p in files:
        if p.is_file():
            h.update(f"{p.name}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def versions(pkg_sha: str) -> dict:
    import torch
    import triton
    import tensorfold

    cap = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else []
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": triton.__version__, "tensorfold": tensorfold.__version__, "cc": cap,
            "pkg_sha": pkg_sha, "tfpp_sha": tfpp_sha()}


def check_device() -> None:
    import torch

    if not torch.cuda.is_available():
        raise EnvError("no CUDA device")
    cap = tuple(torch.cuda.get_device_capability(0))
    if cap != (12, 1):
        raise EnvError(f"GPU compute capability {cap[0]}.{cap[1]}: tfpp-glm runs on sm_121 (N1X / GB10)")
