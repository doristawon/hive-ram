# SPDX-License-Identifier: Apache-2.0
# Portions adapted from TensorFold (Apache-2.0); modified by the hive-ram contributors.
"""Load one rank's layer range with TensorFold's own Flash Next loader.

The loader (``weights.load``) walks ``chosen = list(range(cfg.layers))``; we compile a copy of its module source with
that one line reading the range instead, so every tensor is built by the unchanged code.

The norm-storage convention (``around_one``) is then guessed from this rank's layers only; HELLO compares it across
the ranks (``proto.check_chain``), so ranks that disagree refuse to start instead of computing with both.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

from .proto import check_range, config_fingerprint

_LINE = "    chosen = list(range(cfg.layers))"


def _module(layers: list[int]):
    from tensorfold.families.qwen4_exp.cuda import weights as base

    src = inspect.getsource(base)
    if src.count(_LINE) != 1:
        raise RuntimeError("tensorfold's Flash Next loader changed: the layer list line was not found once")
    src = src.replace(_LINE, "    chosen = list(_TFPP_LAYERS)")
    mod = types.ModuleType(base.__name__ + "_tfpp")
    mod.__package__, mod.__file__ = base.__package__, base.__file__
    mod._TFPP_LAYERS = layers
    exec(compile(src, base.__file__, "exec"), mod.__dict__)
    return mod


def model_layers(model_dir) -> int:
    """The checkpoint's decoder layers (config.json), after refusing a checkpoint this loader cannot split."""

    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda.weight_types import Config

    md = Path(model_dir)
    if exl3.is_exl3(md):
        raise ValueError(f"{md} is an EXL3 pack; tfpp splits MLX and NVFP4 Flash Next checkpoints (the EXL3 loader "
                         "loads every layer whatever the range)")
    return Config.read(md).layers


def load_range(model_dir, first: int, end: int, *, mtp: bool, ple_on_ssd: bool = True,
               draft_vocab: str | int | None = None):
    """Weights for layers [first, end); the embedding, mixer and head always load (they are small)."""

    check_range(first, end, model_layers(model_dir))      # before any weight loads
    mod = _module(list(range(first, end)))
    w = mod.load(model_dir, mtp=mtp, tp=None, draft_vocab=draft_vocab if mtp else None, ple_on_ssd=ple_on_ssd)
    got = [layer.index for layer in w.layers]
    if got != list(range(first, end)):
        raise RuntimeError(f"loaded layers {got[:3]}..{got[-3:]}, wanted {first}..{end - 1}")
    w.meta["pp_range"] = (first, end)
    w.meta["pp_config"] = config_fingerprint(model_dir)
    return w
