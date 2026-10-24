"""The fork's functions tfpp-ds41 recompiles with a few lines changed (spec 3.1, 3.2, 3.6). No torch at import.

``patched(func, anchors, label)`` takes the function's own source (``inspect.getsource``: the LF tree, read with
universal newlines), replaces each anchor -- which must occur exactly once --, compiles it under
``from __future__ import annotations`` and runs the ``def`` in the fork module's own globals (so its names, its
relative imports and its kernels are the module's), binding the new function in a private namespace only: no fork
module is changed, no module is created, no dataclass is defined (lesson 1 does not arise), and none of the patched
functions uses a zero-argument ``super()``.

Anchors (the strings are the dedented forms; F/ = families/deepseek_v41/cuda/):
  M1  Model._forward_k (F/model.py:1325)     floor starts at the stage's replay floor
  M2  Model._forward_k (F/model.py:1419-1420) a stage without the head returns its exit state after the last post
  R1  RoundDecoder.run (F/rounds.py:600)     the round's ACT is read after the first Engram reads are submitted
  R2  RoundDecoder.run (F/rounds.py:614)     the round's ACT goes out after every stretch is enqueued
  R3  RoundRunner._forward (F/rounds.py:703) the decoder class comes from the runner
  MU1 MultiDecoder.__init__ (F/multi.py:401) the runner class comes from the decoder
F1 (only with TFPP_DS41_CUT_POST=carry): the last sublayer's post crosses the cut unapplied (spec 3.6):
  F1a/F1b RoundDecoder._layers_fused (F/rounds.py:486, 549-551); F1c/F1d Model._forward_k (F/model.py:1331-1332,
  1416-1419).
"""

from __future__ import annotations

import __future__ as _future
import ast
import inspect
import sys
import textwrap
from pathlib import Path


class LoaderChanged(RuntimeError):
    """An anchor does not occur exactly once: the fork changed under tfpp-ds41 (re-read the seam)."""


FAMILY = "families/deepseek_v41/cuda"

# id -> (file under F/, qualified name, old, new)
ANCHORS: dict[str, tuple[str, str, str, str]] = {
    "M1": ("model.py", "Model._forward_k",
           "    floor, kv_done = 0, False\n",
           "    floor, kv_done = self._tfpp_floor, False\n"),
    "M2": ("model.py", "Model._forward_k",
           "    sc.length = start + n\n    if not all_logits:\n",
           "    sc.length = start + n\n    if not self._tfpp_last:\n"
           "        return (\"cut\", h, pre, shared, floor, start, n)\n    if not all_logits:\n"),
    "R1": ("rounds.py", "RoundDecoder.run",
           "    start(-1)\n",
           "    start(-1)\n    self._tfpp_in()\n"),
    "R2": ("rounds.py", "RoundDecoder.run",
           "    return self.logits\n",
           "    self._tfpp_out()\n    return self.logits\n"),
    "R3": ("rounds.py", "RoundRunner._forward",
           "        g = RoundDecoder(m, self.pool, R, b, True, out=self.out)\n",
           "        g = self._tfpp_decoder(m, self.pool, R, b, True, out=self.out)\n"),
    "MU1": ("multi.py", "MultiDecoder.__init__",
            "    self.runner = RoundRunner(m, self.pool, graphs=graphs,\n",
            "    self.runner = self._tfpp_runner(m, self.pool, graphs=graphs,\n"),
    # -- F1 (spec 3.6) -------------------------------------------------------------------------------------------
    "F1a": ("rounds.py", "RoundDecoder._layers_fused",
            "    pending = None                                   # the last sublayer's gathered partials, not yet "
            "posted\n",
            "    pending = self._tfpp_pending_in(first)           # (tfpp-ds41 F1) the carried post at a stage's "
            "entry\n"),
    "F1b": ("rounds.py", "RoundDecoder._layers_fused",
            "    if pending is not None:\n        K.hc_post(pending, h, post, comb, h)\n    return h, pre, pre_f\n",
            "    if pending is not None and not self._tfpp_keep_pending(end, pending):\n"
            "        K.hc_post(pending, h, post, comb, h)\n    return h, pre, pre_f\n"),
    "F1c": ("model.py", "Model._forward_k",
            "    pending = None\n    self.taps_start = start\n",
            "    pending = self._tfpp_pending_in_k(post, comb)\n    self.taps_start = start\n"),
    "F1d": ("model.py", "Model._forward_k",
            "    if pending is not None:\n        with _T(\"hc\"):\n            K.hc_post(pending, h, post, comb, h)\n"
            "    sc.length = start + n\n",
            "    if pending is not None and not self._tfpp_keep_pending_k(pending, post, comb):\n"
            "        with _T(\"hc\"):\n            K.hc_post(pending, h, post, comb, h)\n    sc.length = start + n\n"),
}

# the four recompiled functions and the anchors each takes (F1's only with the carry fallback)
GROUPS: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {
    "forward_k": ("model.py", "Model._forward_k", ("M1", "M2"), ("F1c", "F1d")),
    "round_run": ("rounds.py", "RoundDecoder.run", ("R1", "R2"), ()),
    "runner_forward": ("rounds.py", "RoundRunner._forward", ("R3",), ()),
    "multi_init": ("multi.py", "MultiDecoder.__init__", ("MU1",), ()),
    "layers_fused": ("rounds.py", "RoundDecoder._layers_fused", (), ("F1a", "F1b")),
}


def apply_anchors(src: str, ids, label: str) -> str:
    for aid in ids:
        _, _, old, new = ANCHORS[aid]
        n = src.count(old)
        if n != 1:
            raise LoaderChanged(f"{label}: anchor {aid} {old.strip()[:60]!r} occurs {n} times, not once (the fork "
                                "changed: re-read the seam, python -m tfpp.ds41.pins --diff TREE)")
        src = src.replace(old, new)
    return src


def compile_def(src: str, label: str):
    return compile(src, f"<tfpp-ds41:{label}>", "exec", flags=_future.annotations.compiler_flag, dont_inherit=True)


def patched(func, ids, label: str):
    """``func`` recompiled with anchors ``ids`` applied, defined in its own module's globals."""

    src = textwrap.dedent(inspect.getsource(func))
    src = apply_anchors(src, ids, label)
    code = compile_def(src, label)
    ns: dict = {}
    exec(code, sys.modules[func.__module__].__dict__, ns)      # noqa: S102 (the fork's own source, anchors applied)
    fn = ns[func.__name__]
    fn.__qualname__ = f"tfpp_ds41.{label}"
    fn.__tfpp_anchors__ = tuple(ids)
    return fn


def group_ids(group: str, carry_post: bool) -> tuple[str, ...]:
    _, _, base, f1 = GROUPS[group]
    return tuple(base) + (tuple(f1) if carry_post else ())


# -- the same anchors on a tree's files, without importing it (tests, pins) ----------------------------------------
def function_source(path: str | Path, qualname: str) -> str:
    """The dedented source of ``qualname`` in a file (decorators included, as inspect.getsource gives it)."""

    text = Path(path).read_text(encoding="utf-8")                 # universal newlines: the CRLF work tree reads as LF
    tree = ast.parse(text)
    node: ast.AST = tree
    for part in qualname.split("."):
        node = next(c for c in getattr(node, "body", []) if isinstance(c, (ast.FunctionDef, ast.ClassDef))
                    and c.name == part)
    start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
    lines = text.splitlines(keepends=True)[start:node.end_lineno]
    return textwrap.dedent("".join(lines))


def check_tree(family_dir: str | Path, carry_post: bool = True) -> list[str]:
    """Every group's anchors on ``family_dir`` (F/): each occurs once and the patched source compiles. Returns
    "group: ok" lines; raises LoaderChanged on the first failure."""

    out = []
    for group, (file, qual, base, f1) in GROUPS.items():
        ids = tuple(base) + (tuple(f1) if carry_post else ())
        if not ids:
            continue
        src = function_source(Path(family_dir) / file, qual)
        if carry_post and group == "forward_k":                   # both sets on one function: either order
            src2 = apply_anchors(apply_anchors(src, f1, group), base, group)
            compile_def(src2, group)
        src = apply_anchors(src, ids, group)
        compile_def(src, group)
        out.append(f"{group} ({file}::{qual}): {', '.join(ids)} once each, compiles")
    return out
