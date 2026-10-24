"""Oracle drafts for T1-oracle (spec 15.6): the drafter's proposals replaced by the reference's own serial tokens,
with errors planted at positions chosen by a hash of the position alone. A shortened model's DSpark accepts almost
nothing, so without this the rounds that keep 2..5 drafts and every rollback would go untested.

``install(engine, oracle)`` replaces ``multi._drafts`` (an instance attribute: the fork's code is not touched) on the
leader or on the reference alike: each drafting stream's block is the reference reply's next tokens after its pending
one; ``drafters[N].last_conf`` is filled with zeros (F/multi.py:1053 reads it). Run with TF_DS_DEPTH_POLICY=fixed on
both sides (the drafts' confidences are not the drafter's). The oracle file is a bench output (``--bench-out``) of a
serial run (draft false) of the same prompt set: prompt sha -> its reply.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .bench import token_sha


def load_oracle(path: str | Path) -> dict:
    out = {}
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        if r.get("error") or "prompt_sha" not in r:
            continue
        out.setdefault(r["prompt_sha"], [int(x) for x in r["tokens"]])
    return out


def planted(pos: int, every: int = 4) -> bool:
    """Whether the draft for absolute position ``pos`` is made wrong (about one in ``every``)."""

    return hashlib.sha256(f"tfpp-ds41-oracle:{int(pos)}".encode()).digest()[0] % every == 0


def install(eng, oracle: dict, *, every: int = 4, log=print) -> None:
    mu = eng.multi
    if mu is None or eng.drafter is None:
        raise RuntimeError("--oracle needs the drafted --parallel path (a drafter and --parallel 2 or more)")
    cfg = eng.model.cfg
    block = int(cfg.dspark_block)
    vocab = int(cfg.vocab)
    misses = {"streams": 0}

    def stream_at(slot: int):
        for s in list(mu.streams.values()) + list(mu.filling):
            if s.st is not None and s.st.index == slot:
                return s
        return None

    def _drafts(tokens, q0, slots):
        rows = []
        for tok, p, slot in zip(tokens, q0, slots):
            s = stream_at(int(slot))
            ref = oracle.get(token_sha(s.prompt)) if s is not None else None
            if ref is None:
                misses["streams"] += 1
            j = int(p) - (len(s.prompt) if s is not None else 0)         # the pending token is reply token j
            want = ref[j + 1:j + 1 + block] if ref is not None and j >= 0 else []
            row = []
            for i in range(block):
                t = want[i] if i < len(want) else 0
                if planted(int(p) + 1 + i, every):
                    t = (t + 1) % vocab
                row.append(int(t))
            rows.append(row)
        mu.drafters[len(tokens)] = SimpleNamespace(last_conf=[[0.0] * block for _ in tokens])
        return rows

    mu._drafts = _drafts
    log(f"[tfpp-ds41] oracle drafts: {len(oracle)} reference replies, one draft in {every} planted wrong "
        "(TF_DS_DEPTH_POLICY must be fixed on both runs)")
