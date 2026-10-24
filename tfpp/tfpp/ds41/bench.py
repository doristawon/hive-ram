"""Run a prompt set through an engine's ``generate`` without HTTP (``--bench PROMPTS.jsonl``): the chain's leader or
the reference (ref.py), the same way, and record the replies (spec 15.6).

Each line of the prompt file is one request:
  {"id": "r17", "rand_len": 17, "seed": 17}          token ids drawn from a fixed seed (1000 .. 99999)
  {"id": "x", "ids": [...]}                           token ids as they are
  {"id": "t", "text": "...", "thinking": false}      a user message through the checkpoint's chat template
  options: "max_tokens" (64), "draft" (true), "ignore_eos" (true), "temperature" / "top_k" / "top_p" / "min_p" /
           "seed" (greedy without temperature), "stop_after" (the client stops after that many tokens),
           "repeat" (that many times in a row), "group" (consecutive items of one group run at once, from threads:
           T1-multi)
Items whose prompt and reply do not fit the context, or longer than --bench-max-prompt, are skipped (logged), alike
in both runs.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

EFFORT_HIGH = 75            # DsApp's "Reasoning Effort" for thinking without an effort (F/app.py:17-37)


def load_prompts(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            if not any(k in item for k in ("text", "messages", "ids", "rand_len")):
                raise ValueError(f"{path}:{i}: a prompt needs text, messages, ids or rand_len")
            item.setdefault("id", f"p{i}")
            items.append(item)
    return items


def rand_ids(n: int, seed: int) -> list[int]:
    import numpy as np

    return [int(x) for x in np.random.default_rng(int(seed)).integers(1000, 100000, int(n))]


class Renderer:
    """Messages -> token ids through the checkpoint's chat template and tokenizer (the HTTP path's rendering)."""

    def __init__(self, model_dir) -> None:
        self.model_dir = Path(model_dir)
        self.tok = None
        self.template = None

    def ids(self, item: dict) -> list[int]:
        if "ids" in item:
            return [int(t) for t in item["ids"]]
        if "rand_len" in item:
            return rand_ids(item["rand_len"], item.get("seed", item["rand_len"]))
        if self.template is None:
            from tokenizers import Tokenizer

            from tensorfold.cuda.chat_template import ChatTemplate

            self.tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            self.template = ChatTemplate(self.model_dir)
        messages = list(item.get("messages") or [{"role": "user", "content": item["text"]}])
        thinking = bool(item.get("thinking", False))
        text = self.template.render(messages, tools=None, enable_thinking=thinking,
                                    extra={"reasoning_effort": EFFORT_HIGH} if thinking else None)
        return self.tok.encode(text, add_special_tokens=False).ids

    def text(self, ids: list[int]) -> str:
        if self.tok is None:
            return ""
        return self.tok.decode(ids, skip_special_tokens=False)


def sampling_of(item: dict):
    temperature = float(item.get("temperature", 0.0) or 0.0)
    if temperature <= 0:
        return None
    from tensorfold.engine.exact_sampling import Sampling

    return Sampling(int(item.get("seed", 0)), temperature, int(item.get("top_k", 20)), float(item.get("top_p", 0.95)),
                    float(item.get("min_p", 0.0)))


def token_sha(ids) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:16]


def _one(eng, item: dict, ids: list[int], rep: int, log, r: Renderer) -> dict:
    got: list[int] = []
    stop_after = int(item.get("stop_after", 0) or 0)

    def on_tokens(toks):
        got.extend(int(t) for t in toks)
        return bool(stop_after) and len(got) >= stop_after

    eng.request.stop_eos = not bool(item.get("ignore_eos", True))
    t = time.perf_counter()
    res = {"id": item["id"], "rep": rep, "prompt_tokens": len(ids), "prompt_sha": token_sha(ids)}
    try:
        stats = eng.generate(ids, int(item.get("max_tokens", 64)), sampling_of(item), on_tokens,
                             draft=bool(item.get("draft", True)))
    except Exception as exc:  # noqa: BLE001
        if getattr(eng, "broken", None):
            raise                                 # the chain broke: fail-stop
        log(f"[tfpp-ds41] bench {item['id']}#{rep}: failed ({type(exc).__name__}: {exc}); going on")
        res.update(tokens=got, error=f"{type(exc).__name__}: {exc}")
        return res
    wall = time.perf_counter() - t
    res.update(tokens=got, token_sha=token_sha(got), wall_s=round(wall, 3), stats=stats)
    log(f"[tfpp-ds41] bench {item['id']}#{rep}: {len(ids)} prompt + {len(got)} tokens in {wall:.2f}s, rounds "
        f"{(stats or {}).get('rounds')}, accepted {(stats or {}).get('accepted')}/{(stats or {}).get('drafted')} | "
        f"{r.text(got)[:60]!r}")
    return res


def run(eng, items: list[dict], *, model_dir, out_path: str | None = None, context: int | None = None,
        max_prompt: int | None = None, log=print) -> list[dict]:
    r = Renderer(model_dir)
    limit = int(context or getattr(eng, "limit", 0) or 0)
    plan = []
    for item in items:
        ids = r.ids(item)
        mt = int(item.get("max_tokens", 64))
        if (max_prompt and len(ids) > max_prompt) or (limit and len(ids) + mt + 16 > limit):
            log(f"[tfpp-ds41] bench {item['id']}: skipped ({len(ids)} prompt tokens; context {limit}, max prompt "
                f"{max_prompt})")
            continue
        for rep in range(int(item.get("repeat", 1))):
            plan.append((item, ids, rep))
    results: list[dict] = []
    i = 0
    while i < len(plan):
        g = plan[i][0].get("group")
        if g is None:
            results.append(_one(eng, plan[i][0], plan[i][1], plan[i][2], log, r))
            i += 1
            continue
        j = i
        while j < len(plan) and plan[j][0].get("group") == g:
            j += 1
        box: list = [None] * (j - i)
        errors: list = []

        def worker(k: int) -> None:
            item, ids, rep = plan[i + k]
            try:
                box[k] = _one(eng, item, ids, rep, log, r)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(j - i)]
        for t in threads:
            t.start()
            time.sleep(0.05)                     # arrival order: the group's order (admissions stay in it)
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        results += box
        i = j
    if out_path:
        Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return results
