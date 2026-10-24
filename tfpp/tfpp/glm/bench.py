"""Run a prompt set through the leader's engine without HTTP (``--bench PROMPTS.jsonl``) and record the replies.

Each line of the prompt file is one request:
  {"id": "zh-short", "text": "..."}                   a user message (or "messages": [...], or "ids": [...])
  optional: "text_repeat" (the text that many times: long prompts), "system" + "system_repeat" (a system message
            before the messages: shared prefixes), "max_tokens" (128), "draft" (true), "policy" ("fnc7:0.3",
            "c3:0.35", "0", ...), "thinking" (false),
            "temperature" / "top_k" / "top_p" / "min_p" / "seed" (greedy without temperature), "stop_after" (end the
            reply after that many tokens: StopVote), "repeat" (run it that many times in a row)
Images and structured output go through HTTP (T1-http); this runner does not prepare pictures or grammars.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def load_prompts(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            if not any(k in item for k in ("text", "messages", "ids")):
                raise ValueError(f"{path}:{i}: a prompt needs text, messages or ids")
            item.setdefault("id", f"p{i}")
            items.append(item)
    return items


class Renderer:
    """Messages -> token ids through the checkpoint's own chat template (TF/cuda/chat_template.py) and tokenizer."""

    def __init__(self, model_dir) -> None:
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.template = None

    def ids(self, item: dict) -> list[int]:
        if "ids" in item:
            return [int(t) for t in item["ids"]]
        if self.template is None:
            from tensorfold.cuda.chat_template import ChatTemplate

            self.template = ChatTemplate(self.model_dir)
        messages = list(item.get("messages") or
                        [{"role": "user", "content": item["text"] * int(item.get("text_repeat", 1))}])
        if item.get("system"):
            messages = [{"role": "system", "content": item["system"] * int(item.get("system_repeat", 1))}] + messages
        text = self.template.render(messages, tools=None, enable_thinking=bool(item.get("thinking", False)))
        if not item.get("thinking", False):
            from tensorfold.families.glm5_next.prompts import thinking_off

            text = thinking_off(text)
        return self.tok.encode(text, add_special_tokens=False).ids

    def text(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


def sampling_of(item: dict):
    temperature = float(item.get("temperature", 0.0) or 0.0)
    if temperature <= 0:
        return None
    from tensorfold.engine.exact_sampling import Sampling

    return Sampling(int(item.get("seed", 0)), temperature, int(item.get("top_k", 20)), float(item.get("top_p", 0.95)),
                    float(item.get("min_p", 0.0)))


def run(eng, items: list[dict], *, model_dir, out_path: str | None = None, log=print) -> list[dict]:
    r = Renderer(model_dir)
    results = []
    for item in items:
        ids = r.ids(item)
        for rep in range(int(item.get("repeat", 1))):
            got: list[int] = []
            stop_after = int(item.get("stop_after", 0) or 0)

            def on_tokens(toks, got=got, stop_after=stop_after):
                got.extend(int(t) for t in toks)
                return bool(stop_after) and len(got) >= stop_after

            eng.request.policy = item.get("policy")
            eng.request.stop_eos = not bool(item.get("ignore_eos", False))
            t = time.perf_counter()
            try:
                stats = eng.generate(ids, int(item.get("max_tokens", 128)), sampling_of(item), on_tokens,
                                     draft=bool(item.get("draft", True)))
            except Exception as exc:  # noqa: BLE001
                if getattr(eng, "broken", None):
                    raise                         # the chain broke: fail-stop
                log(f"[tfpp-glm] bench {item['id']}#{rep}: failed ({type(exc).__name__}: {exc}); going on")
                results.append({"id": item["id"], "rep": rep, "prompt_tokens": len(ids), "tokens": got,
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            wall = time.perf_counter() - t
            res = {"id": item["id"], "rep": rep, "prompt_tokens": len(ids), "tokens": got,
                   "token_sha": hashlib.sha256(json.dumps(got).encode()).hexdigest()[:16], "wall_s": round(wall, 3),
                   "stats": stats}
            results.append(res)
            dec = stats.get("decode_s") or 0.0
            log(f"[tfpp-glm] bench {item['id']}#{rep}: {len(ids)} prompt + {len(got)} tokens in {wall:.2f}s"
                f"{f', decode {(len(got) - 1) / dec:.1f} tok/s' if dec and len(got) > 1 else ''}"
                f", rounds {stats.get('rounds')}, accepted {stats.get('accepted')}/{stats.get('drafted')} | "
                f"{r.text(got)[:60]!r}")
    if out_path:
        Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return results
