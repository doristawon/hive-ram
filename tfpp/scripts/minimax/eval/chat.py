import json, sys, time, urllib.request
def chat(msg, n=200, think=False):
    body = {"model": "minimax-m31", "messages": [{"role": "user", "content": msg}], "max_tokens": n, "temperature": 0}
    if not think: body["chat_template_kwargs"] = {"enable_thinking": False}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18100/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    m = r["choices"][0]["message"]; u = r["usage"]
    return (m.get("reasoning_content") or m.get("reasoning") or "")[:300], (m.get("content") or "")[:400], u["completion_tokens"], time.time() - t
for q in ["What is the capital of France? Answer in one word.", "What is 17 * 23?", "Write a Python function is_prime(n).", "Translate to French: The weather is nice today."]:
    rs, c, n, dt = chat(q)
    print(f"Q: {q}\n  reasoning: {rs!r}\n  answer: {c!r}\n  ({n} tokens, {dt:.1f}s)")
