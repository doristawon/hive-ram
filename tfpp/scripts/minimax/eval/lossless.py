"""Greedy chat outputs for a fixed prompt set; saved as JSON for comparing a speculative server against a plain one."""
import json, sys, urllib.request
PROMPTS = [
    "Write a Python function that merges two sorted lists, with a docstring and three assert tests.",
    "Explain in three short paragraphs why the sky is blue and why sunsets are red.",
    "Return a JSON array of 5 fictional books with title, author, year and a one-sentence summary. Only JSON.",
    "Solve step by step: a train leaves at 9:40 going 84 km/h; another leaves the same station at 10:10 going 110 km/h. When does the second catch up?",
    "Translate into French, German and Japanese: 'The meeting has been moved to Thursday afternoon.'",
    "Write a SQL query that returns the top 3 customers by total order value in 2025, with a short explanation.",
    "Summarize the plot of Romeo and Juliet in exactly four sentences.",
    "List the first 30 prime numbers separated by commas.",
]
out = {}
for p in PROMPTS:
    body = {"model": "minimax-m31", "messages": [{"role": "user", "content": p}], "max_tokens": 220, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18100/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    m = r["choices"][0]["message"]
    out[p] = (m.get("reasoning_content") or m.get("reasoning") or "") + "␞" + (m.get("content") or "")
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("saved", sys.argv[1], sum(len(v) for v in out.values()), "chars")
