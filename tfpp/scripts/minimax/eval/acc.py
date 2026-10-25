"""Programmatically verifiable mini-eval (arithmetic, primes, code with tests, JSON). Usage: acc.py LABEL"""
import json, re, subprocess, sys, tempfile, os, urllib.request, random, time
URL = "http://127.0.0.1:18100/v1/chat/completions"
def ask(p, n=1400, temp=0):
    body = {"model": "minimax-m31", "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), r["choices"][0]["finish_reason"], r["usage"]["completion_tokens"]
random.seed(7)
res = []
def rec(kind, ok, fin, toks): res.append((kind, bool(ok), fin, toks))
t0 = time.time()
# 1. arithmetic (12)
for _ in range(12):
    a, b = random.randint(100, 999), random.randint(100, 999)
    c, fin, tk = ask(f"What is {a} * {b}? Answer with the number only.", 1200)
    nums = re.findall(r"\d[\d,]*", c.replace(" ", ""))
    rec("arith", nums and nums[-1].replace(",", "") == str(a * b), fin, tk)
# 2. first 30 primes (1) + 2 more list tasks
P = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
c, fin, tk = ask("List the first 30 prime numbers separated by commas. Output only the numbers.", 1400)
rec("primes", [int(x) for x in re.findall(r"\d+", c)][:30] == P, fin, tk)
sq = [i * i for i in range(1, 21)]
c, fin, tk = ask("List the squares of 1 through 20 separated by commas. Output only the numbers.", 1400)
rec("squares", [int(x) for x in re.findall(r"\d+", c)][:20] == sq, fin, tk)
# 3. code with tests (6)
CODE = [
 ("Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters. Output only a python code block.", "assert is_palindrome('A man, a plan, a canal: Panama') and not is_palindrome('abc') and is_palindrome('')"),
 ("Write a Python function gcd(a, b). Output only a python code block.", "assert gcd(48, 18) == 6 and gcd(17, 5) == 1 and gcd(0, 9) == 9"),
 ("Write a Python function roman(n) converting 1..3999 to a Roman numeral. Output only a python code block.", "assert roman(4) == 'IV' and roman(1994) == 'MCMXCIV' and roman(3999) == 'MMMCMXCIX'"),
 ("Write a Python function merge_sorted(a, b) merging two sorted lists into one sorted list without using sorted(). Output only a python code block.", "assert merge_sorted([1,4,9],[2,3,10,11]) == [1,2,3,4,9,10,11] and merge_sorted([], [1]) == [1]"),
 ("Write a Python function word_freq(text) returning a dict of lowercase word counts, splitting on whitespace and stripping .,!? punctuation. Output only a python code block.", "assert word_freq('Hi, hi! There.') == {'hi': 2, 'there': 1}"),
 ("Write a Python function flatten(x) that flattens arbitrarily nested lists into one flat list. Output only a python code block.", "assert flatten([1,[2,[3,[4]],5],[]]) == [1,2,3,4,5]"),
]
for prompt, test in CODE:
    c, fin, tk = ask(prompt, 1400)
    m = re.search(r"```(?:python)?\n(.*?)```", c, re.S)
    ok = False
    if m:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(m.group(1) + "\n" + test + "\n")
        ok = subprocess.run([sys.executable, f.name], capture_output=True, timeout=30).returncode == 0
        os.unlink(f.name)
    rec("code", ok, fin, tk)
# 4. JSON (4)
for k in (3, 5, 7, 4):
    c, fin, tk = ask(f"Return a JSON array of exactly {k} objects, each with keys \"name\" (string), \"age\" (integer) and \"tags\" (array of strings). Output only the JSON.", 1400)
    m = re.search(r"\[.*\]", c, re.S)
    ok = False
    try:
        d = json.loads(m.group(0)); ok = len(d) == k and all(isinstance(o.get("name"), str) and isinstance(o.get("age"), int) and isinstance(o.get("tags"), list) for o in d)
    except Exception:
        pass
    rec("json", ok, fin, tk)
tot = {}
for kind, ok, fin, tk in res:
    t = tot.setdefault(kind, [0, 0]); t[0] += ok; t[1] += 1
print(f"[{sys.argv[1]}] " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in tot.items()) + f"  | TOTAL {sum(r[1] for r in res)}/{len(res)}  truncated {sum(r[2]=='length' for r in res)}  tokens {sum(r[3] for r in res)}  {time.time()-t0:.0f}s")
