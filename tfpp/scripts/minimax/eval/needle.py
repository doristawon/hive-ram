"""Needle-in-a-haystack check for long contexts, run inside the tunnel namespace next to the API.

    python3 needle.py 8000 32000 96000      prompt sizes in tokens (measured with the server's /tokenize)

Each prompt is non-repeating filler text with three access codes planted at 10 %, 50 % and 90 % of the way through; the
model is asked for all three. The filler is seeded, so a run is repeatable. One request per size, temperature 0, no
thinking. Prints the real prompt size, the time to the answer and which codes came back exactly.
"""
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:18100"
MODEL = "minimax-m31"

SUBJ = ["The harbour authority", "A visiting surveyor", "The regional council", "Our maintenance crew", "The night shift",
        "An independent auditor", "The orchard cooperative", "A retired engineer", "The ferry operator", "The county archive",
        "A group of students", "The weather station", "The municipal library", "A delivery driver", "The research team"]
VERB = ["recorded", "reviewed", "postponed", "approved", "measured", "rebuilt", "catalogued", "inspected", "relocated",
        "summarised", "questioned", "scheduled", "repainted", "documented", "forwarded"]
OBJ = ["the northern pier", "last spring's invoices", "a set of old maps", "the cold storage room", "three broken gauges",
       "the east stairwell", "an unmarked crate", "the annual budget", "a stack of survey forms", "the water meter readings",
       "the village bridge", "a faded shipping manifest", "the second warehouse", "several loose floorboards", "the archive index"]
TAIL = ["before the end of the month", "after a long discussion", "without any objection", "during the heavy rain",
        "in the presence of two witnesses", "using the older equipment", "while the road was closed", "at the request of the owner",
        "twice in the same week", "in spite of the delay", "following the new guidelines", "ahead of the winter season"]
NAMES = ["Cobalt Heron", "Amber Quill", "Velvet Anchor", "Silent Harbor", "Copper Finch", "Marble Lantern", "Hollow Compass",
         "Crimson Ledger", "Pale Meridian", "Iron Orchard", "Tidal Sparrow", "Quiet Furnace"]


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def sentence(rng):
    return f"{rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(TAIL)}, and {rng.choice(SUBJ).lower()} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(TAIL)}."


def count_tokens(text):
    return post("/tokenize", {"model": MODEL, "prompt": text})["count"]


def build(target, seed):
    rng = random.Random(seed)
    sample = " ".join(sentence(rng) for _ in range(300))
    per_sentence = count_tokens(sample) / 300
    n = int(target / per_sentence)
    sents = [sentence(rng) for _ in range(n)]
    names = rng.sample(NAMES, 3)
    codes = [str(rng.randint(100000, 999999)) for _ in range(3)]
    for depth, name, code in zip((0.10, 0.50, 0.90), names, codes):
        sents.insert(int(len(sents) * depth), f"NOTE: the access code for the {name} vault is {code}.")
    ask = ("Read the text above. List the access codes for the " + ", the ".join(names) +
           " vaults, one per line, as 'NAME: CODE'. Answer with the three lines only.")
    return "\n".join(sents) + "\n\n" + ask, names, codes


def run(target, seed):
    prompt, names, codes = build(target, seed)
    t = time.time()
    r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 120,
                                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    dt = time.time() - t
    text = r["choices"][0]["message"].get("content") or ""
    hits = [c in text for c in codes]
    print(f"target {target:>6}  prompt {r['usage']['prompt_tokens']:>6} tok  {dt:6.1f}s  codes {sum(hits)}/3 "
          f"(10%:{'ok' if hits[0] else 'MISS'} 50%:{'ok' if hits[1] else 'MISS'} 90%:{'ok' if hits[2] else 'MISS'})")
    if sum(hits) < 3:
        print("  expected:", ", ".join(f"{n}: {c}" for n, c in zip(names, codes)))
        print("  answer  :", text.strip().replace("\n", " | ")[:300])
    return sum(hits) == 3


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [8000, 32000, 96000]
    ok = [run(s, 1000 + i) for i, s in enumerate(sizes)]
    print("needle:", "PASS" if all(ok) else "FAIL", f"({sum(ok)}/{len(ok)} prompts with all three codes)")
