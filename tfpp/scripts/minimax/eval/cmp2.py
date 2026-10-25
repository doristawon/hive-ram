import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
tot = same = 0
for k in a:
    x, y = a[k], b[k]; n = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
    tot += 1; same += (x == y)
    print(f"{'IDENTICAL' if x == y else 'diverges at char %d of %d/%d' % (n, len(x), len(y))}  | {k[:60]}")
print(f"{same}/{tot} identical")
