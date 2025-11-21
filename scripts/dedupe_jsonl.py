import json, sys
from collections import OrderedDict

inp = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".jsonl", ".dedup.jsonl")

seen = OrderedDict()
with open(inp, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        seen[r["sid"] if "sid" in r else r.get("id")] = line

with open(out, "w", encoding="utf-8") as f:
    for _, line in seen.items():
        f.write(line)
print(f"[dedupe] {inp} -> {out}, kept={len(seen)}")