"""Build a training set: N programs spread equally over --depths, 2 examples each."""
import argparse, json, random
from common import gen_program, build_example, user_message

ap = argparse.ArgumentParser()
ap.add_argument("--depths", required=True)  # e.g. 1 or 1,2,3
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed0", type=int, default=1_000_000)  # eval seeds are < 100000
a = ap.parse_args()
depths = [int(x) for x in a.depths.split(",")]
rows = []
for i in range(a.n):
    d = depths[i % len(depths)]
    seed = a.seed0 + i
    p = gen_program(d, seed)
    for kind in ("perturbed", "control"):
        pre, tgt = build_example(p, kind, seed)
        rows.append({"depth": d, "seed": seed, "kind": kind, "user": user_message(p),
                     "prefix": pre, "target": tgt})
random.Random(0).shuffle(rows)
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "examples", a.out)
