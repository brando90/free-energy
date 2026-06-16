"""Build pilot.jsonl from pre-generated PrOntoQA-OOD data.

Selects ProofsOnly/Composed test examples with >= 3 derived steps (so early/mid/late
injection points exist), dedupes by question text, caps at 500.
"""
import json, glob, os, re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prontoqa_ood")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "pilot.jsonl")
CAP = 500

def derived_steps(cot):
    # PrOntoQA CoT alternates derived facts ("Rex is a lempus.") and rules
    # ("Numpuses are lempuses."). Derived facts mention the entity (capitalized name).
    out = []
    for i, s in enumerate(cot):
        first = s.split()[0]
        if first[0].isupper() and first.lower() not in (
            "every", "each", "all", "numpuses", "zumpuses"
        ) and not first.lower().endswith("es"):
            out.append(i)
    return out

def main():
    seen, rows = set(), []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*ProofsOnly*.json"))) + \
            sorted(glob.glob(os.path.join(DATA_DIR, "*Composed*.json")))
    for f in files:
        d = json.load(open(f))
        for key, ex in d.items():
            for sub in ["test_example"] + [f"in_context_example{i}" for i in range(8)]:
                q = ex.get(sub)
                if not q or "chain_of_thought" not in q:
                    continue
                qtext = q["question"]
                if qtext in seen:
                    continue
                cot = q["chain_of_thought"]
                dsteps = derived_steps(cot)
                # need >= 3 derived facts strictly before the final conclusion
                if len(dsteps) < 4:
                    continue
                seen.add(qtext)
                target = q["query"].replace("Prove:", "").strip()
                rows.append({
                    "id": f"{os.path.basename(f)}::{key}::{sub}",
                    "question": qtext,
                    "target": target,
                    "gold_cot": cot,
                    "derived_idx": dsteps,
                    "n_hops": len(dsteps),
                })
    rows = rows[:CAP]
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} instances to {OUT}")
    from collections import Counter
    print("hop distribution:", dict(Counter(r["n_hops"] for r in rows)))

if __name__ == "__main__":
    main()
