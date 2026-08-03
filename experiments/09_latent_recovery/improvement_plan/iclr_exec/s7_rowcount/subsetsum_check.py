import json, collections, itertools
p = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/stage0_strict/strict_rows.jsonl"
rows = [json.loads(l) for l in open(p)]
fam = collections.Counter(r["family"] for r in rows)
fams = sorted(fam.items())
counts = [v for _, v in fams]
names = [k for k, _ in fams]

def subset_hits(target, cap=8):
    hits = []
    n = len(counts)
    for r in range(1, n + 1):
        for combo in itertools.combinations(range(n), r):
            if sum(counts[i] for i in combo) == target:
                hits.append([names[i] for i in combo])
                if len(hits) >= cap:
                    return hits
    return hits

for tgt in [4857, 4539]:
    h = subset_hits(tgt)
    print(f"target {tgt}: {len(h)} family-subset solutions" + ("" if h else "  <-- NO clean subset"))
    for s in h[:4]:
        print("   ", len(s), "fams:", s)
