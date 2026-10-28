#!/usr/bin/env python3
"""Review audit 2026-07-28, part 2: program-cluster bootstrap 95% CIs for the
KEY DIFFERENCES behind the paper's qualitative claims (overlapping per-cell CIs
do not settle a difference, so we bootstrap the difference itself).

Contrasts:
  A. fig2 per model: absorbed(onehop_kc1) - absorbed(adjacent_contradiction)
     (computed vs readable gap; independent program clusters, resampled
      independently in each cell).
  B. E11 per model: absorbed(k1_bare) - absorbed(k0_bare)  (wall onset gap).
B=2000, seed=20260728. Writes diff_cis.json + prints table.
"""
import json, math, random
from collections import defaultdict

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
RW = EXP + "/improvement_plan/iclr_exec/recompute_wall"
REG2 = EXP + "/results/EXPH2_FRONTIER_FOLLOWUPS/regime2"
OUT = RW + "/review_audit_20260728"
B = 2000
SEED = 20260728


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def clusters(rows, pred):
    by = defaultdict(list)
    for r in rows:
        by[r["program_id"]].append(1 if pred(r) else 0)
    return [by[k] for k in sorted(by)]


def boot_rates(cl, rng, n_boot):
    m = len(cl)
    out = []
    for _ in range(n_boot):
        num = den = 0
        for _ in range(m):
            v = cl[rng.randrange(m)]
            den += len(v)
            num += sum(v)
        out.append(num / den)
    return out


def diff_ci(rows_a, rows_b, pred_a, pred_b):
    """bootstrap dist of rate(a) - rate(b), independent cluster resamples."""
    rng = random.Random(SEED)
    ca, cb = clusters(rows_a, pred_a), clusters(rows_b, pred_b)
    da = boot_rates(ca, rng, B)
    db = boot_rates(cb, rng, B)
    diffs = sorted(a - b for a, b in zip(da, db))
    pt = (sum(map(sum, ca)) / sum(map(len, ca))) - (sum(map(sum, cb)) / sum(map(len, cb)))
    lo = diffs[int(0.025 * (B - 1))]
    hi = diffs[int(0.975 * (B - 1))]
    return round(pt, 4), round(lo, 4), round(hi, 4)


def cf_rows(rows, cell):
    return [r for r in rows if r["condition"] == cell and r.get("cf")
            and not r.get("generation_failed")]


P_CH = lambda r: bool(r.get("final_output_absorbed"))
P_LAB = lambda r: r.get("label") == "absorbed"
res = {}

# A. fig2: computed(onehop) - readable, per model
fig2 = []
for m in ["qwen1p5b", "qwen7b", "qwen32b", "olmo7b", "llama8b"]:
    rows = read_jsonl(RW + "/e10_widen/results_skampere1/EXPG_PROGTRACE_%s/validated_outputs.jsonl" % m)
    fig2.append(("widen/" + m,
                 [r for r in rows if r["condition"] == "onehop_kc1"],
                 [r for r in rows if r["condition"] == "adjacent_contradiction"]))
for m in ["gpt-3.5-turbo-instruct", "gpt-4o", "gpt-4.1", "gpt-5.1"]:
    rows = read_jsonl(RW + "/e10_api_bridge/validated_%s.jsonl" % m)
    fig2.append(("bridge/" + m, cf_rows(rows, "onehop_kc1"), cf_rows(rows, "adjacent_contradiction")))
for m, b1dir in [("claude-opus-4-8", RW + "/e10_api_bridge"),
                 ("claude-sonnet-5", RW + "/e10_api_bridge"),
                 ("claude-haiku-4-5", REG2), ("claude-sonnet-4-5", REG2)]:
    b1 = read_jsonl(b1dir + "/validated_%s.jsonl" % m)
    b2 = read_jsonl(RW + "/e10_topup_fable/validated_%s.jsonl" % m)
    fig2.append(("claude_pooled/" + m,
                 cf_rows(b1, "onehop_kc1") + cf_rows(b2, "onehop_kc1"),
                 cf_rows(b1, "adjacent_contradiction") + cf_rows(b2, "adjacent_contradiction")))
for name, ra, rb in fig2:
    pt, lo, hi = diff_ci(ra, rb, P_CH, P_CH)
    res["fig2_computedOnehop_minus_readable/" + name] = {
        "point": pt, "boot95": [lo, hi], "excludes_0": (lo > 0 or hi < 0)}

# B. E11: k1_bare - k0_bare, per model
for name, d in [("haiku", "E11_haiku"), ("llama8b", "E11_llama8b"), ("qwen7b", "E11_qwen7b")]:
    rows = read_jsonl(RW + "/e11_run/results/%s/validated_outputs.jsonl" % d)
    ra = [r for r in rows if r["k"] == 1 and r["opacity"] == "bare"]
    rb = [r for r in rows if r["k"] == 0 and r["opacity"] == "bare"]
    pt, lo, hi = diff_ci(ra, rb, P_LAB, P_LAB)
    res["e11_k1_minus_k0/" + name] = {
        "point": pt, "boot95": [lo, hi], "excludes_0": (lo > 0 or hi < 0)}

res["_meta"] = {"B": B, "seed": SEED,
                "method": "independent program-cluster bootstrap of each cell; percentile CI of rate difference"}
with open(OUT + "/diff_cis.json", "w") as f:
    json.dump(res, f, indent=1, sort_keys=True)
for k in sorted(res):
    if k.startswith("_"):
        continue
    v = res[k]
    print("%-60s point=%+.4f  boot95=[%+.4f, %+.4f]  excludes0=%s" %
          (k, v["point"], v["boot95"][0], v["boot95"][1], v["excludes_0"]))
