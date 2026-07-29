#!/usr/bin/env python3
"""Review audit 2026-07-28: program-level cluster bootstrap reanalysis of the
recompute-wall headline cells, from existing per-rollout validated artifacts.

For each headline cell we recompute the absorbed rate two ways:
  (a) rollout-level Wilson 95% CI (paper convention, z=1.96), and
  (b) program-level cluster bootstrap: resample program_ids with replacement
      (same #clusters), B=2000, percentile 2.5/97.5 CI, seed=20260728.
Plus per-program rates (heterogeneity: #programs at 0%, at 100%, in between)
and greedy (rollout==0) vs sampled (rollout>0) split where rollout exists.

Denominator conventions replicate each arm's own analyzer:
  - E10 widen (open models): ALL rows in the condition (unparsed and
    generation-failed retained), pred = truthy final_output_absorbed
    [expg house convention; cross-checked vs summary_tables.json].
  - E10 bridge (GPT + opus/sonnet-5 batch1): rows with condition match,
    cf truthy, not generation_failed  [e10_bridge_analyze.py cf_rows()].
  - E10 topup pooled (Claude): batch1 (bridge or regime2 prefill) + batch2
    (topup), same cf/not-gen-failed filter  [e10topup_analyze.py cell_stats()].
  - E11 (haiku + open): rows with (k, opacity), pred = label=="absorbed"
    (three-way label)  [e11_run.py _cell_summary].

Everything is read-only except writes into review_audit_20260728/.
"""
import json, math, os, random
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / float(n)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def cluster_boot(rows, pred, n_boot=B, seed=SEED):
    """Percentile CI from resampling program_id clusters with replacement."""
    by = defaultdict(list)
    for r in rows:
        by[r["program_id"]].append(1 if pred(r) else 0)
    ids = sorted(by)
    if not ids:
        return (None, None)
    rng = random.Random(seed)
    m = len(ids)
    vals = []
    for _ in range(n_boot):
        num = den = 0
        for _ in range(m):
            v = by[ids[rng.randrange(m)]]
            den += len(v)
            num += sum(v)
        vals.append(num / den if den else float("nan"))
    vals = sorted(v for v in vals if not math.isnan(v))
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return (round(lo, 4), round(hi, 4))


def per_program(rows, pred):
    by = defaultdict(list)
    for r in rows:
        by[r["program_id"]].append(1 if pred(r) else 0)
    rates = {pid: sum(v) / len(v) for pid, v in by.items()}
    n0 = sum(1 for v in rates.values() if v == 0.0)
    n1 = sum(1 for v in rates.values() if v == 1.0)
    return rates, n0, n1, len(rates) - n0 - n1


def cell_summary(rows, pred, has_rollout):
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    rate = k / n if n else None
    rates, p0, p100, pmid = per_program(rows, pred)
    rec = {
        "n_rollouts": n,
        "n_programs": len(rates),
        "absorbed_k": k,
        "rate": round(rate, 4) if rate is not None else None,
        "wilson95": wilson(k, n),
        "cluster_boot95": cluster_boot(rows, pred),
        "programs_at_0pct": p0,
        "programs_at_100pct": p100,
        "programs_intermediate": pmid,
    }
    if has_rollout:
        g = [r for r in rows if r.get("rollout") == 0]
        s = [r for r in rows if r.get("rollout", 0) > 0]
        kg = sum(1 for r in g if pred(r))
        ks = sum(1 for r in s if pred(r))
        rec["greedy_n"] = len(g)
        rec["greedy_rate"] = round(kg / len(g), 4) if g else None
        rec["sampled_n"] = len(s)
        rec["sampled_rate"] = round(ks / len(s), 4) if s else None
    else:
        rec["greedy_n"] = n
        rec["greedy_rate"] = rec["rate"]  # arm is greedy-only (1 rollout/program)
        rec["sampled_n"] = 0
        rec["sampled_rate"] = None
    return rec


results = {"_meta": {
    "B": B, "seed": SEED,
    "date": "2026-07-28",
    "metric": "absorbed (final_output_absorbed for E10 arms; label=='absorbed' for E11)",
    "cluster": "program_id, resampled with replacement, percentile CI",
}}
xchecks = []

CELLS3 = ["adjacent_contradiction", "onehop_kc1", "deep_kc5"]
CELL_LABEL = {"adjacent_contradiction": "readable",
              "onehop_kc1": "computed_onehop",
              "deep_kc5": "computed_deep_k5"}

# ---------------- 1. E10 widen: open models, greedy-only -------------------
WIDEN = ["qwen1p5b", "qwen7b", "qwen32b", "olmo7b", "llama8b"]
for m in WIDEN:
    d = RW + "/e10_widen/results_skampere1/EXPG_PROGTRACE_" + m
    rows = read_jsonl(d + "/validated_outputs.jsonl")
    summ = json.load(open(d + "/summary_tables.json"))["cells"]
    for c in CELLS3:
        crows = [r for r in rows if r["condition"] == c]
        if not crows:
            continue
        rec = cell_summary(crows, lambda r: bool(r.get("final_output_absorbed")), False)
        rec["gen_failed_in_denom"] = sum(1 for r in crows if r.get("generation_failed"))
        key = "widen/%s/%s" % (m, c)
        results[key] = rec
        pc = summ.get(c, {}).get("final_output_absorbed", {})
        xchecks.append({"cell": key, "recomputed": [rec["absorbed_k"], rec["n_rollouts"], rec["rate"]],
                        "paper_summary": [pc.get("count"), pc.get("n"), pc.get("rate")],
                        "match": (pc.get("count") == rec["absorbed_k"] and pc.get("n") == rec["n_rollouts"])})

# ---------------- 2. E10 bridge: GPT models, greedy-only -------------------
BRIDGE_GPT = ["gpt-3.5-turbo-instruct", "gpt-4o", "gpt-4.1", "gpt-5.1"]
bridge_analysis = json.load(open(RW + "/e10_api_bridge/bridge_analysis.json"))


def cf_rows(rows, cell):
    return [r for r in rows if r["condition"] == cell and r.get("cf")
            and not r.get("generation_failed")]


for m in BRIDGE_GPT:
    rows = read_jsonl(RW + "/e10_api_bridge/validated_%s.jsonl" % m)
    for c in CELLS3:
        crows = cf_rows(rows, c)
        if not crows:
            continue
        rec = cell_summary(crows, lambda r: bool(r.get("final_output_absorbed")), False)
        key = "bridge/%s/%s" % (m, c)
        results[key] = rec
        try:
            pa = bridge_analysis["models"][m]["cells"][c]
        except (KeyError, TypeError):
            pa = {}
        xchecks.append({"cell": key, "recomputed": [rec["absorbed_k"], rec["n_rollouts"], rec["rate"]],
                        "paper_summary": pa.get("absorbed") if isinstance(pa.get("absorbed"), (int, float, dict)) else pa,
                        "match": None})

# ---------------- 3. Claude pooled (batch1 + batch2), greedy-only ----------
CLAUDE = [("claude-opus-4-8", RW + "/e10_api_bridge"),
          ("claude-sonnet-5", RW + "/e10_api_bridge"),
          ("claude-haiku-4-5", REG2),
          ("claude-sonnet-4-5", REG2)]
topup_analysis = json.load(open(RW + "/e10_topup_fable/topup_analysis.json"))
for m, b1dir in CLAUDE:
    b1 = read_jsonl(b1dir + "/validated_%s.jsonl" % m)
    b2 = read_jsonl(RW + "/e10_topup_fable/validated_%s.jsonl" % m)
    for c in CELLS3:
        crows = cf_rows(b1, c) + cf_rows(b2, c)
        if not crows:
            continue
        rec = cell_summary(crows, lambda r: bool(r.get("final_output_absorbed")), False)
        b1ids = {r["program_id"] for r in cf_rows(b1, c)}
        b2ids = {r["program_id"] for r in cf_rows(b2, c)}
        rec["batch_overlap_programs"] = len(b1ids & b2ids)
        key = "claude_pooled/%s/%s" % (m, c)
        results[key] = rec
        pc = topup_analysis["job1_pooled"].get(m, {}).get("cells", {}).get(c, {}).get("pooled", {})
        xchecks.append({"cell": key, "recomputed": [rec["absorbed_k"], rec["n_rollouts"], rec["rate"]],
                        "paper_summary": [pc.get("absorbed"), pc.get("n_cf"), pc.get("rate")],
                        "match": (pc.get("absorbed") == rec["absorbed_k"] and pc.get("n_cf") == rec["n_rollouts"])})

# ---------------- 4. E11 depth cells (pooled greedy + sampled) -------------
E11 = [("haiku", "E11_haiku"), ("llama8b", "E11_llama8b"), ("qwen7b", "E11_qwen7b")]
for name, d in E11:
    rows = read_jsonl(RW + "/e11_run/results/%s/validated_outputs.jsonl" % d)
    summ = json.load(open(RW + "/e11_run/results/%s/summary_tables.json" % d))["cells"]
    for kk in (0, 1):
        crows = [r for r in rows if r["k"] == kk and r["opacity"] == "bare"]
        if not crows:
            continue
        rec = cell_summary(crows, lambda r: r.get("label") == "absorbed", True)
        key = "e11/%s/k%d_bare" % (name, kk)
        results[key] = rec
        pc = summ.get("k%d_bare" % kk, {}).get("absorbed", {})
        xchecks.append({"cell": key, "recomputed": [rec["absorbed_k"], rec["n_rollouts"], rec["rate"]],
                        "paper_summary": [pc.get("count"), summ.get("k%d_bare" % kk, {}).get("n"), pc.get("rate")],
                        "match": (pc.get("count") == rec["absorbed_k"] and summ.get("k%d_bare" % kk, {}).get("n") == rec["n_rollouts"])})

results["_crosschecks_vs_paper_summaries"] = xchecks

with open(OUT + "/cluster_boot_results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)

# ---------------- markdown table -------------------------------------------
def fmt_ci(ci):
    if not ci or ci[0] is None:
        return "-"
    return "[%.4f, %.4f]" % (ci[0], ci[1])


lines = ["# Cluster-bootstrap reanalysis of recompute-wall headline cells",
         "",
         "B=%d, seed=%d, cluster=program_id, percentile CI. Metric = absorbed." % (B, SEED),
         "Greedy-only arms (widen/bridge/claude_pooled): 1 rollout per program, so",
         "cluster bootstrap == row bootstrap there; it differs from Wilson only in method.",
         "",
         "| cell | n_roll | n_prog | rate | Wilson95 | clusterBoot95 | greedy | sampled | prog@0% | prog@100% | prog mid |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for key in sorted(k for k in results if not k.startswith("_")):
    r = results[key]
    lines.append("| %s | %d | %d | %.4f | %s | %s | %s | %s | %d | %d | %d |" % (
        key, r["n_rollouts"], r["n_programs"], r["rate"],
        fmt_ci(r["wilson95"]), fmt_ci(r["cluster_boot95"]),
        ("%.4f" % r["greedy_rate"]) if r["greedy_rate"] is not None else "-",
        ("%.4f" % r["sampled_rate"]) if r["sampled_rate"] is not None else "-",
        r["programs_at_0pct"], r["programs_at_100pct"], r["programs_intermediate"]))
lines.append("")
lines.append("## Cross-checks vs stored paper summaries")
for x in xchecks:
    lines.append("- %s: recomputed k/n/rate=%s paper=%s match=%s" % (
        x["cell"], x["recomputed"], x["paper_summary"], x["match"]))
with open(OUT + "/cluster_boot_table.md", "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
