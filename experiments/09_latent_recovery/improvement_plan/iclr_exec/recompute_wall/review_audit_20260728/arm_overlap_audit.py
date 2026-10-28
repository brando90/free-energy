#!/usr/bin/env python3
"""Review audit 2026-07-28: E12 instruction-arm selection/overlap analysis.

Settles reviewer's post-treatment-selection attack on E12 from existing artifacts.
READ-ONLY over experiment artifacts; writes only into review_audit_20260728/.
"""
import json, os, math, hashlib, itertools
from collections import defaultdict, Counter

RW = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall"
OUT = os.path.join(RW, "review_audit_20260728")
os.makedirs(OUT, exist_ok=True)

ARMS = ["A0", "A1", "A2", "A3"]
ROSTERS = {
    "haiku": {"dir": os.path.join(RW, "e12_fixes"), "programs": os.path.join(RW, "e12_fixes", "programs.jsonl")},
    "llama8b": {"dir": os.path.join(RW, "e11_run", "results", "E12_open", "llama8b"),
                "programs": os.path.join(RW, "e11_run", "results", "E12_open", "llama8b", "programs.jsonl")},
    "qwen7b": {"dir": os.path.join(RW, "e11_run", "results", "E12_open", "qwen7b"),
               "programs": os.path.join(RW, "e11_run", "results", "E12_open", "qwen7b", "programs.jsonl")},
}

def wilson(cnt, n, z=1.96):
    if n == 0:
        return None
    p = cnt / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - e) / d, 4), round((c + e) / d, 4)]

def jl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

report = {}

# ---------------- world-file accounting (denominator for solvability) ----------
world_counts = {}
for tag, cfg in ROSTERS.items():
    ck = Counter()
    ids_by_k = defaultdict(set)
    for r in jl(cfg["programs"]):
        ck[r["k"]] += 1
        ids_by_k[r["k"]].add(r["program_id"])
    world_counts[tag] = {str(k): ck[k] for k in sorted(ck)}
report["world_counts_per_k"] = world_counts

# ---------------- per-arm evaluated (kept) sets + label tallies ----------------
# kept[tag][arm][k] = set(program_id); rows[tag][arm] = list of validated rows (slim)
kept = {t: {} for t in ROSTERS}
labels_full = {t: {} for t in ROSTERS}
rows_by = {t: {} for t in ROSTERS}   # (arm) -> list of (k, program_id, rollout, label)
for tag, cfg in ROSTERS.items():
    for arm in ARMS:
        vpath = os.path.join(cfg["dir"], arm, "validated_outputs.jsonl")
        ks = defaultdict(set)
        lab = defaultdict(Counter)
        slim = []
        for r in jl(vpath):
            k = r["k"]; pid = r["program_id"]
            ks[k].add(pid)
            lab[k][r["label"]] += 1
            slim.append((k, pid, r.get("rollout"), r["label"]))
        kept[tag][arm] = {k: ks[k] for k in ks}
        labels_full[tag][arm] = {str(k): dict(lab[k]) for k in lab}
        rows_by[tag][arm] = slim

# ---------------- overlap computation ------------------------------------------
overlap = {}
for tag in ROSTERS:
    overlap[tag] = {}
    all_ks = sorted({k for arm in ARMS for k in kept[tag][arm]})
    for k in all_ks:
        sets = {arm: kept[tag][arm].get(k, set()) for arm in ARMS}
        inter4 = set.intersection(*sets.values())
        union4 = set.union(*sets.values())
        pw = {}
        for a, b in itertools.combinations(ARMS, 2):
            pw[f"{a}&{b}"] = len(sets[a] & sets[b])
        overlap[tag][str(k)] = {
            "per_arm_kept": {a: len(sets[a]) for a in ARMS},
            "pairwise_intersection": pw,
            "fourway_intersection": len(inter4),
            "fourway_union": len(union4),
            "identical_sets_all4": all(sets[a] == sets["A0"] for a in ARMS),
            "fourway_intersection_ids_sha256": hashlib.sha256(
                ("\n".join(sorted(inter4))).encode()).hexdigest()[:16],
        }
        overlap[tag][str(k)]["_inter_ids"] = sorted(inter4)
report["overlap"] = {t: {k: {kk: vv for kk, vv in d.items() if kk != "_inter_ids"}
                         for k, d in overlap[t].items()} for t in overlap}

# ---------------- intersection reanalysis --------------------------------------
# absorbed rate per arm per k: full kept cohort vs 4-way-intersection cohort.
# Also rollout-0-only (temp-0 deterministic rollout) on the intersection.
reanalysis = {}
for tag in ROSTERS:
    reanalysis[tag] = {}
    for kstr, od in overlap[tag].items():
        k = int(kstr)
        inter = set(od["_inter_ids"])
        per_arm = {}
        for arm in ARMS:
            full = [t for t in rows_by[tag][arm] if t[0] == k]
            sub = [t for t in full if t[1] in inter]
            sub_r0 = [t for t in sub if t[2] == 0]
            def stat(rowset):
                n = len(rowset)
                ab = sum(1 for t in rowset if t[3] == "absorbed")
                return {"n": n, "absorbed": ab,
                        "rate": round(ab / n, 4) if n else None,
                        "wilson95": wilson(ab, n),
                        "labels": dict(Counter(t[3] for t in rowset))}
            per_arm[arm] = {"full": stat(full), "intersection": stat(sub),
                            "intersection_rollout0": stat(sub_r0)}
        # deltas vs A0 on both cohorts
        deltas = {}
        for arm in ["A1", "A2", "A3"]:
            f0, fi = per_arm["A0"]["full"]["rate"], per_arm["A0"]["intersection"]["rate"]
            fa, ia = per_arm[arm]["full"]["rate"], per_arm[arm]["intersection"]["rate"]
            deltas[arm] = {"full_minus_A0": (None if None in (fa, f0) else round(fa - f0, 4)),
                           "intersection_minus_A0": (None if None in (ia, fi) else round(ia - fi, 4))}
        reanalysis[tag][kstr] = {"per_arm": per_arm, "delta_vs_A0": deltas,
                                 "n_intersection_programs": len(inter)}
report["intersection_reanalysis"] = reanalysis

# ---------------- per-arm base solvability -------------------------------------
# Denominator: number of gold traces actually generated per arm per k.
# haiku: countable directly from raw_generations gold rows (110 per k, not the
# 180/k in programs.jsonl -- the runner subsampled per manifest
# n_programs_in_target_ks=330). Open models: gold rows not logged; denominator
# assumed = full world file (800/k), flagged as such.
gold_gen_counts = {}
gold_keysets = {}
for arm in ARMS:
    ck = Counter()
    keys = set()
    for r in jl(os.path.join(ROSTERS["haiku"]["dir"], arm, "raw_generations.jsonl")):
        if r.get("condition") == "gold":
            ck[r["k"]] += 1
            keys.add((r["k"], r["program_id"]))
    gold_gen_counts[arm] = {str(k): ck[k] for k in sorted(ck)}
    gold_keysets[arm] = keys
report["haiku_gold_generation_worlds"] = {
    "per_arm_per_k": gold_gen_counts,
    "same_worlds_entered_gold_phase_all4": all(gold_keysets[a] == gold_keysets["A0"] for a in ARMS),
}
solv = {}
for tag, cfg in ROSTERS.items():
    solv[tag] = {}
    for arm in ARMS:
        q = json.load(open(os.path.join(cfg["dir"], arm, "queue_status.json")))
        gc = q["gold_cohort"]
        per_k = {}
        for kstr, d in gc.items():
            if tag == "haiku":
                denom = gold_gen_counts[arm].get(kstr)
                src = "gold rows in raw_generations.jsonl"
            else:
                denom = world_counts[tag].get(kstr)
                src = "ASSUMED full world file (gold rows not logged; runner code absent on skampere2)"
            per_k[kstr] = {"eligible": d["eligible"], "kept": d["kept"],
                           "worlds_tested": denom, "denominator_source": src,
                           "solve_rate": round(d["eligible"] / denom, 4) if denom else None}
        solv[tag][arm] = per_k
report["per_arm_base_solvability"] = solv

# ---------------- haiku gold-trace identity across arms ------------------------
# golds are logged only for haiku. Compare gold continuation text across arms.
gold = {}   # arm -> {(k, pid): (text, failed)}
for arm in ARMS:
    g = {}
    for r in jl(os.path.join(ROSTERS["haiku"]["dir"], arm, "raw_generations.jsonl")):
        if r.get("condition") == "gold":
            g[(r["k"], r["program_id"])] = (r.get("continuation"), r.get("failed_generation"))
    gold[arm] = g
gold_cmp = {"n_gold_rows_per_arm": {a: len(gold[a]) for a in ARMS},
            "failed_generation_per_arm": {a: sum(1 for v in gold[a].values() if v[1]) for a in ARMS}}
pairs = {}
for a, b in itertools.combinations(ARMS, 2):
    shared = set(gold[a]) & set(gold[b])
    diff = [kp for kp in shared if gold[a][kp][0] != gold[b][kp][0]]
    by_k = Counter(kp[0] for kp in diff)
    pairs[f"{a}vs{b}"] = {"shared_gold_keys": len(shared), "n_text_differs": len(diff),
                          "differs_by_k": {str(k): by_k[k] for k in sorted(by_k)}}
gold_cmp["pairwise_gold_text_diff"] = pairs
# within 4-way kept intersection: programs whose gold text is identical in all 4 arms
ident = {}
for kstr, od in overlap["haiku"].items():
    k = int(kstr)
    inter = od["_inter_ids"]
    same = sum(1 for pid in inter
               if len({gold[a][(k, pid)][0] for a in ARMS if (k, pid) in gold[a]}) == 1
               and all((k, pid) in gold[a] for a in ARMS))
    ident[kstr] = {"intersection_n": len(inter), "gold_text_identical_all4": same}
gold_cmp["gold_identical_within_4way_intersection"] = ident
report["haiku_gold_trace_identity"] = gold_cmp

# ---------------- prompt-logging check -----------------------------------------
plog = {}
for tag, cfg in ROSTERS.items():
    kc = Counter()
    n = 0
    for arm in ARMS:
        for r in jl(os.path.join(cfg["dir"], arm, "raw_generations.jsonl")):
            kc.update(r.keys()); n += 1
    plog[tag] = {"total_raw_rows": n, "keys_seen": sorted(kc),
                 "any_prompt_field": any("prompt" in k.lower() or "message" in k.lower()
                                          or "user" in k.lower() for k in kc)}
report["raw_prompt_field_check"] = plog

with open(os.path.join(OUT, "arm_overlap_audit.json"), "w") as f:
    json.dump(report, f, indent=1, sort_keys=True)
# also dump the intersection id lists for reproducibility
with open(os.path.join(OUT, "fourway_intersection_ids.json"), "w") as f:
    json.dump({t: {k: overlap[t][k]["_inter_ids"] for k in overlap[t]} for t in overlap},
              f, indent=0, sort_keys=True)
print(json.dumps(report, indent=1, sort_keys=True))
