#!/usr/bin/env python
"""E6 — Legacy-gap decomposition by measured refutation distance (ICLR item E6, reviewer Q1).

Inputs (read-only):
  stage0_distance/row_level_distances.jsonl  (per-row measured d, problem_id; retro audit)
  stage0_strict/strict_rows.jsonl            (per-row strict + permissive DVs)

Outputs (written to improvement_plan/iclr_exec/e6_legacy_gap_decomposition/):
  q1_decomposition.json   all numbers
  (REPORT.md is written separately)

Zero GPU. Deterministic (seed below).
"""
import json
import os
import sys
import random
from collections import defaultdict

BASE = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan"
OUT = os.path.join(BASE, "iclr_exec", "e6_legacy_gap_decomposition")
os.makedirs(OUT, exist_ok=True)

N_BOOT = 10000
SEED = 20260722
DVS = ["closure_valid", "doubt", "hop_sound_valid", "srr_task"]  # permissive x2, strict x2
DV_LABEL = {
    "closure_valid": "closure-valid (permissive)",
    "doubt": "doubt (permissive)",
    "hop_sound_valid": "hop-sound valid (strict)",
    "srr_task": "SRR task (strict)",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def rate(rows, dv):
    if not rows:
        return None
    return sum(1 for r in rows if r[dv]) / len(rows)


def d_bin4(row):
    """Bin measured d into {0, 1, 2+, unreachable}."""
    d = row["d"]
    if d is None:
        return "unreachable"
    if d == 0:
        return "0"
    if d == 1:
        return "1"
    return "2+"


def cluster_boot_rate(rows, dv, n_boot, seed):
    """Percentile 95% CI for a rate, cluster bootstrap over problem_id."""
    rng = random.Random(seed)
    clus = defaultdict(list)
    for r in rows:
        clus[r["problem_id"]].append(r[dv])
    pids = sorted(clus)
    stats = []
    for _ in range(n_boot):
        num = den = 0
        for pid in (pids[rng.randrange(len(pids))] for _ in range(len(pids))):
            vals = clus[pid]
            num += sum(vals)
            den += len(vals)
        stats.append(num / den if den else float("nan"))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(int(0.975 * n_boot), n_boot - 1)]
    return lo, hi


def cluster_boot_diff(rows_a, rows_b, dv, n_boot, seed, paired):
    """Percentile 95% CI for rate(A) - rate(B), cluster bootstrap over problem_id.

    paired=True: A and B share the same problem set; resample problems once per
    replicate and take both arms from the same resampled problems (preserves
    within-problem correlation). paired=False: independent resampling per arm.
    """
    rng = random.Random(seed)
    ca, cb = defaultdict(list), defaultdict(list)
    for r in rows_a:
        ca[r["problem_id"]].append(r[dv])
    for r in rows_b:
        cb[r["problem_id"]].append(r[dv])
    diffs = []
    if paired:
        pids = sorted(set(ca) | set(cb))
        for _ in range(n_boot):
            na = da = nb = db = 0
            for pid in (pids[rng.randrange(len(pids))] for _ in range(len(pids))):
                va, vb = ca.get(pid, ()), cb.get(pid, ())
                na += sum(va); da += len(va)
                nb += sum(vb); db += len(vb)
            if da and db:
                diffs.append(na / da - nb / db)
    else:
        pa, pb = sorted(ca), sorted(cb)
        for _ in range(n_boot):
            na = da = 0
            for pid in (pa[rng.randrange(len(pa))] for _ in range(len(pa))):
                va = ca[pid]; na += sum(va); da += len(va)
            nb = db = 0
            for pid in (pb[rng.randrange(len(pb))] for _ in range(len(pb))):
                vb = cb[pid]; nb += sum(vb); db += len(vb)
            if da and db:
                diffs.append(na / da - nb / db)
    diffs.sort()
    n = len(diffs)
    return diffs[int(0.025 * n)], diffs[min(int(0.975 * n), n - 1)]


def main():
    dist = load_jsonl(os.path.join(BASE, "stage0_distance", "row_level_distances.jsonl"))
    strict = load_jsonl(os.path.join(BASE, "stage0_strict", "strict_rows.jsonl"))

    # --- join (EXPA only; run_id unique within EXPA, verified) ---
    sd = {r["run_id"]: r for r in strict if r["family"].startswith("EXPA:")}
    assert len(sd) == 1800, f"expected 1800 unique EXPA strict rows, got {len(sd)}"
    rows = []
    for r in dist:
        if r["dataset"] != "EXPA" or r["family"] not in ("one_hop_falsehood", "global_falsehood"):
            continue
        s = sd[r["run_id"]]
        assert s["family"] == "EXPA:" + r["family"]
        assert s["doubt"] == r["verbalized_doubt"]
        rows.append({
            "run_id": r["run_id"],
            "family": r["family"],
            "problem_id": r["problem_id"],
            "position": r["position"],
            "d": r["d"],
            "dbin": d_bin4(r),
            "closure_valid": bool(s["closure_valid"]),
            "doubt": bool(s["doubt"]),
            "hop_sound_valid": bool(s["hop_sound_valid"]),
            "srr_task": bool(s["srr_task"]),
        })
    one = [r for r in rows if r["family"] == "one_hop_falsehood"]
    glo = [r for r in rows if r["family"] == "global_falsehood"]
    assert len(one) == 450 and len(glo) == 450

    # --- (a) reproduction against published anchors ---
    anchors = {  # (family_rows, dv): (published_rate, published_count, source)
        ("one", "closure_valid"): (0.671, 302, "STRICT_METRICS_REPORT EXPA:one_hop_falsehood pooled"),
        ("one", "hop_sound_valid"): (0.300, 135, "STRICT_METRICS_REPORT"),
        ("one", "doubt"): (0.336, None, "STRICT_METRICS_REPORT"),
        ("one", "srr_task"): (0.251, 113, "STRICT_METRICS_REPORT"),
        ("glo", "closure_valid"): (0.393, 177, "STRICT_METRICS_REPORT / RETRO 0.3933"),
        ("glo", "hop_sound_valid"): (0.213, 96, "STRICT_METRICS_REPORT"),
        ("glo", "doubt"): (0.013, None, "STRICT_METRICS_REPORT / RETRO 0.0133"),
        ("glo", "srr_task"): (0.002, 1, "STRICT_METRICS_REPORT"),
    }
    repro = {}
    ok = True
    for (fam, dv), (pub, cnt, src) in anchors.items():
        rr = one if fam == "one" else glo
        got = rate(rr, dv)
        gotc = sum(1 for r in rr if r[dv])
        match = abs(got - pub) < 0.0016 and (cnt is None or gotc == cnt)
        ok &= match
        repro[f"{fam}.{dv}"] = {"recomputed": round(got, 4), "count": gotc, "published": pub,
                                "published_count": cnt, "match": match, "source": src}
        print(f"[repro] {fam:>3} {dv:<16} recomputed {got:.4f} ({gotc}) vs published {pub} ({cnt}) -> {'OK' if match else 'MISMATCH'}")
    # retro addendum anchors (one-hop re-binned by measured d)
    d0 = [r for r in one if r["dbin"] == "0"]
    d1 = [r for r in one if r["dbin"] == "1"]
    retro_checks = {
        "d0_n": (len(d0), 67), "d1_n": (len(d1), 345),
        "d0_doubt": (round(rate(d0, "doubt"), 3), 0.284),
        "d1_doubt": (round(rate(d1, "doubt"), 3), 0.368),
        "d0_closure": (round(rate(d0, "closure_valid"), 3), 0.552),
        "d1_closure": (round(rate(d1, "closure_valid"), 3), 0.690),
    }
    for k, (got, pub) in retro_checks.items():
        match = abs(got - pub) < 0.0015
        ok &= match
        print(f"[repro-retro] {k}: recomputed {got} vs RETRO addendum {pub} -> {'OK' if match else 'MISMATCH'}")
    if not ok:
        print("ANCHOR MISMATCH — aborting before bootstrap", file=sys.stderr)
        sys.exit(1)

    # paired vs independent bootstrap
    paired = {r["problem_id"] for r in one} == {r["problem_id"] for r in glo}
    print(f"[design] problem sets identical across families: {paired} "
          f"(one-hop {len({r['problem_id'] for r in one})}, global {len({r['problem_id'] for r in glo})})")

    # --- (b) re-bin by measured d ---
    bins = ["0", "1", "2+", "unreachable"]
    by_bin = {}
    seed_i = SEED
    for b in bins:
        br = [r for r in rows if r["dbin"] == b]
        ent = {"n": len(br), "n_problems": len({r["problem_id"] for r in br}),
               "families": sorted({r["family"] for r in br})}
        for dv in DVS:
            seed_i += 1
            lo, hi = cluster_boot_rate(br, dv, N_BOOT, seed_i)
            ent[dv] = {"rate": round(rate(br, dv), 4), "count": sum(1 for r in br if r[dv]),
                       "ci95": [round(lo, 4), round(hi, 4)]}
        by_bin[b] = ent
        print(f"[bin d={b}] n={ent['n']} " + " ".join(f"{dv}={ent[dv]['rate']}" for dv in DVS))

    # --- (c)+(d) contrasts with cluster-bootstrap CIs ---
    contrasts = {}
    specs = {
        "full_cell": (one, "all one-hop rows (as published), incl. 14.9% d=0"),
        "excl_d0": ([r for r in one if r["dbin"] != "0"], "one-hop rows with measured d>=1 (d=0 contamination excluded)"),
        "d1_only": ([r for r in one if r["dbin"] == "1"], "strictly-d=1 one-hop rows (purest designed contrast)"),
    }
    for name, (sub, desc) in specs.items():
        ent = {"desc": desc, "n_one_hop": len(sub), "n_global": len(glo)}
        for dv in DVS:
            seed_i += 1
            ra, rb = rate(sub, dv), rate(glo, dv)
            lo, hi = cluster_boot_diff(sub, glo, dv, N_BOOT, seed_i, paired)
            ent[dv] = {"one_hop": round(ra, 4), "global": round(rb, 4),
                       "gap": round(ra - rb, 4), "gap_ci95": [round(lo, 4), round(hi, 4)]}
        contrasts[name] = ent
        print(f"[contrast {name}] n={len(sub)} " +
              " ".join(f"{dv}: {ent[dv]['gap']:+.4f} [{ent[dv]['gap_ci95'][0]:+.4f},{ent[dv]['gap_ci95'][1]:+.4f}]" for dv in DVS))

    # per-position supplement for the excl_d0 contrast (JSON only)
    pos_sup = {}
    for pos in ("early", "mid", "late"):
        po = [r for r in one if r["dbin"] != "0" and r["position"] == pos]
        pg = [r for r in glo if r["position"] == pos]
        pos_sup[pos] = {"n_one_hop_excl_d0": len(po), "n_global": len(pg)}
        for dv in DVS:
            pos_sup[pos][dv] = {"one_hop": round(rate(po, dv), 4), "global": round(rate(pg, dv), 4),
                                "gap": round(rate(po, dv) - rate(pg, dv), 4)}

    out = {
        "generated_by": "iclr_exec/e6_legacy_gap_decomposition (item E6)",
        "seed": SEED, "n_boot": N_BOOT,
        "bootstrap": "cluster bootstrap over problem_id, percentile 95% CI; paired resampling"
                     if paired else "independent per-arm cluster bootstrap over problem_id",
        "inputs": ["stage0_distance/row_level_distances.jsonl", "stage0_strict/strict_rows.jsonl"],
        "dv_labels": DV_LABEL,
        "reproduction_anchors": repro,
        "retro_addendum_anchors": {k: {"recomputed": v[0], "published": v[1]} for k, v in retro_checks.items()},
        "by_measured_d_bin": by_bin,
        "one_hop_vs_global_contrasts": contrasts,
        "excl_d0_by_position": pos_sup,
        "notes": [
            "All 900 rows are audited-false plants (verified in join).",
            "d bins: 0 / 1 / 2+ pool one-hop rows by measured d; 'unreachable' = the global_falsehood cell (all 450 rows measure d=inf per RETRO audit, 0 contamination).",
            "DVs recomputed from strict_rows.jsonl; doubt identical to row_level_distances.verbalized_doubt (0 disagreements).",
        ],
    }
    with open(os.path.join(OUT, "q1_decomposition.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", os.path.join(OUT, "q1_decomposition.json"))


if __name__ == "__main__":
    main()
