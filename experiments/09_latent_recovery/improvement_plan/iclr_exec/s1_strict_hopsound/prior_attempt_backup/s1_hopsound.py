#!/usr/bin/env python
"""S1 - Strict / hop-sound pass over EXPD + EXPE, and re-adjudication of H2'
under hop-soundness. Zero GPU; reads results/* read-only; writes ONLY into
improvement_plan/iclr_exec/s1_strict_hopsound/.

Adapts the STAGE-0 strict metric (improvement_plan/stage0_strict/strict_metrics.py,
byte-identical to the published cluster original) to the EXPD/EXPE row schema, and
re-runs the EXPD H2' TOST (aff_false vs neg_false) with the DV swapped from the
paper's closure_valid (= stored valid_recovery) to hop_sound_valid, reusing the
published EXPD confirmatory estimator (expd_confirmatory_analysis.py) so the cohort,
paired-family TOST and family-cluster bootstrap are IDENTICAL to the registered test.

SCHEMA MAPPING DECISIONS (legacy strict_metrics -> EXPD/EXPE), all documented in
the JSON under "mapping_decisions":
  * closure_valid  := stored row 'valid_recovery' (the paper's registered DV; mval()
                      in expd_confirmatory_analysis reads exactly this). Verified to
                      equal analyze()'s recomputed valid_rederivation class at
                      agreement 1.0000 on 4000 EXPD rows (anchor A1).
  * hop_sound_valid := valid_recovery AND analyze().hop_sound_all. Because recomputed
                      closure == stored closure (1.0000), this is a strict subset of
                      the paper's closure_valid.
  * is_false (audited-false plant): EXPD -> audited_truth_status=='false';
                      EXPE -> injected_statement_truth_status=='false' (EXPE integrity
                      guarantees all planted falsehoods audited false).
  * cluster unit:   EXPD -> family_id (matches confirmatory cluster_boot); EXPE -> problem_id.
  * cell:           EXPD -> 'condition' (encodes polarity x truth x claim x d);
                      EXPE -> 'condition' (arm|position), pooled to arm@position.
  * doubt/echo/corrective/targeted/deriv: recomputed by analyze() exactly as in the
                      legacy strict metric (V.DOUBT regex etc.), NOT the stored fields,
                      so SRR/stated-reliance use the identical definitions as the
                      published STRICT_METRICS_REPORT.md.
"""
import os, sys, json, math, random, argparse, datetime
from collections import defaultdict

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
os.environ.setdefault("LR_ROOT", EXP)
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expd"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "stage0_strict"))
sys.path.insert(0, os.path.join(EXP, "src"))

import expd_confirmatory_analysis as CA   # noqa: E402
import strict_metrics as SM               # noqa: E402
import validator as V                     # noqa: E402

OUT_DIR = os.path.join(EXP, "improvement_plan", "iclr_exec", "s1_strict_hopsound")
os.makedirs(OUT_DIR, exist_ok=True)
EXPD_DIR = os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT")
EXPE_DIR = os.path.join(EXP, "results", "EXPE_EVIDENCE_MOVER")


# ----------------------------------------------------------------- primitives
def strict_primitives(m, r):
    """Run the legacy analyze() on a stored row; return the strict per-row flags.
    Uses stored valid_recovery for the closure conjunct (the paper's DV)."""
    if r.get("failed_generation"):
        return {"closure_valid": 0, "hop_sound_all": 0, "hop_sound_valid": 0,
                "goal_jump0": 0, "goal_jump_strict": 0, "doubt": 0, "targeted": 0,
                "corrective": 0, "echo": 0, "deriv": 0, "used_any": 0,
                "srr_task": 0, "srr_plan": 0, "stated_reliance": 0,
                "recomp_valid": 0, "recomp_agrees": 1}
    a = SM.analyze(m["question"], m.get("prefix_steps", []), m["injected_statement"],
                   r.get("continuation", ""), m["target"], m["entity"])
    stored_valid = int(bool(r.get("valid_recovery")))
    recomp_valid = int(a["class"] == "valid_rederivation")
    hop_all = int(bool(a["hop_sound_all"]))
    jump0 = int(stored_valid == 1 and a.get("inter_facts", 0) == 0)
    echo = int(bool(a["echo"]))
    deriv = int(bool(a["deriv_use"] or a["deriv_use_rule"]))
    doubt = int(bool(a["doubt"]))
    corrective = int(bool(a["corrective"]))
    targeted = int(bool(a["targeted"]))
    flag_or_correct = int(doubt or corrective)
    return {
        "closure_valid": stored_valid,
        "hop_sound_all": hop_all,
        "hop_sound_valid": int(stored_valid == 1 and hop_all == 1),
        "goal_jump0": jump0,
        "goal_jump_strict": int(jump0 == 1 and hop_all == 0),
        "doubt": doubt, "targeted": targeted, "corrective": corrective,
        "echo": echo, "deriv": deriv,
        "used_any": int(echo or deriv),
        "srr_task": int(stored_valid == 1 and flag_or_correct == 1 and jump0 == 0),   # gated on is_false at aggregation
        "srr_plan": int(stored_valid == 1 and (targeted or corrective) == 1 and jump0 == 0 and echo == 0),
        "stated_reliance": int(stored_valid == 1 and echo == 1),
        "recomp_valid": recomp_valid,
        "recomp_agrees": int(stored_valid == recomp_valid),
        "a_star_kind": a.get("a_star_kind"),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cluster_boot(rows, key, cluster_field, seed=0, n_boot=1000):
    ids = sorted({r[cluster_field] for r in rows})
    if not ids:
        return [None, None]
    by = defaultdict(list)
    for r in rows:
        by[r[cluster_field]].append(int(bool(r[key])))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        num = den = 0
        for cid in (rng.choice(ids) for _ in ids):
            num += sum(by[cid]); den += len(by[cid])
        vals.append(num / den if den else float("nan"))
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return [None, None]
    return [round(vals[int(0.025 * (len(vals) - 1))], 4),
            round(vals[int(0.975 * (len(vals) - 1))], 4)]


def rate_block(rows, key, cluster_field, ci=False, denom=None):
    use = rows if denom is None else denom
    n = len(use)
    k = sum(int(bool(r[key])) for r in use)
    out = {"count": k, "n": n, "rate": round(k / n, 4) if n else None}
    if ci and n:
        out["wilson95"] = wilson(k, n)
        out["cluster_boot95"] = cluster_boot(use, key, cluster_field)
    return out


# ----------------------------------------------------------------- EXPD load
def load_expd():
    rows, manifest = CA.load_rows(EXPD_DIR)      # attaches stated_complement, valid_recovery, _entity
    cohort = CA.read_jsonl(os.path.join(EXPD_DIR, "gold_cohort.jsonl"))
    joint, prefix_same = CA.attr_cohort_maps(cohort)
    CA.annotate_cohort(rows, joint, prefix_same)
    agree = tot = 0
    for r in rows:
        m = manifest.get(r["run_id"])
        prim = strict_primitives(m, r)
        r["_prim"] = prim
        r["is_false"] = int(r.get("audited_truth_status") == "false")
        agree += prim["recomp_agrees"]; tot += 1
        # expose flags at top level for CA.mval monkeypatch
        r["_hop_sound_valid"] = prim["hop_sound_valid"]
    return rows, manifest, {"closure_recompute_agreement": round(agree / tot, 6), "n": tot}


# ----------------------------------------------------------------- EXPE load
def load_expe():
    manifest = {r["run_id"]: r for r in CA.read_jsonl(os.path.join(EXPE_DIR, "manifest.jsonl"))}
    rows = []
    for r in CA.read_jsonl(os.path.join(EXPE_DIR, "validated_outputs.jsonl")):
        m = manifest.get(r["run_id"])
        if not m:
            continue
        prim = strict_primitives(m, r)
        r["_prim"] = prim
        # EXPE: planted falsehood always audited false (integrity guarantee)
        ts = m.get("injected_statement_truth_status") or m.get("falsehood_status_augmented")
        r["is_false"] = int(ts == "false")
        r["_stated_complement"] = int(bool(r.get("stated_complement_of_falsehood")))
        r["_cell"] = "%s@%s" % (r.get("arm", r.get("condition")), r["injection_position"])
        rows.append(r)
    return rows, manifest


# ----------------------------------------------------------------- cell summary
def summarize_cells(rows, cell_field, cluster_field, extra_keys=()):
    cells = {}
    for cell in sorted({r[cell_field] for r in rows}):
        sub = [r for r in rows if r[cell_field] == cell]
        false_sub = [r for r in sub if r["is_false"]]
        valid_sub = [r for r in sub if r["_prim"]["closure_valid"]]
        c = {"n": len(sub), "cluster_n": len({r[cluster_field] for r in sub}),
             "n_false_audited": len(false_sub)}
        # flatten prim into row-level for rate_block
        for r in sub:
            for k, v in r["_prim"].items():
                r[k] = v
        for key, ci in [("closure_valid", True), ("hop_sound_valid", True),
                        ("goal_jump0", False), ("goal_jump_strict", False),
                        ("doubt", False), ("used_any", False), ("echo", False),
                        ("deriv", False), ("stated_reliance", False)]:
            c[key] = rate_block(sub, key, cluster_field, ci)
        # hop-unsound share among valid
        c["hop_unsound_given_valid"] = rate_block(valid_sub, "goal_jump_strict", cluster_field) if valid_sub else {"n": 0}
        c["hop_sound_given_valid"] = {"n": len(valid_sub),
                                      "count": sum(r["hop_sound_all"] for r in valid_sub),
                                      "rate": round(sum(r["hop_sound_all"] for r in valid_sub) / len(valid_sub), 4) if valid_sub else None}
        # SRR + stated-reliance over audited-false plants
        if false_sub:
            c["srr_task"] = rate_block(false_sub, "srr_task", cluster_field, ci=True)
            c["srr_plan"] = rate_block(false_sub, "srr_plan", cluster_field)
            c["stated_reliance_false"] = rate_block(false_sub, "stated_reliance", cluster_field)
            c["doubt_false"] = rate_block(false_sub, "doubt", cluster_field)
            c["corrective_false"] = rate_block(false_sub, "corrective", cluster_field)
        else:
            c["srr_task"] = c["srr_plan"] = c["stated_reliance_false"] = "n/a (no audited-false)"
        for ek in extra_keys:
            if ek in sub[0]:
                c[ek] = rate_block(sub, ek, cluster_field)
        cells[cell] = c
    return cells


# ----------------------------------------------------------------- H2' re-adjudication
def h2prime(rows):
    """Reproduce EXPD H2' with metric=closure_valid (anchor) then hop_sound_valid."""
    # monkeypatch mval to understand hop_sound_valid
    orig_mval = CA.mval

    def mval2(r, metric):
        if metric == "hop_sound_valid":
            return int(bool(r.get("_hop_sound_valid")))
        return orig_mval(r, metric)
    CA.mval = mval2

    a_cells = ["aff_false_attr_d1", "aff_false_attr_d3"]
    b_cells = ["neg_false_attr_d1", "neg_false_attr_d3"]
    rows_a = CA.cells_rows(rows, a_cells, cohort_only=True)
    rows_b = CA.cells_rows(rows, b_cells, cohort_only=True)

    def run(metric):
        ps = CA.paired_family_stats(rows_a, rows_b, metric)
        t = CA.tost(ps)
        boot_a = cluster_boot(rows_a, metric_key(metric), "family_id")
        boot_b = cluster_boot(rows_b, metric_key(metric), "family_id")
        return {"metric": metric,
                "n_rows_aff": ps["n_rows_a"], "n_rows_neg": ps["n_rows_b"],
                "paired_families": ps["n_families_paired"],
                "rate_aff_false": round(ps["rate_a"], 4), "rate_neg_false": round(ps["rate_b"], 4),
                "boot95_aff": boot_a, "boot95_neg": boot_b,
                "mean_diff": t["mean_diff"], "se": t["se"], "df": t["df"],
                "ci95": t["ci95"], "ci90_equiv": t["ci90"],
                "t_lower": t["t_lower"], "t_upper": t["t_upper"],
                "p_lower": t["p_lower"], "p_upper": t["p_upper"], "p_raw": t["p_raw"],
                "tost_pass_at_0.05": (t["p_raw"] is not None and t["p_raw"] < 0.05),
                "tost_pass_at_holm_alpha_over_6": (t["p_raw"] is not None and t["p_raw"] < 0.05 / 6),
                "margin": t["margin"], "note": t.get("note")}

    res = {"closure_valid_ANCHOR": run("closure_valid"),
           "hop_sound_valid": run("hop_sound_valid")}
    CA.mval = orig_mval
    return res


def metric_key(metric):
    # rows carry _hop_sound_valid; closure_valid is via valid_recovery. For cluster_boot
    # we need a boolean row key; add a mirror for closure_valid.
    return "_hop_sound_valid" if metric == "hop_sound_valid" else "valid_recovery"


# ----------------------------------------------------------------- legacy anchor
def legacy_anchor():
    """Anchor A2: re-run the authoritative strict metric on EXPA + legacy, confirm
    published hop-sound numbers (EXPA one_hop pooled 0.300, legacy negstep 0.300)."""
    notes = {}
    devnull = open(os.devnull, "w")
    fams = {}
    expa = SM.load_expa(os.path.join(EXP, "results", "EXPA_GLOBAL_EXPANSION"), notes, devnull, "EXPA")
    legacy = SM.load_legacy(EXP, notes, devnull)
    devnull.close()
    out = {}
    for label, rows in [("EXPA", expa), ("legacy", legacy)]:
        byfam = defaultdict(list)
        for r in rows:
            byfam[r["family"]].append(r)
        for fam, rr in byfam.items():
            n = len(rr); k = sum(int(bool(x["hop_sound_valid"])) for x in rr)
            cv = sum(int(bool(x["closure_valid"])) for x in rr)
            out[fam] = {"n": n, "hop_sound_valid": round(k / n, 4), "closure_valid": round(cv / n, 4)}
    return {k: out[k] for k in out if k in (
        "EXPA:one_hop_falsehood", "EXPA:global_falsehood", "EXPA:benign_paraphrase",
        "legacy:negstep", "legacy:falsehood")}


# ----------------------------------------------------------------- main
def main():
    result = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "mapping_decisions": {
                  "closure_valid": "stored valid_recovery (paper DV; mval reads this)",
                  "hop_sound_valid": "valid_recovery AND analyze().hop_sound_all",
                  "is_false_EXPD": "audited_truth_status=='false'",
                  "is_false_EXPE": "injected_statement_truth_status=='false'",
                  "cluster_unit_EXPD": "family_id", "cluster_unit_EXPE": "problem_id",
                  "cell_EXPD": "condition", "cell_EXPE": "condition (arm@position)",
                  "SRR_and_doubt_source": "recomputed by analyze() (V.DOUBT etc.), identical to legacy strict metric"}}

    # ---- anchors ----
    print("A2 legacy anchor...")
    result["anchor_A2_legacy_strict"] = legacy_anchor()

    print("loading EXPD...")
    expd_rows, expd_man, a1 = load_expd()
    result["anchor_A1_expd_closure_recompute"] = a1

    print("H2' re-adjudication...")
    result["H2prime"] = h2prime(expd_rows)

    print("EXPD strict cells...")
    result["EXPD_cells"] = summarize_cells(expd_rows, "condition", "family_id",
                                           extra_keys=("stated_complement",))

    print("loading EXPE...")
    expe_rows, expe_man = load_expe()
    result["EXPE_cells"] = summarize_cells(expe_rows, "_cell", "problem_id",
                                           extra_keys=("_stated_complement",))

    with open(os.path.join(OUT_DIR, "s1_strict_metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("WROTE", os.path.join(OUT_DIR, "s1_strict_metrics.json"))

    # short console summary
    print("\n=== ANCHOR A1 (EXPD closure recompute agreement):", a1)
    print("=== ANCHOR A2 (legacy strict):")
    for k, v in result["anchor_A2_legacy_strict"].items():
        print(f"    {k}: hop_sound={v['hop_sound_valid']} closure={v['closure_valid']} n={v['n']}")
    print("=== H2' closure (anchor) vs hop-sound:")
    for k in ("closure_valid_ANCHOR", "hop_sound_valid"):
        h = result["H2prime"][k]
        print(f"    {k}: diff={h['mean_diff']} aff={h['rate_aff_false']} neg={h['rate_neg_false']} "
              f"p_raw={h['p_raw']} pass0.05={h['tost_pass_at_0.05']} passHolm={h['tost_pass_at_holm_alpha_over_6']} "
              f"fams={h['paired_families']} ci95={h['ci95']}")


if __name__ == "__main__":
    main()
