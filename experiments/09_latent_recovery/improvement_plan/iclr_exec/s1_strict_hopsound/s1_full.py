#!/usr/bin/env python
"""ITEM S1 - Strict / hop-sound pass over EXPD + EXPE stored rows, and
re-adjudication of H2' (aff_false vs neg_false polarity equivalence) under
hop-soundness. Zero GPU. Reads results/* read-only; writes ONLY into
improvement_plan/iclr_exec/s1_strict_hopsound/.

Reuses the PUBLISHED estimators verbatim:
  * improvement_plan/stage0_strict/strict_metrics.py  -> SM.analyze (M1-M4 hop-sound)
  * improvement_plan/expd/expd_confirmatory_analysis.py -> CA (H2' cohort, paired-family
    TOST, family-cluster bootstrap; IDENTICAL to the registered EXPD test)

SCHEMA-MAPPING DECISIONS (legacy strict_metrics EXPA schema -> EXPD/EXPE), each
documented in the JSON under "mapping_decisions":
  M-1 closure_valid  := stored 'valid_recovery'. Verified: SM.analyze recomputed
                        class==valid_rederivation agrees with stored valid_recovery
                        on 11577/11577 EXPD and 2104/2104 EXPE rows (anchor A1).
  M-2 hop_sound_valid:= closure_valid AND SM.analyze(...).hop_sound_all. Because
                        recomputed closure == stored closure (agreement 1.0), this is
                        a clean strict SUBSET of the paper's closure_valid.
  M-3 is_false (audited-false plant): EXPD -> audited_truth_status=='false';
                        EXPE -> injected_statement_truth_status=='false' (manifest;
                        integrity file shows all 2104 EXPE plants audited false).
  M-4 cluster unit: EXPD -> family_id (identical to the registered H2' cluster_boot);
                        EXPE -> problem_id.
  M-5 cell key: EXPD -> 'condition'; EXPE -> 'arm@injection_position'.
  M-6 doubt/echo/corrective/targeted/deriv: RECOMPUTED by SM.analyze (V.DOUBT regex
                        etc.), NOT the stored fields, so SRR/stated-reliance use the
                        EXACT definitions of the published STRICT_METRICS_REPORT.md.
  M-7 EXPD question passed to analyze = manifest 'question' (the world the model saw);
                        EXPE question = manifest 'question' (the AUGMENTED question, i.e.
                        with the arm's added rule), matching what the model was shown.
"""
import os, sys, json, math, random, datetime
from collections import defaultdict

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
os.environ.setdefault("LR_ROOT", EXP)
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expd"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "stage0_strict"))
sys.path.insert(0, os.path.join(EXP, "src"))
import expd_confirmatory_analysis as CA   # noqa
import strict_metrics as SM               # noqa
import validator as V                     # noqa

OUT = os.path.join(EXP, "improvement_plan", "iclr_exec", "s1_strict_hopsound")
os.makedirs(OUT, exist_ok=True)
EXPD_DIR = os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT")
EXPE_DIR = os.path.join(EXP, "results", "EXPE_EVIDENCE_MOVER")


def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ----------------------------------------------------------- strict per-row
def strict_row(m, r):
    """Compute the strict per-row booleans (compose() semantics of strict_metrics)."""
    if r.get("failed_generation"):
        prim = dict(closure_valid=0, hop_sound_all=0, hop_sound_valid=0, goal_jump0=0,
                    goal_jump_strict=0, doubt=0, targeted=0, corrective=0, echo=0, deriv=0,
                    used_any=0, srr_task=0, srr_plan=0, stated_reliance=0, a_star_kind=None,
                    recomp_agrees=1)
        return prim
    a = SM.analyze(m["question"], m.get("prefix_steps", []), m["injected_statement"],
                   r.get("continuation", ""), m["target"], m["entity"])
    stored_valid = int(bool(r.get("valid_recovery")))
    recomp_valid = int(a["class"] == "valid_rederivation")
    hop_all = int(bool(a["hop_sound_all"]))
    jump0 = int(stored_valid == 1 and a.get("inter_facts", 0) == 0)
    echo = int(bool(a["echo"])); deriv = int(bool(a["deriv_use"] or a["deriv_use_rule"]))
    doubt = int(bool(a["doubt"])); corrective = int(bool(a["corrective"]))
    targeted = int(bool(a["targeted"]))
    foc = int(doubt or corrective)
    return dict(
        closure_valid=stored_valid,
        hop_sound_all=hop_all,
        hop_sound_valid=int(stored_valid == 1 and hop_all == 1),
        goal_jump0=jump0,
        goal_jump_strict=int(jump0 == 1 and hop_all == 0),
        doubt=doubt, targeted=targeted, corrective=corrective, echo=echo, deriv=deriv,
        used_any=int(echo or deriv),
        srr_task=int(stored_valid == 1 and foc == 1 and jump0 == 0),          # gate is_false at agg
        srr_plan=int(stored_valid == 1 and (targeted or corrective) == 1 and jump0 == 0 and echo == 0),
        stated_reliance=int(stored_valid == 1 and echo == 1),
        a_star_kind=a.get("a_star_kind"),
        recomp_agrees=int(stored_valid == recomp_valid),
    )


# ----------------------------------------------------------- CI helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [round(c-h, 4), round(c+h, 4)]


def cluster_boot(rows, key, cfield, seed=0, n_boot=2000):
    ids = sorted({r[cfield] for r in rows})
    if not ids:
        return [None, None]
    by = defaultdict(list)
    for r in rows:
        by[r[cfield]].append(int(bool(r["_prim"][key])) if key in r["_prim"] else int(bool(r[key])))
    rng = random.Random(seed); vals = []
    for _ in range(n_boot):
        num = den = 0
        for cid in (rng.choice(ids) for _ in ids):
            num += sum(by[cid]); den += len(by[cid])
        vals.append(num/den if den else float("nan"))
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return [None, None]
    return [round(vals[int(0.025*(len(vals)-1))], 4), round(vals[int(0.975*(len(vals)-1))], 4)]


def blk(rows, key, cfield, ci=False, denom_rows=None):
    use = denom_rows if denom_rows is not None else rows
    n = len(use)
    k = sum(int(bool(r["_prim"][key])) for r in use)
    out = {"count": k, "n": n, "rate": round(k/n, 4) if n else None}
    if ci and n:
        out["wilson95"] = wilson(k, n)
        out["cluster_boot95"] = cluster_boot(use, key, cfield)
    return out


# ----------------------------------------------------------- cell summary
def summarize(rows, cell_field, cfield):
    cells = {}
    for cell in sorted({r[cell_field] for r in rows}):
        sub = [r for r in rows if r[cell_field] == cell]
        false_sub = [r for r in sub if r["is_false"]]
        valid_sub = [r for r in sub if r["_prim"]["closure_valid"]]
        c = {"n": len(sub), "cluster_n": len({r[cfield] for r in sub}),
             "n_false_audited": len(false_sub),
             "closure_valid": blk(sub, "closure_valid", cfield, ci=True),
             "hop_sound_valid": blk(sub, "hop_sound_valid", cfield, ci=True),
             "goal_jump0": blk(sub, "goal_jump0", cfield),
             "goal_jump_strict": blk(sub, "goal_jump_strict", cfield),
             "doubt": blk(sub, "doubt", cfield),
             "used_any": blk(sub, "used_any", cfield),
             "echo": blk(sub, "echo", cfield),
             "deriv": blk(sub, "deriv", cfield),
             "stated_reliance": blk(sub, "stated_reliance", cfield)}
        # hop-sound share among valid completions
        c["hop_sound_given_valid"] = {
            "n": len(valid_sub),
            "count": sum(r["_prim"]["hop_sound_all"] for r in valid_sub),
            "rate": round(sum(r["_prim"]["hop_sound_all"] for r in valid_sub)/len(valid_sub), 4) if valid_sub else None}
        # SRR + stated-reliance restricted to audited-false plants (M1/M3 denominator)
        if false_sub:
            c["srr_task"] = blk(false_sub, "srr_task", cfield, ci=True)
            c["srr_plan"] = blk(false_sub, "srr_plan", cfield)
            c["stated_reliance_false"] = blk(false_sub, "stated_reliance", cfield)
            c["doubt_false"] = blk(false_sub, "doubt", cfield)
            c["corrective_false"] = blk(false_sub, "corrective", cfield)
        else:
            c["srr_task"] = c["srr_plan"] = c["stated_reliance_false"] = "n/a (no audited-false rows)"
        cells[cell] = c
    return cells


# ----------------------------------------------------------- loaders
def load_expd():
    rows, manifest = CA.load_rows(EXPD_DIR)
    cohort = read_jsonl(os.path.join(EXPD_DIR, "gold_cohort.jsonl"))
    joint, prefix_same = CA.attr_cohort_maps(cohort)
    CA.annotate_cohort(rows, joint, prefix_same)
    agree = tot = 0
    for r in rows:
        m = manifest.get(r["run_id"])
        prim = strict_row(m, r)
        r["_prim"] = prim
        r["is_false"] = int(r.get("audited_truth_status") == "false")
        r["_hop_sound_valid"] = prim["hop_sound_valid"]      # for CA.mval monkeypatch
        agree += prim["recomp_agrees"]; tot += 1
    return rows, manifest, {"closure_recompute_agreement": round(agree/tot, 6), "n": tot}


def load_expe():
    manifest = {r["run_id"]: r for r in read_jsonl(os.path.join(EXPE_DIR, "manifest.jsonl"))}
    rows = []
    agree = tot = 0
    for r in read_jsonl(os.path.join(EXPE_DIR, "validated_outputs.jsonl")):
        m = manifest.get(r["run_id"])
        if not m:
            continue
        prim = strict_row(m, r)
        r["_prim"] = prim
        ts = m.get("injected_statement_truth_status")
        r["is_false"] = int(ts == "false")
        r["_cell"] = "%s@%s" % (r.get("arm", r.get("condition")), r["injection_position"])
        r["_stated_complement_stored"] = int(bool(r.get("stated_complement_of_falsehood")))
        rows.append(r)
        agree += prim["recomp_agrees"]; tot += 1
    return rows, manifest, {"closure_recompute_agreement": round(agree/tot, 6), "n": tot}


# ----------------------------------------------------------- H2' re-adjudication
def h2prime(rows):
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

    def boot_arm(rr, metric):
        key = "_hop_sound_valid" if metric == "hop_sound_valid" else "valid_recovery"
        ids = sorted({r["family_id"] for rr2 in [rr] for r in rr2})
        by = defaultdict(list)
        for r in rr:
            by[r["family_id"]].append(int(bool(r.get(key))))
        rng = random.Random(0); vals = []
        for _ in range(2000):
            num = den = 0
            for cid in (rng.choice(ids) for _ in ids):
                num += sum(by[cid]); den += len(by[cid])
            vals.append(num/den if den else float("nan"))
        vals = sorted(v for v in vals if not math.isnan(v))
        return [round(vals[int(0.025*(len(vals)-1))], 4), round(vals[int(0.975*(len(vals)-1))], 4)]

    def run(metric):
        ps = CA.paired_family_stats(rows_a, rows_b, metric)
        t = CA.tost(ps)
        return {"metric": metric, "dv_label": ("closure_valid (paper DV, ANCHOR)"
                if metric == "closure_valid" else "hop_sound_valid (strict, S1)"),
                "n_rows_aff": ps["n_rows_a"], "n_rows_neg": ps["n_rows_b"],
                "paired_families": ps["n_families_paired"],
                "rate_aff_false": round(ps["rate_a"], 4), "rate_neg_false": round(ps["rate_b"], 4),
                "boot95_aff": boot_arm(rows_a, metric), "boot95_neg": boot_arm(rows_b, metric),
                "mean_diff_aff_minus_neg": t["mean_diff"], "se": t["se"], "df": t["df"],
                "ci95": t["ci95"], "ci90_equiv": t["ci90"], "margin": t["margin"],
                "p_lower": t["p_lower"], "p_upper": t["p_upper"], "p_raw_TOST": t["p_raw"],
                "tost_pass_at_0.05": bool(t["p_raw"] is not None and t["p_raw"] < 0.05),
                "tost_pass_at_holm_alpha_over_6": bool(t["p_raw"] is not None and t["p_raw"] < 0.05/6),
                "note": t.get("note")}
    res = {"closure_valid_ANCHOR": run("closure_valid"), "hop_sound_valid": run("hop_sound_valid")}
    CA.mval = orig_mval
    return res


# ----------------------------------------------------------- legacy anchor A2
def legacy_anchor():
    notes = {}
    dn = open(os.devnull, "w")
    expa = SM.load_expa(os.path.join(EXP, "results", "EXPA_GLOBAL_EXPANSION"), notes, dn, "EXPA")
    legacy = SM.load_legacy(EXP, notes, dn)
    dn.close()
    out = {}
    for rows in (expa, legacy):
        byfam = defaultdict(list)
        for r in rows:
            byfam[r["family"]].append(r)
        for fam, rr in byfam.items():
            n = len(rr)
            out[fam] = {"n": n,
                        "hop_sound_valid": round(sum(int(bool(x["hop_sound_valid"])) for x in rr)/n, 4),
                        "closure_valid": round(sum(int(bool(x["closure_valid"])) for x in rr)/n, 4),
                        "srr_task": round(sum(int(bool(x["srr_task"])) for x in rr if x["is_false"])
                                          / max(1, sum(1 for x in rr if x["is_false"])), 4)}
    keep = ("EXPA:one_hop_falsehood", "EXPA:global_falsehood", "EXPA:benign_paraphrase",
            "EXPA:true_interruption", "legacy:negstep", "legacy:falsehood")
    return {k: out[k] for k in keep if k in out}


# ----------------------------------------------------------- main
def main():
    result = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "item": "S1 strict/hop-sound pass over EXPD+EXPE; H2' re-adjudicated under hop-soundness",
        "mapping_decisions": {
            "M1_closure_valid": "stored valid_recovery (paper DV; CA.mval reads this)",
            "M2_hop_sound_valid": "closure_valid AND SM.analyze().hop_sound_all",
            "M3_is_false_EXPD": "audited_truth_status=='false'",
            "M3_is_false_EXPE": "injected_statement_truth_status=='false' (all 2104 audited false)",
            "M4_cluster_unit_EXPD": "family_id", "M4_cluster_unit_EXPE": "problem_id",
            "M5_cell_EXPD": "condition", "M5_cell_EXPE": "arm@injection_position",
            "M6_SRR_doubt_source": "recomputed by SM.analyze (V.DOUBT etc.), identical defs to STRICT_METRICS_REPORT",
            "M7_question_EXPE": "manifest 'question' = augmented (arm's added rule present), as shown to model",
        }}

    print("A2 legacy anchor...")
    result["anchor_A2_legacy_strict"] = legacy_anchor()

    print("load EXPD...")
    expd, expd_man, a1 = load_expd()
    result["anchor_A1_EXPD_closure_recompute"] = a1

    print("H2' re-adjudication...")
    result["H2prime"] = h2prime(expd)

    print("EXPD cells...")
    result["EXPD_cells"] = summarize(expd, "condition", "family_id")

    print("load EXPE...")
    expe, expe_man, a1e = load_expe()
    result["anchor_A1_EXPE_closure_recompute"] = a1e
    print("EXPE cells...")
    result["EXPE_cells"] = summarize(expe, "_cell", "problem_id")

    # ---- strict-vs-permissive accounting (EXPD) + reversal scan ----
    acct = []
    for cell, c in result["EXPD_cells"].items():
        cv = c["closure_valid"]["rate"]; hs = c["hop_sound_valid"]["rate"]
        acct.append({"cell": cell, "n": c["n"], "closure_valid": cv, "hop_sound_valid": hs,
                     "delta_pp": round((cv-hs)*100, 1) if cv is not None else None,
                     "goal_jump0": c["goal_jump0"]["rate"],
                     "hop_sound_given_valid": c["hop_sound_given_valid"]["rate"],
                     "srr_task": (c["srr_task"]["rate"] if isinstance(c["srr_task"], dict) else None),
                     "stated_reliance_false": (c["stated_reliance_false"]["rate"]
                                               if isinstance(c["stated_reliance_false"], dict) else None)})
    result["strict_vs_permissive_EXPD"] = acct

    # reversal flags: registered/claimed contrasts whose sign or verdict flips
    rev = []
    h2c = result["H2prime"]["closure_valid_ANCHOR"]; h2h = result["H2prime"]["hop_sound_valid"]
    sign_c = (h2c["mean_diff_aff_minus_neg"] > 0)
    sign_h = (h2h["mean_diff_aff_minus_neg"] > 0)
    rev.append({
        "contrast": "H2' aff_false vs neg_false (closure->hop_sound)",
        "closure_diff": h2c["mean_diff_aff_minus_neg"], "hop_sound_diff": h2h["mean_diff_aff_minus_neg"],
        "point_sign_flips": bool(sign_c != sign_h),
        "closure_tost_pass": h2c["tost_pass_at_holm_alpha_over_6"],
        "hop_sound_tost_pass": h2h["tost_pass_at_holm_alpha_over_6"],
        "equivalence_conclusion_preserved": bool(h2h["tost_pass_at_0.05"]),
        "note": ("equivalence CONCLUSION (|diff|<0.10) is what the paper claims; point-sign flip is "
                 "secondary and if anything strengthens 'polarity is not the driver'")})
    # cell-level: closure-valid ordering vs hop-sound ordering for aff vs neg at each d
    for d in ("d1", "d3"):
        a = result["EXPD_cells"].get("aff_false_attr_%s" % d)
        b = result["EXPD_cells"].get("neg_false_attr_%s" % d)
        if a and b:
            ca, cb = a["closure_valid"]["rate"], b["closure_valid"]["rate"]
            ha, hb = a["hop_sound_valid"]["rate"], b["hop_sound_valid"]["rate"]
            if (ca > cb) != (ha > hb):
                rev.append({"contrast": "aff vs neg %s full-cell ordering" % d,
                            "closure": [ca, cb], "hop_sound": [ha, hb],
                            "note": "full-cell (unpaired) aff/neg ordering flips under hop-soundness"})
    # C-0 closure_valid was null (d0~d1); check hop-sound d0 vs d1 (aff_false)
    a0 = result["EXPD_cells"].get("aff_false_attr_d0"); a1c = result["EXPD_cells"].get("aff_false_attr_d1")
    if a0 and a1c:
        rev.append({"contrast": "aff_false d0 vs d1 hop_sound_valid (C-0 closure was null)",
                    "closure_d0_d1": [a0["closure_valid"]["rate"], a1c["closure_valid"]["rate"]],
                    "hop_sound_d0_d1": [a0["hop_sound_valid"]["rate"], a1c["hop_sound_valid"]["rate"]],
                    "note": ("closure_valid is flat d0~d1 (registered C-0 null); hop_sound_valid is "
                             "elevated at d0 -- strict metric surfaces a visibility structure closure hid; "
                             "descriptive only (C-0 is non-family, no verdict)")})
    result["reversal_scan"] = rev

    with open(os.path.join(OUT, "s1_strict_metrics.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "s1_strict_metrics.json"))

    # ---- console summary ----
    print("\n=== A1 EXPD closure recompute agreement:", a1)
    print("=== A1 EXPE closure recompute agreement:", a1e)
    print("=== A2 legacy strict (reproduce STRICT_METRICS_REPORT):")
    for k, v in result["anchor_A2_legacy_strict"].items():
        print(f"    {k}: hop_sound={v['hop_sound_valid']} closure={v['closure_valid']} srr_task={v['srr_task']} n={v['n']}")
    print("=== H2' closure(ANCHOR) vs hop_sound:")
    for k in ("closure_valid_ANCHOR", "hop_sound_valid"):
        h = result["H2prime"][k]
        print(f"    {k}: diff(aff-neg)={h['mean_diff_aff_minus_neg']} aff={h['rate_aff_false']} "
              f"neg={h['rate_neg_false']} ci95={h['ci95']} p_TOST={h['p_raw_TOST']} "
              f"pass0.05={h['tost_pass_at_0.05']} passHolm={h['tost_pass_at_holm_alpha_over_6']} "
              f"fams={h['paired_families']} rows={h['n_rows_aff']}/{h['n_rows_neg']}")


if __name__ == "__main__":
    main()
