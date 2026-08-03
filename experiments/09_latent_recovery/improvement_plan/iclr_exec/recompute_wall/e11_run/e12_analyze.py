"""E12 cross-arm analysis + report. Reads e12_fixes/{A0..A3}/summary_tables.json
(and validated rows for pooled contrasts), emits the ladder plot data, the
instruction-breach contrasts (absorbed neutral - absorbed verify, Wilson-based),
the readable-control deltas, compliance cost (tokens), and the recompute
signature rates. Writes E12_FIXES_REPORT.md.

Breach rule (spec S6): instruction breaches wall for cell c iff absorbed drops
>=0.15 from A0 to the verify arm AND the two Wilson95 intervals are disjoint
(one-sided-clear proxy). Reported per cell per arm; honest either direction."""
import argparse
import json
import math
import os

ARMS = ["A0", "A1", "A2", "A3"]
ARM_LABEL = {"A0": "baseline", "A1": "generic", "A2": "targeted", "A3": "few-shot"}
CELLS = [("k0_bare", "adjacent_contradiction (readable control)"),
         ("k1_bare", "onehop_kc1 (shallow computed)"),
         ("k5_bare", "deep_kc5 (deep computed)")]


def load(base, arm):
    p = os.path.join(base, arm, "summary_tables.json")
    return json.load(open(p)) if os.path.exists(p) else None


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def disjoint(ci_a, ci_b):
    if None in ci_a or None in ci_b:
        return None
    return ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", required=True)
    a = ap.parse_args()
    S = {arm: load(a.out_base, arm) for arm in ARMS}
    present = [arm for arm in ARMS if S[arm]]

    ladder = {}          # cell -> arm -> {rate,count,n,ci,tokens,shows_recompute}
    for cell_key, _label in CELLS:
        ladder[cell_key] = {}
        for arm in present:
            c = (S[arm].get("cells") or {}).get(cell_key)
            if not c:
                continue
            ab = c["absorbed"]
            ladder[cell_key][arm] = {
                "absorbed_rate": ab["rate"], "absorbed_count": ab["count"],
                "n": c["n"], "program_n": c["program_n"], "wilson95": ab["wilson95"],
                "silently_corrected": c["silently_corrected"]["rate"],
                "flagged": c["flagged"]["rate"],
                "output_tokens_mean": c.get("output_tokens_mean"),
                "shows_recompute_rate": c.get("shows_recompute_rate")}

    # ---- breach contrasts vs A0
    contrasts = {}
    for cell_key, _label in CELLS:
        base = ladder[cell_key].get("A0")
        contrasts[cell_key] = {}
        if not base:
            continue
        for arm in [x for x in present if x != "A0"]:
            row = ladder[cell_key].get(arm)
            if not row:
                continue
            drop = round(base["absorbed_rate"] - row["absorbed_rate"], 4)
            dj = disjoint(base["wilson95"], row["wilson95"])
            breach = (drop >= 0.15 and dj is True)
            contrasts[cell_key][arm] = {
                "absorbed_A0": base["absorbed_rate"], "absorbed_arm": row["absorbed_rate"],
                "drop": drop, "wilson_disjoint": dj,
                "verdict": "BREACH" if breach else "wall_holds",
                "displaced_to": {"silently_corrected": row["silently_corrected"],
                                 "flagged": row["flagged"]}}

    power = {}
    for cell_key, _l in CELLS:
        for arm in present:
            r = ladder[cell_key].get(arm)
            if r:
                n = r["program_n"]
                power["%s/%s" % (cell_key, arm)] = (
                    "ok" if n >= 150 else "scored_flagged" if n >= 100 else
                    "ci_widened" if n >= 50 else "INVALID_underpowered")

    analysis = {"experiment": "E12_FIXES_LADDER", "model": "claude-haiku-4-5",
                "arms_present": present, "ladder": ladder, "breach_contrasts": contrasts,
                "power_flags": power,
                "gold_cohort_by_arm": {arm: S[arm].get("gold_cohort") for arm in present},
                "usd_by_arm": {arm: S[arm].get("this_run_usd") for arm in present},
                "breach_rule": "absorbed drop >=0.15 from A0 AND Wilson95 intervals disjoint"}
    with open(os.path.join(a.out_base, "e12_analysis.json"), "w") as fh:
        json.dump(analysis, fh, indent=2, sort_keys=True)

    # ---- report
    L = []
    def w(s=""):
        L.append(s)
    w("# E12 FIXES-LADDER report -- claude-haiku-4-5 (TRUE prefill)")
    w("")
    w("Does an explicit verify instruction breach the recompute wall (E10/E11: "
      "computed-value plants absorbed ~1.0)? Arms A0 baseline / A1 generic / "
      "A2 targeted / A3 few-shot. Instruction prepended to the user block; prefill "
      "unchanged; golds regenerated per arm. Arms present: %s." % ", ".join(present))
    w("")
    w("## Ladder: absorbed rate by arm (Wilson95), per cell")
    w("")
    for cell_key, label in CELLS:
        w("### %s" % label)
        w("")
        w("| arm | absorbed | Wilson95 | silently_corr | flagged | out_tok/cont | shows_recompute | n(prog) | power |")
        w("|---|---|---|---|---|---|---|---|---|")
        for arm in present:
            r = ladder[cell_key].get(arm)
            if not r:
                w("| %s (%s) | -- absent -- |||||||" % (arm, ARM_LABEL[arm]))
                continue
            lo, hi = r["wilson95"]
            ci = "[%.2f,%.2f]" % (lo, hi) if lo is not None else "n.m."
            w("| %s (%s) | %s | %s | %s | %s | %s | %s | %d(%d) | %s |" % (
                arm, ARM_LABEL[arm],
                _f(r["absorbed_rate"]), ci, _f(r["silently_corrected"]),
                _f(r["flagged"]), _f(r["output_tokens_mean"], 0),
                _f(r["shows_recompute_rate"]), r["n"], r["program_n"],
                power.get("%s/%s" % (cell_key, arm), "?")))
        w("")
    w("## Instruction-breach contrasts (vs A0 baseline)")
    w("")
    w("Breach = absorbed drop >= 0.15 AND Wilson95 intervals disjoint.")
    w("")
    w("| cell | arm | absorbed A0 -> arm | drop | CIs disjoint | verdict |")
    w("|---|---|---|---|---|---|")
    for cell_key, label in CELLS:
        for arm, cc in contrasts.get(cell_key, {}).items():
            w("| %s | %s | %s -> %s | %s | %s | **%s** |" % (
                cell_key, arm, _f(cc["absorbed_A0"]), _f(cc["absorbed_arm"]),
                _f(cc["drop"]), cc["wilson_disjoint"], cc["verdict"]))
    w("")
    w("## Notes")
    w("")
    w("- Readable control (k0 adjacent_contradiction): a prompt should not need to "
      "fix what is already re-readable; large A0 movement here would indicate the "
      "instruction changes readable behaviour too (interpret computed-cell drops "
      "against this baseline).")
    w("- Compliance cost = out_tok/cont column; recompute signature (shows_recompute) "
      "is the mechanical grep defined in each arm's run_manifest.json.")
    w("- Flag channel: doubt_lex frozen lexicon (no 32B judge on the API arm).")
    w("- Cost: %s (per arm, this run). Cumulative hard stop enforced by api_gen ledger." %
      json.dumps(analysis["usd_by_arm"]))
    w("- Gold cohort (eligible/kept) per arm: `%s`" %
      json.dumps(analysis["gold_cohort_by_arm"]))
    w("")
    rep = os.path.join(a.out_base, "E12_FIXES_REPORT.md")
    with open(rep, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[e12_analyze] wrote %s and e12_analysis.json" % rep)
    print(json.dumps({"ladder_absorbed": {c: {arm: ladder[c].get(arm, {}).get("absorbed_rate")
                      for arm in present} for c, _ in CELLS}}, indent=2))


def _f(x, nd=3):
    if x is None:
        return "n.m."
    return ("%.*f" % (nd, x)) if nd else "%d" % round(x)


if __name__ == "__main__":
    main()
