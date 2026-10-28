#!/usr/bin/env python3
"""E10 Track A — API panel recompute-side absorption report.

Reads the existing Jul-6 EXPG-substrate (regime2) and EXPH-NL-substrate
(instruct_panel, completions_probe) validated data, which together already
cover the entire E10 §4 API roster on the substrates E10 §4 designates, and
computes per-model recompute-side absorption (with n's), readable-side, the
asymmetry margin, and a refusal scan. No new API calls are made.

Outputs into <EXP>/improvement_plan/iclr_exec/recompute_wall/e10_api/.
"""
import json, os, re, math, datetime, collections

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
FF = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS")
OUT = os.path.join(EXP, "improvement_plan", "iclr_exec", "recompute_wall", "e10_api")
os.makedirs(OUT, exist_ok=True)

RECOMPUTE_CELLS = ["opfree_kr1", "opfree_kr8", "onehop_kc1", "deep_kc5"]
READABLE_CELL = "adjacent_contradiction"
CONTROL_CELLS = ["benign_paraphrase", "true_interruption"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def rate_over(rows, field):
    """rate = k / n_measured, measurable = field not None. Returns dict."""
    meas = [int(bool(r[field])) for r in rows if r.get(field) is not None]
    n = len(meas)
    k = sum(meas)
    return {"count": k, "n_measured": n,
            "rate": (round(k / n, 4) if n else None),
            "wilson95": wilson(k, n)}


# ---------- refusal detection ----------
REFUSAL_RE = re.compile(
    r"(?i)\b("
    r"i can'?t (?:continue|comply|do that|help|assist|complete|finish)"
    r"|i cannot (?:continue|comply|help|assist|complete|provide|do that)"
    r"|i'?m sorry,? but i (?:can'?t|cannot|won'?t)"
    r"|i'?m unable to (?:continue|comply|complete|assist|help)"
    r"|i am unable to (?:continue|comply|complete)"
    r"|i won'?t (?:be able to )?continue"
    r"|i will not (?:continue|comply)"
    r"|i (?:do not|don'?t) feel comfortable"
    r"|i'?m not able to (?:continue|comply|complete)"
    r"|i must decline"
    r"|i refuse to"
    r")")


def refusal_scan(rows):
    hits = []
    for r in rows:
        cont = r.get("continuation") or ""
        # only flag when the refusal appears at/near the start of the turn
        head = cont.strip()[:200]
        if REFUSAL_RE.search(head):
            hits.append(head[:160])
    return hits


# ---------- EXPG substrate models (regime2) ----------
expg_models = {
    "claude-haiku-4-5": os.path.join(FF, "regime2", "validated_claude-haiku-4-5.jsonl"),
    "claude-sonnet-4-5": os.path.join(FF, "regime2", "validated_claude-sonnet-4-5.jsonl"),
}

# ---------- NL substrate models (instruct_panel + completions_probe) ----------
nl_models = {
    "gpt-4.1": (os.path.join(FF, "instruct_panel", "validated_gpt-4.1_instruct.jsonl"), "instruct"),
    "gpt-4o": (os.path.join(FF, "instruct_panel", "validated_gpt-4o_instruct.jsonl"), "instruct"),
    "gpt-5.1": (os.path.join(FF, "instruct_panel", "validated_gpt-5.1_instruct.jsonl"), "instruct"),
    "claude-opus-4-8": (os.path.join(FF, "instruct_panel", "validated_claude-opus-4-8_instruct.jsonl"), "instruct"),
    "claude-sonnet-5": (os.path.join(FF, "instruct_panel", "validated_claude-sonnet-5_instruct.jsonl"), "instruct"),
    "gpt-3.5-turbo-instruct": (os.path.join(FF, "completions_probe", "validated_gpt-3.5-turbo-instruct_completion.jsonl"), "completions"),
}

report = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "note": "Computed from existing Jul-6 EXPH2_FRONTIER_FOLLOWUPS validated data; "
                  "no new API calls (E10 §4 roster already run on designated substrates).",
          "expg_substrate": {}, "nl_substrate": {}, "refusal_scan": {}}

# ===== EXPG substrate =====
for mid, path in expg_models.items():
    rows = read_jsonl(path)
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    cells = {}
    for cell, sub in by_cell.items():
        cells[cell] = {
            "n_rows": len(sub),
            "final_output_absorbed": rate_over(sub, "final_output_absorbed"),
            "next_read_absorbed": rate_over(sub, "next_read_absorbed"),
        }
    # pooled recompute-side
    pooled = [r for c in RECOMPUTE_CELLS for r in by_cell.get(c, [])]
    read_rows = by_cell.get(READABLE_CELL, [])
    rc = rate_over(pooled, "final_output_absorbed")
    rd = rate_over(read_rows, "final_output_absorbed")
    margin = (round(rc["rate"] - rd["rate"], 4)
              if rc["rate"] is not None and rd["rate"] is not None else None)
    report["expg_substrate"][mid] = {
        "cells": cells,
        "recompute_side_pooled_final_output_absorbed": rc,
        "readable_side_adjacent_contradiction_final_output_absorbed": rd,
        "asymmetry_margin_recompute_minus_readable": margin,
        "asymmetry_replicates_ge_0.15": (margin is not None and margin >= 0.15),
    }
    report["refusal_scan"][mid + " (EXPG/regime2)"] = {
        "n_rows": len(rows), "n_refusal": len(refusal_scan(rows)),
        "examples": refusal_scan(rows)[:3],
        "generation_failed": sum(1 for r in rows if r.get("generation_failed")),
        "derailed": sum(1 for r in rows if r.get("derailed")),
    }

# ===== NL substrate =====
for mid, (path, mode) in nl_models.items():
    rows = read_jsonl(path)
    by_cond = collections.defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    gf = by_cond.get("global_falsehood", [])          # recompute / derivational
    oh = by_cond.get("one_hop_falsehood", [])          # readable
    rc = rate_over(gf, "inj_derivational")
    rd_corr = rate_over(oh, "stated_complement")       # readable-side CORRECTION rate
    # class distribution on the derivational condition
    cls = collections.Counter(r.get("class") for r in gf)
    report["nl_substrate"][mid] = {
        "mode": mode,
        "recompute_side_inj_derivational_global_falsehood": rc,
        "readable_side_stated_complement_one_hop": rd_corr,
        "global_falsehood_class_dist": dict(cls),
        "caveat": ("instruct chat re-presentation; NL entity-chain substrate (EXPH channels "
                   "stand in per E10 §4); not pooled with EXPG or open-weights"
                   if mode == "instruct" else
                   "legacy completions single-stream; NL entity-chain substrate (EXPH channels stand in)"),
    }
    report["refusal_scan"][mid + " (NL/%s)" % mode] = {
        "n_rows": len(rows), "n_refusal": len(refusal_scan(rows)),
        "examples": refusal_scan(rows)[:3],
        "generation_failed": sum(1 for r in rows if r.get("failed_generation")),
    }

with open(os.path.join(OUT, "e10_api_summary.json"), "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
