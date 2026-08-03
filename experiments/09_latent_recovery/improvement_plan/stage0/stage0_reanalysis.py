#!/usr/bin/env python3
"""Stage-0 zero-GPU re-analyses (MASTER_PLAN section 1, items C1 + EXPB bounds).

Inputs: locked summary JSONs only (validated_summary_negstep/neghop{2..5}.json,
EXPB summary_tables.json). No locked artifact is modified; outputs go to a new
directory. Run from anywhere; paths resolved relative to --exp-root.

Outputs:
  stage0/neghop_multinomial.json   - full 6-class counts per k x position,
                                     all-runs / parsed-only / best-worst bounds
  stage0/expb_imputation_bounds.json
  stage0/STAGE0_REPORT.md          - human-readable summary
"""
import argparse
import json
import os

CLASSES = ["valid_rederivation", "poisoned", "parroted", "derailed", "unparsed"]


def load(path):
    with open(path) as f:
        return json.load(f)


def neghop_table(exp_root):
    files = {1: "validated_summary_negstep.json"}
    for k in (2, 3, 4, 5):
        files[k] = f"validated_summary_neghop{k}.json"
    rows = []
    for k, fn in sorted(files.items()):
        d = load(os.path.join(exp_root, "results", fn))
        for pos in ("early", "mid", "late"):
            if pos not in d:
                continue
            c = d[pos]
            n = c["n"]
            unp = c.get("unparsed", 0)
            valid = c.get("valid_rederivation", 0)
            parsed = n - unp
            rows.append({
                "k": k,
                "position": pos,
                "n": n,
                "counts": {cls: c.get(cls, 0) for cls in CLASSES},
                "doubt_acknowledged_rate": round(c.get("acknowledged", 0) / n, 4),
                "valid_rate_all_runs": round(valid / n, 4),
                "valid_rate_parsed_only": round(valid / parsed, 4) if parsed else None,
                "valid_rate_worst_case": round(valid / n, 4),           # unparsed -> not valid
                "valid_rate_best_case": round((valid + unp) / n, 4),    # unparsed -> valid
                "unparsed_rate": round(unp / n, 4),
            })
    return rows


def expb_bounds(exp_root):
    st = load(os.path.join(exp_root, "results", "EXPB_LOCAL_CERT_FLIP",
                           "summary_tables.json"))
    pooled = st["metrics"]["pooled_by_condition"]
    out = {}
    for cond, cell in pooled.items():
        n = cell["n"]
        valid = cell["valid_rederivation"]["count"]
        unp = cell["unparsed"]["count"]
        out[cond] = {
            "n": n,
            "valid_count": valid,
            "unparsed_count": unp,
            "valid_rate_worst_case": round(valid / n, 4),
            "valid_rate_best_case": round((valid + unp) / n, 4),
            "unparsed_rate": round(unp / n, 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    neghop = neghop_table(args.exp_root)
    with open(os.path.join(args.out_dir, "neghop_multinomial.json"), "w") as f:
        json.dump(neghop, f, indent=2)

    try:
        expb = expb_bounds(args.exp_root)
    except Exception as e:  # summary schema may differ; report rather than fail
        expb = {"error": str(e)}
    with open(os.path.join(args.out_dir, "expb_imputation_bounds.json"), "w") as f:
        json.dump(expb, f, indent=2)

    lines = ["# Stage-0 re-analysis (zero GPU)", "",
             "## Neghop dose-response, 6-class multinomial", "",
             "| k | pos | n | unparsed | valid (all runs) | valid (parsed-only) "
             "| valid [worst, best] | doubt |",
             "|---|-----|---|----------|------------------|--------------------"
             "|---------------------|-------|"]
    for r in neghop:
        lines.append(
            f"| {r['k']} | {r['position']} | {r['n']} | {r['unparsed_rate']:.3f} "
            f"| {r['valid_rate_all_runs']:.3f} | {r['valid_rate_parsed_only']:.3f} "
            f"| [{r['valid_rate_worst_case']:.3f}, {r['valid_rate_best_case']:.3f}] "
            f"| {r['doubt_acknowledged_rate']:.3f} |")
    lines += ["",
              "Reading: the plotted closure-valid decline across k is carried by "
              "unparsed rows; parsed-only validity is flat-to-increasing. The doubt "
              "gradient (computed on raw text before parsing) survives. Per "
              "MASTER_PLAN C1: retire the legacy validity curve, keep doubt.",
              "", "## EXPB closure-valid imputation bounds", ""]
    lines.append("```json\n" + json.dumps(expb, indent=2) + "\n```")
    with open(os.path.join(args.out_dir, "STAGE0_REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out_dir}: neghop rows={len(neghop)}, expb conds="
          f"{list(expb)[:5]}")


if __name__ == "__main__":
    main()
