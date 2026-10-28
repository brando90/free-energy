"""E10 bridge analyzer: per-model per-cell 3-way outcomes (absorbed /
silently_corrected / flagged), Wilson CIs, and the readable-vs-recompute margin.
Also pulls the regime2 TRUE-PREFILL numbers for the two old Anthropic models so
the prefill->bridge regime delta is visible (bridge-mode confound bound)."""
import json
import os
import sys

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "exph"))
import exph_common as C  # noqa: E402

BR = os.path.join(EXP, "improvement_plan", "iclr_exec", "recompute_wall", "e10_api_bridge")
R2 = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS", "regime2")

MODEL_MODE = [
    ("claude-opus-4-8", "bridge", "NEW Anthropic (prefill BLOCKED by API)"),
    ("claude-sonnet-5", "bridge", "NEW Anthropic (prefill BLOCKED by API)"),
    ("claude-haiku-4-5", "bridge", "calibration (has prefill in regime2)"),
    ("claude-sonnet-4-5", "bridge", "calibration (has prefill in regime2)"),
    ("gpt-4.1", "bridge", "OpenAI chat"),
    ("gpt-4o", "bridge", "OpenAI chat"),
    ("gpt-5.1", "bridge", "OpenAI chat (reasoning=none)"),
    ("gpt-3.5-turbo-instruct", "completion", "OpenAI legacy completions (true single stream)"),
]
RECOMPUTE = ["opfree_kr1", "opfree_kr8", "onehop_kc1", "deep_kc5"]
READABLE = "adjacent_contradiction"


def cf_rows(rows, cell):
    return [r for r in rows if r["condition"] == cell and r.get("cf")
            and not r.get("generation_failed")]


def three_way(rows):
    n = len(rows)
    ab = si = fl = ot = 0
    for r in rows:
        if r.get("doubt_lex"):
            fl += 1
        elif r.get("final_output_absorbed"):
            ab += 1
        elif r.get("final_output_valid"):
            si += 1
        else:
            ot += 1
    def w(k):
        return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
                "wilson95": C.wilson(k, n) if n else None}
    return {"absorbed": w(ab), "silently_corrected": w(si),
            "flagged": w(fl), "other": w(ot)}


def absorbed_rate(rows):
    n = len(rows)
    k = sum(bool(r.get("final_output_absorbed")) for r in rows)
    return k, n, (round(k / n, 4) if n else None)


def load(model):
    p = os.path.join(BR, "validated_%s.jsonl" % model)
    return C.read_jsonl(p) if os.path.exists(p) else []


def prefill_ref(model):
    """regime2 TRUE-PREFILL absorbed rates (final_output_absorbed) if present."""
    p = os.path.join(R2, "validated_%s.jsonl" % model)
    if not os.path.exists(p):
        return None
    rows = C.read_jsonl(p)
    out = {}
    for cell in RECOMPUTE + [READABLE]:
        sub = cf_rows(rows, cell)
        _, _, r = absorbed_rate(sub)
        out[cell] = r
    return out


def main():
    report = {"models": {}, "note": "3-way on cf cells; absorbed=final_output_absorbed "
              "& not flagged; silently_corrected=final_output_valid & not flagged & "
              "not absorbed; flagged=doubt_lex (frozen code lexicon). Margin = pooled "
              "recompute absorbed - readable(adjacent_contradiction) absorbed."}
    lines = []
    hdr = ("%-24s %-11s | read_adj | opf_kr1 opf_kr8 1hop_kc1 deep_kc5 | pooled_rec | margin | prefill(1hop/deep)"
           % ("model", "mode"))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for model, mode, tag in MODEL_MODE:
        rows = load(model)
        if not rows:
            continue
        m = {"mode": mode, "tag": tag, "cells": {}}
        # readable
        rd = cf_rows(rows, READABLE)
        rk, rn, rr = absorbed_rate(rd)
        m["cells"][READABLE] = {"absorbed": three_way(rd)["absorbed"],
                                "three_way": three_way(rd)}
        # recompute cells
        pooled = []
        percell = {}
        for cell in RECOMPUTE:
            sub = cf_rows(rows, cell)
            k, nn, rt = absorbed_rate(sub)
            percell[cell] = rt
            m["cells"][cell] = {"absorbed": three_way(sub)["absorbed"],
                                "three_way": three_way(sub)}
            pooled += sub
        pk, pn, pr = absorbed_rate(pooled)
        margin = round((pr or 0) - (rr or 0), 4)
        pf = prefill_ref(model)
        m["readable_absorbed"] = rr
        m["pooled_recompute_absorbed"] = pr
        m["margin_recompute_minus_readable"] = margin
        m["prefill_reference"] = pf
        report["models"][model] = m
        pf_s = ("%.2f/%.2f" % (pf["onehop_kc1"], pf["deep_kc5"])) if pf else "  -  "
        lines.append("%-24s %-11s |   %.2f   |  %.2f    %.2f     %.2f     %.2f  |    %.2f    |  %+.2f  | %s"
                     % (model, mode, rr or 0, percell["opfree_kr1"] or 0,
                        percell["opfree_kr8"] or 0, percell["onehop_kc1"] or 0,
                        percell["deep_kc5"] or 0, pr or 0, margin, pf_s))
    table = "\n".join(lines)
    report["ascii_table"] = table
    C.write_json(os.path.join(BR, "bridge_analysis.json"), report)
    print(table)
    print("\nWritten: bridge_analysis.json")


if __name__ == "__main__":
    main()
