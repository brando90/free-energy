"""E11 depth x opacity generator dry-run (CPU only; NO model, NO GPU).

For each depth cell k in {0,1,2,3,5}: rejection-sample up to N accepted worlds
under the FULL fail-closed audit stack, and report:
  * funnel: attempts, accepted, per-reason reject counts (= audit pass-rates)
  * measured_min_k distribution (must be a point mass at k)
  * cross-cell match metrics (L, ops, whitespace tokens, nearest-anc distance)
  * optional BPE token spread across the roster tokenizers (diagnostic)
  * one fully rendered sample world (listing + 3 opacity site renderings)

Writes dry_run_summary.json and prints a human-readable report. Honestly
reports the achievable k-ladder: any starved / audit-hard cell is flagged, not
gap-filled.

Usage: python dry_run.py [N]   (default N=200)
"""
import json
import os
import statistics
import sys

for _cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "..", "expg"),):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from interp import execute, execute_cf, parse_program, value_at
from trace_format import make_listing_text, render_true_trace
import gen_depth as gd
import audit_depth as ad


def _targets_for(k):
    t = ad.default_targets()
    t["nearest_anc_dist"] = None if k == 0 else 4
    return t


def _try_tokenizers():
    """Best-effort load of roster tokenizers for a BPE diagnostic; returns
    {name: tokenizer} (possibly empty). CPU-only, no network if cached."""
    out = {}
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        for name, repo in (("qwen7b", "Qwen/Qwen2.5-7B-Instruct"),
                           ("llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct")):
            try:
                out[name] = AutoTokenizer.from_pretrained(repo)
            except Exception:
                pass
    except Exception:
        pass
    return out


def sample_world_printout(prog):
    lines = []
    lines.append("program_id: %s   k=%d  L=%d  ops=%d  ws_tokens=%d  site_line=%d"
                 % (prog["program_id"], prog["k"], prog["L"], prog["total_ops"],
                    prog["listing_ws_tokens"], prog["site_line"]))
    lines.append("operands=%s  site_true=%d  measured_min_k=%s  nearest_anc_dist=%d"
                 % (prog["operands"], prog["site_true_value"],
                    prog["audit_metrics"].get("measured_min_k"),
                    prog["audit_metrics"].get("nearest_anc_dist")))
    lines.append("--- listing ---")
    lines.append(make_listing_text(prog["stmt_texts"]))
    lines.append("--- site-line opacity arms (planted value spliced) ---")
    if prog["k"] == 0:
        f = prog["families"]["anchor_k0"]
        lines.append("anchor note plant: note: %s = %d  (true=%d, readable at line %d)"
                     % (f["planted_var"], f["planted_value"], f["true_value"],
                        prog["root_line"]))
    else:
        for opac in gd.OPACITIES:
            f = prog["families"]["depth_k%d_%s" % (prog["k"], opac)]
            lines.append("%-8s: %s" % (opac, f["site_body"]))
        f = prog["families"]["depth_k%d_bare" % prog["k"]]
        lines.append("(true value would be %d; planted %d; out_true=%d out_cf=%d)"
                     % (f["true_value"], f["planted_value"], prog["out_true"],
                        f["out_cf"]))
    return "\n".join(lines)


def run(N=200):
    toks = _try_tokenizers()
    summary = {"N_target": N, "tokenizers": list(toks.keys()), "cells": {}}
    report = []
    for k in gd.KS:
        tgt = _targets_for(k)
        af = ad.make_audit_fn(tgt)
        # record nearest_anc_dist into metrics for the printout
        def af_wrap(prog, _af=af):
            fails = _af(prog)
            prog.setdefault("audit_metrics", {})["nearest_anc_dist"] = \
                ad.nearest_ancestor_token_distance(prog)
            return fails
        progs, funnel = gd.generate_cell(k, N, 100000 + k * 777, audit_fn=af_wrap)
        min_ks = [p["audit_metrics"]["measured_min_k"] for p in progs]
        ws = sorted({p["listing_ws_tokens"] for p in progs})
        ancs = sorted({p["audit_metrics"]["nearest_anc_dist"] for p in progs})
        cell = {
            "k": k, "target": tgt,
            "accepted": len(progs), "starved": funnel["starved"],
            "attempts": funnel["attempts"],
            "accept_rate": round(len(progs) / max(1, funnel["attempts"]), 4),
            "rejects": funnel["rejects"],
            "measured_min_k_set": sorted(set(min_ks)),
            "measured_min_k_all_eq_k": all(m == k for m in min_ks) and bool(min_ks),
            "ws_tokens_set": ws, "nearest_anc_dist_set": ancs,
        }
        # BPE diagnostic on the accepted listings
        for tname, tok in toks.items():
            counts = [len(tok(make_listing_text(p["stmt_texts"]))["input_ids"])
                      for p in progs[:min(len(progs), 60)]]
            if counts:
                cell["bpe_%s" % tname] = {
                    "mean": round(statistics.mean(counts), 2),
                    "min": min(counts), "max": max(counts),
                    "sd": round(statistics.pstdev(counts), 2)}
        summary["cells"][str(k)] = cell
        report.append("=" * 78)
        report.append("CELL k=%d  accepted=%d/%d  attempts=%d  accept_rate=%.3f  starved=%s"
                      % (k, len(progs), N, funnel["attempts"], cell["accept_rate"],
                         funnel["starved"]))
        report.append("  measured_min_k: %s  (all==k: %s)"
                      % (cell["measured_min_k_set"], cell["measured_min_k_all_eq_k"]))
        report.append("  ws_tokens: %s   nearest_anc_dist: %s"
                      % (ws, ancs))
        for tname in toks:
            if "bpe_%s" % tname in cell:
                b = cell["bpe_%s" % tname]
                report.append("  BPE[%s]: mean=%.2f sd=%.2f range=[%d,%d]"
                              % (tname, b["mean"], b["sd"], b["min"], b["max"]))
        report.append("  reject reasons: %s" % (funnel["rejects"] or "{none}"))
        if progs:
            report.append("  --- sample world ---")
            report.append(sample_world_printout(progs[0]))
    # cross-cell token identity summary
    all_ws = set()
    for kk, c in summary["cells"].items():
        all_ws |= set(c["ws_tokens_set"])
    summary["all_cells_ws_tokens"] = sorted(all_ws)
    summary["ws_tokens_matched_across_all_cells"] = (len(all_ws) == 1)
    report.insert(0, "WHITESPACE-TOKEN COUNT ACROSS ALL CELLS: %s  (matched=%s)"
                  % (sorted(all_ws), len(all_ws) == 1))

    with open("dry_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    text = "\n".join(report)
    with open("dry_run_report.txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print("\n[wrote dry_run_summary.json and dry_run_report.txt]")
    return summary


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(N)
