#!/usr/bin/env python3
"""Opus 5 on the E10 bridge protocol — post-review probe (author-directed).

Scope: the two fig2 cells only (adjacent_contradiction n<=70, onehop_kc1 n<=60),
same regime2 worlds, same gold-gate/injection/grader as the shipped roster.
Isolated output dir (e10_opus5/) and a STANDALONE cost ledger (not summed into
the historical cumulative-budget check; disclosed in README_OPUS5.md).
Reuses e10bridge.py machinery verbatim via import + module-global override.
"""
import os, sys, json, argparse, math

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
RW = os.path.join(EXP, "improvement_plan", "iclr_exec", "recompute_wall")
BRIDGE_DIR = os.path.join(RW, "e10_api_bridge")
OUT = os.path.join(RW, "e10_opus5")
sys.path.insert(0, BRIDGE_DIR)

import e10bridge as B          # noqa: E402
import api_gen as A            # noqa: E402
import exph_common as C        # noqa: E402

os.makedirs(OUT, exist_ok=True)
B.OUT_DIR = OUT
B.CELLS = ("adjacent_contradiction", "onehop_kc1")
CFG = {"key": "claude-opus-5_bridge", "provider": "anthropic",
       "model_id": "claude-opus-5", "mode": "bridge", "temperature": None,
       "thinking": {"type": "disabled"}}
B.MODELS.append(CFG)

_orig_programs = B.programs
B.programs = lambda: [p for p in _orig_programs() if p["shape"] in ("kr1", "kc1")]

# Standalone ledger for this run; opus-class pricing assumed (in,out $/Mtok).
A.PRICES["claude-opus-5"] = (15.0, 75.0)
B.ledger = lambda: A.CostLedger(os.path.join(OUT, "cost_ledger.json"), extra_paths=())

with open(os.path.join(OUT, "README_OPUS5.md"), "w") as f:
    f.write("Opus 5 bridge probe, author-directed post-review run.\n"
            "Protocol/code: e10_api_bridge/e10bridge.py imported verbatim; only\n"
            "OUT_DIR, CELLS (2 fig2 cells), roster entry, and a standalone\n"
            "cost ledger overridden (this ledger is NOT summed into the E10\n"
            "cumulative budget files; add it to _extra_ledgers if extended).\n"
            "Worlds: regime2 kr1+kc1 subset (130 programs). Mode: bridge,\n"
            "temperature omitted, thinking disabled, max gold 512 / cont 384.\n")

args = argparse.Namespace(model="claude-opus-5_bridge")
print("== prepare =="); B.cmd_prepare(None)
print("== gold ==");    B.cmd_gold(args)
print("== perturb =="); B.cmd_perturb(args)
print("== validate =="); B.cmd_validate(args)

# ---- summary: both absorbed conventions + Wilson, vs shipped registry
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - hw), 4), round(min(1.0, c + hw), 4))

rows = C.read_jsonl(os.path.join(OUT, "validated_claude-opus-5.jsonl"))
summary = {}
for cell in B.CELLS:
    sub = [r for r in rows if r["condition"] == cell]
    live = [r for r in sub if not r.get("generation_failed")]
    n = len(live)
    k_plain = sum(bool(r.get("final_output_absorbed")) for r in live)
    k_flag = sum(bool(r.get("final_output_absorbed")) and not r.get("doubt_lex") for r in live)
    summary[cell] = {
        "n_manifest": len(sub), "n_live": n, "generation_failed": len(sub) - n,
        "absorbed_plain": {"k": k_plain, "rate": round(k_plain / n, 4) if n else None,
                           "wilson95": wilson(k_plain, n)},
        "absorbed_flag_first": {"k": k_flag, "rate": round(k_flag / n, 4) if n else None,
                                "wilson95": wilson(k_flag, n)},
        "doubt_lex": sum(bool(r.get("doubt_lex")) for r in live),
        "silently_corrected_like": sum(bool(r.get("final_output_valid")) for r in live),
        "no_output_claim": sum(bool(r.get("no_output_claim")) for r in live),
    }
led = B.ledger()
summary["_run"] = {"model_id": "claude-opus-5", "mode": "bridge",
                   "cost_usd_assumed_pricing": round(led.cost_usd(), 2)}
summary["_shipped_fig2_for_comparison"] = {
    "opus-4-8": {"readable": 0.0000, "computed": 1.0000},
    "sonnet-5": {"readable": 0.0105, "computed": 1.0000},
    "gpt-5.1":  {"readable": 0.0857, "computed": 1.0000}}
C.write_json(os.path.join(OUT, "opus5_summary.json"), summary)
print(json.dumps(summary, indent=1))
print("DONE")
