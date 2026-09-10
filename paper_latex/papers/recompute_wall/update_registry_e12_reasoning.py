#!/usr/bin/env python3
"""Add e12_fixes and reasoning_arm blocks to verified_numbers.json (Sec 6 and Sec 7 had
no registry support; number trace 2026-09-03). Sources: e12_fixes_r8/<arm>/summary_tables.json
(Haiku 4.5, R=8+greedy, registered depth), e11_run/results/E12_open/<model>/<arm>/summary_tables.json
(open models), reasoning_arm/arm_a_gpt51/summary_arm_a.json, reasoning_arm/arm_b_r1distill/out/summary_arm_b.json.
All-or-nothing: backup .bak_sep3, tmp write, round-trip validate, atomic replace. Refuses if blocks exist.
Reads results dirs only."""
import json, os, shutil, sys
PAPER = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall"
EXP = ("/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/"
       "improvement_plan/iclr_exec/recompute_wall")
REG = os.path.join(PAPER, "results", "verified_numbers.json"); BAK = REG + ".bak_sep3"
reg = json.load(open(REG))
for b in ("e12_fixes", "reasoning_arm"):
    if b in reg: sys.exit(f"{b} already present; refusing to overwrite")
ARMS = ["A0", "A1", "A2", "A3"]; CELLS = ["k0_bare", "k1_bare", "k5_bare"]
def cell_entry(c):
    return {"absorbed": c["absorbed"]["rate"], "wilson95": c["absorbed"]["wilson95"], "n": c["n"],
            "program_n": c.get("program_n"), "silently_corrected": c["silently_corrected"]["rate"],
            "flagged": c["flagged"]["rate"], "unresolved": c.get("unresolved", {}).get("rate"),
            "output_tokens_mean": c.get("output_tokens_mean"), "shows_recompute_rate": c.get("shows_recompute_rate")}
e12 = {}
for model, base in (("haiku", os.path.join(EXP, "e12_fixes_r8")),
                    ("llama8b", os.path.join(EXP, "e11_run", "results", "E12_open", "llama8b")),
                    ("qwen7b", os.path.join(EXP, "e11_run", "results", "E12_open", "qwen7b"))):
    m = {}
    for arm in ARMS:
        s = json.load(open(os.path.join(base, arm, "summary_tables.json")))
        cells = s["cells"]
        for cell in CELLS:
            c = cells[cell]
            assert 0 <= c["absorbed"]["rate"] <= 1 and c["n"] > 0, (model, arm, cell)
            m[f"{arm}|{cell}"] = cell_entry(c)
        m[f"{arm}|instruction_sha256"] = s.get("instruction_sha256")
    e12[model] = m
h = e12["haiku"]
e12["_prose"] = {
    "haiku_k1_absorbed_by_arm": [h[f"{a}|k1_bare"]["absorbed"] for a in ARMS],
    "haiku_k5_absorbed_by_arm": [h[f"{a}|k5_bare"]["absorbed"] for a in ARMS],
    "haiku_k1_output_tokens_by_arm": [h[f"{a}|k1_bare"]["output_tokens_mean"] for a in ARMS],
    "haiku_k1_shows_recompute_by_arm": [h[f"{a}|k1_bare"]["shows_recompute_rate"] for a in ARMS],
    "llama_k1_absorbed_min_max": [min(e12["llama8b"][f"{a}|k1_bare"]["absorbed"] for a in ARMS), max(e12["llama8b"][f"{a}|k1_bare"]["absorbed"] for a in ARMS)],
    "qwen_k1_absorbed_min_max": [min(e12["qwen7b"][f"{a}|k1_bare"]["absorbed"] for a in ARMS), max(e12["qwen7b"][f"{a}|k1_bare"]["absorbed"] for a in ARMS)],
    "qwen_k0_absorbed_min_max": [min(e12["qwen7b"][f"{a}|k0_bare"]["absorbed"] for a in ARMS), max(e12["qwen7b"][f"{a}|k0_bare"]["absorbed"] for a in ARMS)],
    "note": "haiku = e12_fixes_r8 (R=8+greedy, n=540/cell, 60 programs); open models = E12_open (n=700 at k0/k1; k5 attrition-limited, n in block). Arms: A0 baseline, A1 generic, A2 targeted, A3 demonstrated.",
}
a = json.load(open(os.path.join(EXP, "reasoning_arm", "arm_a_gpt51", "summary_arm_a.json")))
b = json.load(open(os.path.join(EXP, "reasoning_arm", "arm_b_r1distill", "out", "summary_arm_b.json")))
ra = {"gpt51": {}, "r1distill": {}}
for cell in a["cells"]:
    for cond in ("off", "on"):
        c = a["by_cell_cond"][cell][cond]
        ra["gpt51"][f"{cond}|{cell}"] = {"absorbed": c["final_output_absorbed"]["rate"], "wilson95": c["final_output_absorbed"]["wilson95"],
            "n": c["n_cf"], "silently_corrected": c["silently_corrected"]["rate"], "flagged": c["flagged"]["rate"],
            "mean_reasoning_tokens": c["mean_reasoning_tokens"]}
    t = a["rederive_taxonomy"][cell]
    ra["gpt51"][f"taxonomy|{cell}"] = {"summary_present": t["any_summary_present"], "n_on": a["by_cell_cond"][cell]["on"]["n_cf"],
        "strict": {k: v["count"] for k, v in t["strict_assign"].items()}, "loose": {k: v["count"] for k, v in t["loose_token"].items()}}
for cell in b["cells"]:
    for arm in ("B1", "B2"):
        c = b[arm][cell]
        ra["r1distill"][f"{arm}|{cell}"] = {"absorbed": c["final_output_absorbed"]["rate"], "wilson95": c["final_output_absorbed"]["wilson95"],
            "n": c["n_cf"], "silently_corrected": c["silently_corrected"]["rate"], "flagged": c["flagged"]["rate"],
            "no_output_claim": c.get("no_output_claim", {}).get("rate")}
    t = b["B2_rederive_taxonomy"][cell]
    ra["r1distill"][f"taxonomy_B2|{cell}"] = {"n": t["n"], "strict": {k: v["count"] for k, v in t["strict_assign"].items()},
        "loose": {k: v["count"] for k, v in t["loose_token"].items()}}
g = ra["gpt51"]; present = sum(g[f"taxonomy|{c}"]["summary_present"] for c in a["cells"]); n_on = sum(g[f"taxonomy|{c}"]["n_on"] for c in a["cells"])
deep = g["taxonomy|deep_kc5"]; r1deep = ra["r1distill"]["taxonomy_B2|deep_kc5"]
ra["_prose"] = {
    "gpt51_readable_off_on": [g["off|adjacent_contradiction"]["absorbed"], g["on|adjacent_contradiction"]["absorbed"]],
    "gpt51_onehop_off_on": [g["off|onehop_kc1"]["absorbed"], g["on|onehop_kc1"]["absorbed"]],
    "gpt51_deep_off_on": [g["off|deep_kc5"]["absorbed"], g["on|deep_kc5"]["absorbed"]],
    "gpt51_reasoning_tokens_onehop_deep": [g["on|onehop_kc1"]["mean_reasoning_tokens"], g["on|deep_kc5"]["mean_reasoning_tokens"]],
    "gpt51_empty_summary_frac_pooled": round(1 - present / n_on, 4),
    "gpt51_deep_summary_present_frac": round(deep["summary_present"] / deep["n_on"], 4),
    "gpt51_deep_strict_never_frac": round(deep["strict"]["never_rederive"] / deep["summary_present"], 4),
    "gpt51_deep_never_if_empty_counted": round((deep["strict"]["never_rederive"] + deep["n_on"] - deep["summary_present"]) / deep["n_on"], 4),
    "gpt51_deep_strict_surfaced_absorb_over_surfaced": [deep["strict"]["rederive_and_absorb"], deep["strict"]["rederive_and_absorb"] + deep["strict"]["rederive_and_correct"]],
    "r1_onehop_B1_B2": [ra["r1distill"]["B1|onehop_kc1"]["absorbed"], ra["r1distill"]["B2|onehop_kc1"]["absorbed"]],
    "r1_deep_B1_B2": [ra["r1distill"]["B1|deep_kc5"]["absorbed"], ra["r1distill"]["B2|deep_kc5"]["absorbed"]],
    "r1_deep_strict_never_frac": round(r1deep["strict"]["never_rederive"] / r1deep["n"], 4),
    "r1_readable_B2_strict_rederive_correct_frac": round(ra["r1distill"]["taxonomy_B2|adjacent_contradiction"]["strict"]["rederive_and_correct"] / ra["r1distill"]["taxonomy_B2|adjacent_contradiction"]["n"], 4),
    "r1_deep_strict_surfaced_absorb_over_surfaced": [r1deep["strict"]["rederive_and_absorb"], r1deep["strict"]["rederive_and_absorb"] + r1deep["strict"]["rederive_and_correct"]],
    "r1_readable_B1_no_output_claim": ra["r1distill"]["B1|adjacent_contradiction"]["no_output_claim"],
    "arm_a_usd": a["arm_a_gpt51_usd"], "note": "gpt-5.1 via Responses API, off=effort none, on=effort medium w/ summarized reasoning (taxonomy over summaries = lower bound); R1-Distill-7B B1 = raw continuation, B2 = think channel; DV final_output_absorbed; Wilson95 over rollouts (R=4/world). n = n_cf.",
}
reg["e12_fixes"] = e12; reg["reasoning_arm"] = ra
reg["_sources"]["e12_fixes"] = "e12_fixes_r8/<arm>/summary_tables.json cells.<cell> (Haiku 4.5, registered R=8+greedy) and e11_run/results/E12_open/<model>/<arm>/summary_tables.json (Llama-3.1-8B, Qwen2.5-7B); absorbed = final-output absorbed; Wilson95 over rollouts; cited in Sec 6 and app Sampling depth / Readable-control drift"
reg["_sources"]["reasoning_arm"] = "reasoning_arm/arm_a_gpt51/summary_arm_a.json by_cell_cond + rederive_taxonomy; reasoning_arm/arm_b_r1distill/out/summary_arm_b.json B1/B2 + B2_rederive_taxonomy; cited in Sec 7 and app Reasoning-arm protocol"
shutil.copy2(REG, BAK); tmp = REG + ".tmp_sep3"
with open(tmp, "w") as f: json.dump(reg, f, indent=1, sort_keys=True); f.write("\n")
json.load(open(tmp)); os.replace(tmp, REG)
print("added e12_fixes + reasoning_arm; backup", BAK)
print(json.dumps(e12["_prose"], indent=1)); print(json.dumps(ra["_prose"], indent=1))
