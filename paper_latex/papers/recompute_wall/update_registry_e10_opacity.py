#!/usr/bin/env python3
"""Add e10_cohorts, e10_wall_cells, opacity_by_depth blocks to verified_numbers.json
(number trace 2026-09-03: Sec 2 cohort taxonomy, Sec 3 five-operation cells and solve rates,
Sec 4 unresolved share, Sec 5 'at every depth', appendix solve rates / R=3 value / Fable spend).
Sources: e10_widen/results_skampere1/EXPG_PROGTRACE_<m>/{cohort,summary_tables}.json,
e10_api_bridge/{bridge_analysis.json,cohort_<m>.json}, e10_topup_fable/topup_analysis.json,
e11_run/results/E11_{haiku_r8,haiku,llama8b,qwen7b}/summary_tables.json.
All-or-nothing, backup .bak_sep3_e10, refuses if any block exists, reads results only."""
import json, os, shutil, sys
PAPER = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall"
EXP = ("/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/"
       "improvement_plan/iclr_exec/recompute_wall")
REG = os.path.join(PAPER, "results", "verified_numbers.json"); BAK = REG + ".bak_sep3_e10"
reg = json.load(open(REG))
for b in ("e10_cohorts", "e10_wall_cells", "opacity_by_depth"):
    if b in reg: sys.exit(f"{b} already present; refusing")
OPEN = ["qwen1p5b", "qwen7b", "qwen32b", "olmo7b", "llama8b"]
API = ["gpt-3.5-turbo-instruct", "gpt-4o", "gpt-4.1", "gpt-5.1", "claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-5", "claude-opus-4-8"]
CELLS = ["adjacent_contradiction", "opfree_kr1", "onehop_kc1", "deep_kc5"]
# ---- cohorts / solve rates
coh = {}
for m in OPEN:
    s = json.load(open(f"{EXP}/e10_widen/results_skampere1/EXPG_PROGTRACE_{m}/cohort.json"))["summary"]
    coh[m] = {"n_programs": s["n_programs"], "solve_rate": s["solve_rate"], "failure_taxonomy": s["failure_taxonomy"],
              "solve_rate_by_shape": {k: v["solve_rate"] for k, v in s["by_shape"].items()}}
    assert s["n_programs"] == 960
for m in API:
    s = json.load(open(f"{EXP}/e10_api_bridge/cohort_{m}.json"))["summary"]
    coh[m] = {"n_programs": s["n_programs"], "solve_rate": s["solve_rate"], "failure_taxonomy": s["failure_taxonomy"],
              "solve_rate_by_shape": {k: v["solve_rate"] for k, v in s["by_shape"].items()}}
top = json.load(open(f"{EXP}/e10_topup_fable/topup_analysis.json"))
coh["_prose"] = {"qwen1p5b_wrong_value_of_960": coh["qwen1p5b"]["failure_taxonomy"]["wrong_value"],
    "qwen1p5b_parse_of_960": coh["qwen1p5b"]["failure_taxonomy"].get("parse", 0),
    "olmo7b_wrong_value_of_960": coh["olmo7b"]["failure_taxonomy"]["wrong_value"],
    "solve_rate_low_two": {"qwen1p5b": coh["qwen1p5b"]["solve_rate"], "olmo7b": coh["olmo7b"]["solve_rate"]},
    "solve_rate_min_other_eleven": min(coh[m]["solve_rate"] for m in OPEN + API if m not in ("qwen1p5b", "olmo7b")),
    "fable5_usd": top["usd_by_model"]["claude-fable-5"],
    "note": "solve_rate = fraction of base programs whose gold trace is correct; open models 960 programs (E10 widen), API 250 (E10 bridge). Sec 2 line 14 cites wrong_value counts (853 / 600) and parse failures (5)."}
# ---- wall cells (all four cells per model)
wc = {}
for m in OPEN:
    c = json.load(open(f"{EXP}/e10_widen/results_skampere1/EXPG_PROGTRACE_{m}/summary_tables.json"))["cells"]
    wc[m] = {cell: {"rate": c[cell]["final_output_absorbed"]["rate"], "n": c[cell]["n"], "wilson95": c[cell]["final_output_absorbed"]["wilson95"], "presentation": "prefill"} for cell in CELLS}
br = json.load(open(f"{EXP}/e10_api_bridge/bridge_analysis.json"))["models"]
for m in API:
    c = br[m]["cells"]
    wc[m] = {cell: {"rate": c[cell]["absorbed"]["rate"], "n": c[cell]["absorbed"]["n"], "wilson95": c[cell]["absorbed"]["wilson95"],
                    "presentation": "completion" if m == "gpt-3.5-turbo-instruct" else "bridge"} for cell in CELLS}
for m, p in top["job1_pooled"].items():
    if m.startswith("claude-fable"): continue
    for cell in CELLS:
        q = p["cells"][cell]["pooled"]
        wc[m][cell + "|pooled_with_topup"] = {"rate": q["rate"], "n": q["n_cf"], "wilson95": q["wilson95"], "presentation": p["mode"]}
# consistency with fig2 (Anthropic pooled, others batch1)
f2 = reg["fig2"]; idx = {k: i for i, k in enumerate(f2["order"])}
alias = {"qwen1p5b": "qwen1p5b", "qwen7b": "qwen7b", "qwen32b": "qwen32b", "olmo7b": "olmo7b", "llama8b": "llama8b", "gpt-3.5-turbo-instruct": "gpt-3.5-instruct",
         "gpt-4o": "gpt-4o", "gpt-4.1": "gpt-4.1", "gpt-5.1": "gpt-5.1", "claude-haiku-4-5": "haiku-4-5", "claude-sonnet-4-5": "sonnet-4-5", "claude-sonnet-5": "sonnet-5", "claude-opus-4-8": "opus-4-8"}
for m, a in alias.items():
    key = "onehop_kc1|pooled_with_topup" if "onehop_kc1|pooled_with_topup" in wc[m] else "onehop_kc1"
    assert abs(wc[m][key]["rate"] - f2["computed"]["rate"][idx[a]]) < 1e-3, (m, wc[m][key]["rate"], f2["computed"]["rate"][idx[a]])
deep = {m: (wc[m].get("deep_kc5|pooled_with_topup", wc[m]["deep_kc5"])) for m in alias}
wc["_prose"] = {"deep_kc5_by_model": {m: [d["rate"], d["n"]] for m, d in deep.items()},
    "deep_kc5_min_over_n_ge_50": min(d["rate"] for d in deep.values() if d["n"] >= 50),
    "deep_kc5_below_floor": {m: [d["rate"], d["n"]] for m, d in deep.items() if d["n"] < 50},
    "note": "Fig 2 computed series = onehop_kc1 (Anthropic pooled batch1+topup, others batch1); deep_kc5 = five-operation cell cited in Sec 3; opfree_kr1 = copy cell reported separately (presentation-sensitive)."}
# ---- opacity by depth (E11)
ob = {}
for m, d in (("haiku", "E11_haiku_r8"), ("haiku_r3", "E11_haiku"), ("llama8b", "E11_llama8b"), ("qwen7b", "E11_qwen7b")):
    c = json.load(open(f"{EXP}/e11_run/results/{d}/summary_tables.json"))["cells"]
    ob[m] = {}
    for cell, v in c.items():
        e = {"absorbed": v["absorbed"]["rate"], "n": v["n"]}
        for k in ("unresolved", "silently_corrected", "flagged"):
            if isinstance(v.get(k), dict): e[k] = v[k]["rate"]
        ob[m][cell] = e
h = ob["haiku"]
ob["_prose"] = {"haiku_full_partial_min_over_k": min(h[c]["absorbed"] for c in h if c.endswith("_full") or c.endswith("_partial")),
    "llama_k1_by_opacity": [ob["llama8b"][f"k1_{o}"]["absorbed"] for o in ("bare", "full", "partial")],
    "qwen_k1_by_opacity": [ob["qwen7b"][f"k1_{o}"]["absorbed"] for o in ("bare", "full", "partial")],
    "qwen_k1_bare_unresolved": ob["qwen7b"]["k1_bare"].get("unresolved"),
    "llama_k1_bare_unresolved": ob["llama8b"]["k1_bare"].get("unresolved"),
    "haiku_r3_k1_partial": ob["haiku_r3"]["k1_partial"]["absorbed"],
    "note": "E11 depth x opacity grid; haiku = registered R=8 run, haiku_r3 = R=3 pilot kept as consistency check. Sec 4 'output failure' share = k1_bare unresolved."}
if ob["_prose"]["qwen_k1_bare_unresolved"] is not None:
    u = ob["_prose"]["qwen_k1_bare_unresolved"]; ob["_prose"]["qwen_k1_absorbed_among_output"] = round(ob["qwen7b"]["k1_bare"]["absorbed"] / (1 - u), 4)
    u2 = ob["_prose"]["llama_k1_bare_unresolved"]; ob["_prose"]["llama_k1_absorbed_among_output"] = round(ob["llama8b"]["k1_bare"]["absorbed"] / (1 - u2), 4) if u2 is not None else None
reg["e10_cohorts"] = coh; reg["e10_wall_cells"] = wc; reg["opacity_by_depth"] = ob
reg["_sources"]["e10_cohorts"] = "e10_widen/results_skampere1/EXPG_PROGTRACE_<m>/cohort.json summary (open, 960 programs) and e10_api_bridge/cohort_<m>.json summary (API, 250); Fable spend from e10_topup_fable/topup_analysis.json usd_by_model; cited Sec 2 line 14 (failure taxonomy), Sec 3 caveats, app Solvability gate, app Fable"
reg["_sources"]["e10_wall_cells"] = "e10_widen/results_skampere1/EXPG_PROGTRACE_<m>/summary_tables.json cells.<cell>.final_output_absorbed (open, prefill); e10_api_bridge/bridge_analysis.json models.<m>.cells.<cell>.absorbed (API batch1); e10_topup_fable/topup_analysis.json job1_pooled.<m>.cells.<cell>.pooled (Anthropic batch1+batch2); Wilson95; cited Sec 3 (one- and five-operation cells, copy cell) and Fig 2"
reg["_sources"]["opacity_by_depth"] = "e11_run/results/E11_<m>/summary_tables.json cells.k<k>_<opacity>.absorbed (+unresolved/silently_corrected/flagged); haiku from E11_haiku_r8; cited Sec 4 (unresolved share), Sec 5 ('at every depth'), app Sampling depth (R=3 0.996)"
shutil.copy2(REG, BAK); tmp = REG + ".tmp_sep3e10"
with open(tmp, "w") as f: json.dump(reg, f, indent=1, sort_keys=True); f.write("\n")
json.load(open(tmp)); os.replace(tmp, REG)
print("added e10_cohorts, e10_wall_cells, opacity_by_depth; backup", BAK)
for b in ("e10_cohorts", "e10_wall_cells", "opacity_by_depth"): print(b, json.dumps(reg[b]["_prose"], indent=1))
