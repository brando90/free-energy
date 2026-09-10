#!/usr/bin/env python3
"""Add e16_probe_steer block (E16 probes + steering nulls, Qwen2.5-7B) to verified_numbers.json for
the proposed discussion paragraph. Sources: e16_steering/results/qwen7b/{probe_neutral.json,
probe_check.json, screen/screen_summary.json, screen_ml/screen_ml_summary.json,
steer_stub/steer_stub_summary.json, confirm/confirm_summary.json}. All-or-nothing, backup
.bak_sep4_e16, refuses if block exists, reads results only."""
import json, os, shutil, sys
PAPER = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall"
R = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e16_steering/results/qwen7b"
REG = os.path.join(PAPER, "results", "verified_numbers.json"); BAK = REG + ".bak_sep4_e16"
reg = json.load(open(REG))
if "e16_probe_steer" in reg: sys.exit("e16_probe_steer already present; refusing")
pn = json.load(open(os.path.join(R, "probe_neutral.json"))); pc = json.load(open(os.path.join(R, "probe_check.json")))
lin = {int(l): v["linear"]["mean"] for l, v in pn["layers"].items()}
blk = {"model": "qwen7b", "probe_neutral_linear_auc_by_layer": lin, "probe_neutral_n_pairs": pn["n_pairs"],
       "probe_neutral_max_auc_layers_1_25": max(lin[l] for l in range(1, 26)),
       "probe_neutral_auc_26_27_28": [lin[26], lin[27], lin[28]],
       "probe_span_L27_auc": pc["layers"]["27"]["auc_orig"], "probe_span_L27_auc_signbalanced": pc["layers"]["27"]["auc_balanced"],
       "excluded_pair_note": "k1_43d06a_00701770 is not a minimal pair (worked gold line); AUCs above computed on all 150, geometry verified unchanged on 149 (analysis_geometry_sep03/geometry_subsets.json)"}
batteries = {}
def battery(name, conds, base):
    ok = {k: v for k, v in conds.items() if v.get("control_ok")}
    batteries[name] = {"n_conditions": len(conds), "n_control_ok": len(ok), "baseline": base,
        "absorbed_min_max_control_ok": [min(v["absorbed"] for v in ok.values()), max(v["absorbed"] for v in ok.values())],
        "checked_max_control_ok": max(v["checked"] for v in ok.values()),
        "checked_max_all": max(v["checked"] for v in conds.values()),
        "conds_checked_ge_0p15": {k: {"checked": v["checked"], "control_valid": v.get("control_valid")} for k, v in conds.items() if v["checked"] >= 0.15}}
s = json.load(open(os.path.join(R, "screen", "screen_summary.json"))); battery("screen_single_layer", s["conditions"], s["baseline"])
s = json.load(open(os.path.join(R, "screen_ml", "screen_ml_summary.json"))); battery("screen_multi_layer", s["conditions"], s["baseline"])
s = json.load(open(os.path.join(R, "steer_stub", "steer_stub_summary.json"))); conds = {k: v for k, v in s.items() if isinstance(v, dict) and "absorbed" in v}
battery("steer_stub_generation_position", conds, {"absorbed": 0.9, "checked": 0.0, "n": 40, "note": "same worlds/baseline as screen batteries"})
c = json.load(open(os.path.join(R, "confirm", "confirm_summary.json")))
blk["confirm_breach_any"] = any(v.get("breach_all_conjuncts") for v in c.values() if isinstance(v, dict))
blk["batteries"] = batteries
oks = [b["absorbed_min_max_control_ok"] for b in batteries.values()]
blk["_prose"] = {"n_conditions_total": sum(b["n_conditions"] for b in batteries.values()),
    "absorbed_min_max_control_ok_all_batteries": [min(o[0] for o in oks), max(o[1] for o in oks)],
    "checked_max_control_ok_all_batteries": max(b["checked_max_control_ok"] for b in batteries.values()),
    "baseline_absorbed": 0.9, "all_high_checked_conds_have_control_valid_zero": all(v["control_valid"] == 0.0 for b in batteries.values() for v in b["conds_checked_ge_0p15"].values()),
    "note": "screen tier n=40 greedy per condition + 20 unperturbed controls; control_ok = unperturbed traces still continued validly; checked = classifier label (hand-read as noise where rare); confirm arms n=450 all BREACH=False"}
reg["e16_probe_steer"] = blk
reg["_sources"]["e16_probe_steer"] = "e16_steering/results/qwen7b/probe_neutral.json layers.<l>.linear.mean (150 minimal pairs, neutral-stub position, 5-fold x 5-seed); probe_check.json layers.27.auc_orig (site-span); screen/screen_summary.json, screen_ml/screen_ml_summary.json, steer_stub/steer_stub_summary.json conditions.<c>.{absorbed,checked,control_ok}; confirm/confirm_summary.json; cited in the proposed Sec 9 probe paragraph"
shutil.copy2(REG, BAK); tmp = REG + ".tmp_e16"
with open(tmp, "w") as f: json.dump(reg, f, indent=1, sort_keys=True); f.write("\n")
json.load(open(tmp)); os.replace(tmp, REG)
print(json.dumps({k: v for k, v in blk.items() if k != "probe_neutral_linear_auc_by_layer"}, indent=1)[:3000])
