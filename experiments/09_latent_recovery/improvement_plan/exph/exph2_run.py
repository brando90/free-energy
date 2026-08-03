"""EXPH2 frontier follow-ups, experiments B/C/D (descriptive rows only).

  ladder      (B) Anthropic generation ladder, prefill: claude-opus-4-1
              (opus-4-0 / sonnet-4-0 probed 404 -- recorded, skipped).
              Same six EXPH cells: 4 core x mid n<=100 + usable/inert n<=75.
  panel       (C) cross-lab SAME-FORMAT instruct panel under the registered
              bridge instruct prompt: gpt-4.1, gpt-5.1 (reasoning none),
              gpt-4o, claude-sonnet-5 (thinking disabled), claude-opus-4-8.
              4 core cells x mid x n<=100. PRIMARY: echo-corrected reuse
              (plant_dependent) on global_falsehood; anchor = instruct-Qwen.
  completions (D) TRUE text-completion continuation on gpt-3.5-turbo-instruct
              (v1/completions; protocol-identical to local prefill).
              4 core cells x mid x n<=100.

Budget: cumulative hard stop $100 across EXPH + EXPH2 ledgers.
Usage: exph2_run.py {gold,perturb,validate,summarize} --exp {ladder,panel,completions} [--model KEY]
"""
import argparse
import datetime as _dt
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exph_common as C  # noqa: E402
import api_gen as A  # noqa: E402

EXP = C.EXP
EXPH2_DIR = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS")
EXPH_LEDGER = os.path.join(EXP, "results", "EXPH_API_MODELS", "cost_ledger.json")
WORLDS_DIR = os.path.join(EXP, "results", "EXPH_API_MODELS", "worlds")
EXPA_DIR = os.path.join(EXP, "results", "EXPA_GLOBAL_EXPANSION")
SEED = 0

EXPERIMENTS = {
    "ladder": {
        "dir": "ladder", "mode_note": "prefill (pre-4.6 models accept assistant prefill)",
        "models": [
            {"key": "claude-opus-4-1_prefill", "provider": "anthropic",
             "model_id": "claude-opus-4-1", "mode": "prefill", "temperature": 0.0,
             "gold_key": "claude-opus-4-1"},
        ],
        "core_n": 100, "reuse_n": 75, "gold_target": 150, "gold_cap": 300,
        "reuse": True,
    },
    "panel": {
        "dir": "instruct_panel", "mode_note": "registered bridge instruct prompt (byte-identical)",
        "models": [
            {"key": "gpt-4.1_instruct", "provider": "openai", "model_id": "gpt-4.1",
             "mode": "instruct", "temperature": 0.0, "gold_key": "gpt-4.1"},
            {"key": "gpt-5.1_instruct", "provider": "openai", "model_id": "gpt-5.1",
             "mode": "instruct", "temperature": None, "reasoning_effort": "none",
             "gold_key": "gpt-5.1"},
            {"key": "gpt-4o_instruct", "provider": "openai", "model_id": "gpt-4o",
             "mode": "instruct", "temperature": 0.0, "gold_key": "gpt-4o"},
            {"key": "claude-sonnet-5_instruct", "provider": "anthropic",
             "model_id": "claude-sonnet-5", "mode": "instruct", "temperature": None,
             "thinking": {"type": "disabled"}, "gold_key": "claude-sonnet-5"},
            {"key": "claude-opus-4-8_instruct", "provider": "anthropic",
             "model_id": "claude-opus-4-8", "mode": "instruct", "temperature": None,
             "gold_key": "claude-opus-4-8"},
        ],
        "core_n": 100, "reuse_n": 0, "gold_target": 100, "gold_cap": 250,
        "reuse": False,
    },
    "completions": {
        "dir": "completions_probe", "mode_note": "TRUE text completion (v1/completions)",
        "models": [
            {"key": "gpt-3.5-turbo-instruct_completion", "provider": "openai",
             "model_id": "gpt-3.5-turbo-instruct", "mode": "completion",
             "temperature": 0.0, "gold_key": "gpt-3.5-turbo-instruct"},
        ],
        "core_n": 100, "reuse_n": 0, "gold_target": 100, "gold_cap": 250,
        "reuse": False,
    },
}


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def exp_paths(exp):
    d = os.path.join(EXPH2_DIR, EXPERIMENTS[exp]["dir"])
    os.makedirs(d, exist_ok=True)
    return d


def all_ledger_paths():
    return [EXPH_LEDGER] + [os.path.join(EXPH2_DIR, EXPERIMENTS[e]["dir"], "cost_ledger.json")
                            for e in EXPERIMENTS] + \
        [os.path.join(EXPH2_DIR, "regime2", "cost_ledger.json")]


def ledger(exp):
    path = os.path.join(exp_paths(exp), "cost_ledger.json")
    extra = [p for p in all_ledger_paths() if p != path]
    return A.CostLedger(path, extra_paths=extra)


def get_cfg(exp, key):
    for m in EXPERIMENTS[exp]["models"]:
        if m["key"] == key:
            return dict(m)
    raise KeyError(key)


def load_candidate_pool():
    from validator import parse_world
    pool = C.read_jsonl(os.path.join(EXPA_DIR, "candidate_pool.jsonl"))
    return [c for c in pool if not parse_world(c["question"])[2]]


def eval_gold(cand, raw):
    from validator import validate_continuation
    rec = {"problem_id": cand["problem_id"], "entity": cand["entity"]}
    if raw is None or raw.get("failed_generation"):
        rec.update({"eligible": False, "reason": "generation_failed"})
        return rec
    text = C.clean_continuation(raw.get("continuation", ""))
    steps = C.split_sentences(text)
    is_solved = C.solved(text, cand["target"])
    rec["solved"] = is_solved
    if not is_solved:
        rec.update({"eligible": False, "reason": "gold_not_solved"})
        return rec
    v = validate_continuation(cand["question"], [], None, text, cand["target"], cand["entity"])
    rec["gold_class"] = v["class"]
    if v["class"] != "valid_rederivation":
        rec.update({"eligible": False, "reason": "gold_not_validator_valid"})
        return rec
    pts = C.expc_injection_points(steps, cand["entity"])
    if pts is None:
        rec.update({"eligible": False, "reason": "too_few_intermediate_entity_steps"})
        return rec
    rec.update({"eligible": True, "reason": "eligible", "gold_steps": steps})
    return rec


def cmd_gold(args):
    exp = EXPERIMENTS[args.exp]
    cfg = get_cfg(args.exp, args.model)
    gold_key = cfg["gold_key"]
    led = ledger(args.exp)
    d = exp_paths(args.exp)
    raw_path = os.path.join(d, "raw_gold_%s.jsonl" % gold_key)
    coh_path = os.path.join(d, "cohort_%s.jsonl" % gold_key)
    pool = load_candidate_pool()
    existing_raw = {r["run_id"] for r in C.read_jsonl(raw_path)}
    cohort = {r["problem_id"]: r for r in C.read_jsonl(coh_path)}
    attempts = len(cohort)
    eligible = sum(1 for r in cohort.values() if r.get("eligible"))
    idx = 0
    while idx < len(pool) and pool[idx]["problem_id"] in cohort:
        idx += 1
    while eligible < exp["gold_target"] and attempts < exp["gold_cap"] and idx < len(pool):
        wave = pool[idx: idx + min(50, exp["gold_cap"] - attempts)]
        idx += len(wave)
        jobs = [{"run_id": C.sha_row(["EXPH2", args.exp, "gold", gold_key, c["problem_id"]]),
                 "kind": "gold", "problem_id": c["problem_id"],
                 "question": c["question"], "target": c["target"], "pre_text": None}
                for c in wave]
        A.generate_batch(cfg, jobs, led, raw_path, existing_raw)
        existing_raw = {r["run_id"] for r in C.read_jsonl(raw_path)}
        raw_map = {r["problem_id"]: r for r in C.read_jsonl(raw_path)}
        for cand in wave:
            rec = eval_gold(cand, raw_map.get(cand["problem_id"]))
            cohort[cand["problem_id"]] = rec
            attempts += 1
            eligible += int(bool(rec.get("eligible")))
        C.write_jsonl(coh_path, list(cohort.values()))
        print("[%s gold %s] attempts=%d eligible=%d cum=$%.2f"
              % (args.exp, gold_key, attempts, eligible,
                 led.cost_usd() + led.extra_cost_usd()), flush=True)

    w_elig = 0
    if exp["reuse"]:
        worlds = C.read_jsonl(os.path.join(WORLDS_DIR, "world_manifest.jsonl"))[:150]
        wraw_path = os.path.join(d, "raw_wgold_%s.jsonl" % gold_key)
        existing_w = {r["run_id"] for r in C.read_jsonl(wraw_path)}
        jobs = [{"run_id": C.sha_row(["EXPH2", args.exp, "wgold", gold_key, w["world_id"]]),
                 "kind": "wgold", "problem_id": w["world_id"],
                 "question": w["question"], "target": w["target"], "pre_text": None}
                for w in worlds]
        A.generate_batch(cfg, jobs, led, wraw_path, existing_w)
        wraw = {r["problem_id"]: r for r in C.read_jsonl(wraw_path)}
        wcoh = []
        for w in worlds:
            cand = {"problem_id": w["world_id"], "entity": w["entity"],
                    "question": w["question"], "target": w["target"]}
            rec = eval_gold(cand, wraw.get(w["world_id"]))
            rec.update({"world_id": w["world_id"], "family_id": w["family_id"],
                        "family_idx": w["family_idx"], "version": w["version"]})
            wcoh.append(rec)
        C.write_jsonl(os.path.join(d, "wcohort_%s.jsonl" % gold_key), wcoh)
        w_elig = sum(1 for r in wcoh if r.get("eligible"))

    n_solved = sum(1 for r in cohort.values() if r.get("solved"))
    funnel = {"gold_key": gold_key, "attempts": attempts, "solved": n_solved,
              "solve_rate": round(n_solved / max(1, attempts), 4),
              "eligible": eligible, "world_eligible": w_elig,
              "reached_target": eligible >= exp["gold_target"]}
    C.write_json(os.path.join(d, "gold_funnel_%s.json" % gold_key), funnel)
    led.save()
    print(json.dumps({"funnel": funnel,
                      "cumulative_usd": round(led.cost_usd() + led.extra_cost_usd(), 2)}))


def cmd_perturb(args):
    exp = EXPERIMENTS[args.exp]
    cfg = get_cfg(args.exp, args.model)
    key, gold_key = cfg["key"], cfg["gold_key"]
    led = ledger(args.exp)
    d = exp_paths(args.exp)
    pool_by_id = {c["problem_id"]: c for c in load_candidate_pool()}
    manifest_path = os.path.join(d, "manifest_%s.jsonl" % key)
    audit_path = os.path.join(d, "audit_%s.jsonl" % key)
    raw_path = os.path.join(d, "raw_pert_%s.jsonl" % key)
    mrows = C.read_jsonl(manifest_path)
    if not mrows:
        pool_order = [c["problem_id"] for c in load_candidate_pool()]
        cohort = {r["problem_id"]: r for r in C.read_jsonl(os.path.join(d, "cohort_%s.jsonl" % gold_key))}
        eligible_ids = [pid for pid in pool_order
                        if pid in cohort and cohort[pid].get("eligible")][:exp["core_n"]]
        for pid in eligible_ids:
            cand, rec = pool_by_id[pid], cohort[pid]
            steps = rec["gold_steps"]
            injs, skips = C.build_core_injections(cand["question"], cand["target"],
                                                  cand["entity"], steps, seed=SEED)
            for s in skips:
                C.append_jsonl(audit_path, {"problem_id": pid, "stage": "core_injection", **s})
            for inj in injs:
                mrows.append({
                    "run_id": C.sha_row(["EXPH2", args.exp, "pert", key, pid, inj["condition"]]),
                    "model_key": key, "provider": cfg["provider"], "model_id": cfg["model_id"],
                    "mode": cfg["mode"], "arm": "core", "condition": inj["condition"],
                    "problem_id": pid, "cluster_id": pid, "injection_position": "mid",
                    "seed": SEED, "sent_idx": inj["sent_idx"],
                    "question": cand["question"], "target": cand["target"],
                    "entity": cand["entity"], "prefix_steps": steps[:inj["sent_idx"]],
                    "injected_statement": inj["injected_statement"],
                    "audited_truth_status": inj["audited_truth_status"],
                    "truth_audit": inj["truth_audit"],
                    "measured_local_d": inj["measured_local_d"],
                    "poisoning_measurable": inj["condition"] == "global_falsehood",
                    "original_proof_validated": True,
                    "prompt_sha256": C.prompt_sha(), "created_at": now_iso(),
                })
        if exp["reuse"]:
            worlds_by_id = {w["world_id"]: w for w in
                            C.read_jsonl(os.path.join(WORLDS_DIR, "world_manifest.jsonl"))}
            wcoh = C.read_jsonl(os.path.join(d, "wcohort_%s.jsonl" % gold_key))
            for version in ("usable", "inert"):
                taken = 0
                for rec in sorted([r for r in wcoh
                                   if r.get("version") == version and r.get("eligible")],
                                  key=lambda x: x["family_idx"]):
                    if taken >= exp["reuse_n"]:
                        break
                    w = worlds_by_id[rec["world_id"]]
                    inj, err = C.build_reuse_injection(w, rec["gold_steps"], seed=SEED)
                    if inj is None:
                        C.append_jsonl(audit_path, {"world_id": rec["world_id"],
                                                    "stage": "reuse_injection", **err})
                        continue
                    mrows.append({
                        "run_id": C.sha_row(["EXPH2", args.exp, "pert", key,
                                             rec["world_id"], inj["condition"]]),
                        "model_key": key, "provider": cfg["provider"],
                        "model_id": cfg["model_id"], "mode": cfg["mode"], "arm": "reuse",
                        "condition": inj["condition"], "problem_id": rec["world_id"],
                        "cluster_id": rec["family_id"], "injection_position": "mid",
                        "seed": SEED, "sent_idx": inj["sent_idx"],
                        "question": w["question"], "target": w["target"],
                        "entity": w["entity"],
                        "prefix_steps": rec["gold_steps"][:inj["sent_idx"]],
                        "injected_statement": inj["injected_statement"],
                        "audited_truth_status": "false", "truth_audit": inj["truth_audit"],
                        "measured_local_d": 1,
                        "poisoning_measurable": version == "usable",
                        "original_proof_validated": True,
                        "prompt_sha256": C.prompt_sha(), "created_at": now_iso(),
                    })
                    taken += 1
        C.write_jsonl(manifest_path, mrows)
    existing = {r["run_id"] for r in C.read_jsonl(raw_path)}
    jobs = [{"run_id": m["run_id"], "kind": "pert", "problem_id": m["problem_id"],
             "condition": m["condition"], "question": m["question"], "target": m["target"],
             "pre_text": C.prefix_text(m["prefix_steps"], m["injected_statement"])}
            for m in mrows]
    n = A.generate_batch(cfg, jobs, led, raw_path, existing)
    led.save()
    print("[%s perturb %s] manifest=%d new=%d cum=$%.2f"
          % (args.exp, key, len(mrows), n, led.cost_usd() + led.extra_cost_usd()))


def cmd_validate(args):
    d = exp_paths(args.exp)
    key = args.model
    manifest = C.read_jsonl(os.path.join(d, "manifest_%s.jsonl" % key))
    raw = {r["run_id"]: r for r in C.read_jsonl(os.path.join(d, "raw_pert_%s.jsonl" % key))}
    rows = []
    for m in manifest:
        v = C.validate_row(m, raw.get(m["run_id"]))
        cont = v.get("continuation") or ""
        v["plant_dependent"] = bool(C.plant_dependent_content(m, cont)) if cont else False
        rows.append(v)
    C.write_jsonl(os.path.join(d, "validated_%s.jsonl" % key), rows)
    print("[%s validate %s] rows=%d" % (args.exp, key, len(rows)))


def cmd_summarize(args):
    metrics = C.METRICS + ("plant_dependent",)
    out = {"created_at": now_iso(), "experiments": {}}
    for exp_name, exp in EXPERIMENTS.items():
        d = os.path.join(EXPH2_DIR, exp["dir"])
        per_model = {}
        for m in exp["models"]:
            p = os.path.join(d, "validated_%s.jsonl" % m["key"])
            if not os.path.exists(p):
                continue
            rows = C.read_jsonl(p)
            per = {}
            for cond in list(C.CORE_CONDITIONS) + list(C.REUSE_CONDITIONS):
                sub = [r for r in rows if r["condition"] == cond]
                if sub:
                    per[cond] = C.summarize_cell(sub, seed=SEED, metrics=metrics)
            per_model[m["key"]] = per
        funnels = {}
        for m in exp["models"]:
            p = os.path.join(d, "gold_funnel_%s.json" % m["gold_key"])
            if os.path.exists(p):
                funnels[m["gold_key"]] = json.load(open(p))
        if per_model:
            out["experiments"][exp_name] = {"models": per_model, "funnels": funnels,
                                            "note": exp["mode_note"]}
    led = ledger("panel")
    out["cumulative_usd"] = round(led.cost_usd() + led.extra_cost_usd(), 4)
    C.write_json(os.path.join(EXPH2_DIR, "summary_bcd.json"), out)
    print("summary_bcd.json written; cumulative=$%.2f" % out["cumulative_usd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gold", "perturb", "validate", "summarize"])
    ap.add_argument("--exp", choices=sorted(EXPERIMENTS), default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    if args.stage == "gold":
        cmd_gold(args)
    elif args.stage == "perturb":
        cmd_perturb(args)
    elif args.stage == "validate":
        cmd_validate(args)
    elif args.stage == "summarize":
        cmd_summarize(args)


if __name__ == "__main__":
    main()
