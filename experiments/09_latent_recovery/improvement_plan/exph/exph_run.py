"""EXPH API fan-out runner (stage 2; gated on the bridge arm passing).

Stages (per model key from api_gen.MODEL_CONFIGS, resolved via model_resolution.json):
  smoke     -- 1 gold + 1 continuation request per model; verifies params
               (e.g. gpt-5.1 reasoning_effort="none") and records substitutions.
  gold      -- own-trace gold rollouts: legacy pool until >=150 pass the double
               filter (cap 400 attempts) + generated worlds (cap 150 attempts).
               Golds are shared across modes of the same model (identical prompts).
  perturb   -- 4 core conditions x mid x <=125 own-trace injections (audited)
               + cat_false_usable_d1 / cat_false_inert_d1 x <=80.
  validate  -- closure validation + echo + stated-complement + doubt per row.
  summarize -- per-cell tables with Wilson + problem-cluster bootstrap CIs.
  report    -- EXPH_REPORT.md + README.md + combined artifacts.

Descriptive/exploratory rows only. Cost ledger + $100 hard stop enforced in api_gen.
"""
import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exph_common as C  # noqa: E402
import api_gen as A  # noqa: E402

EXP = C.EXP
API_DIR = os.path.join(C.RESULTS_DIR, "api")
BRIDGE_DIR = os.path.join(C.RESULTS_DIR, "bridge")
WORLDS_DIR = os.path.join(C.RESULTS_DIR, "worlds")
EXPA_DIR = os.path.join(EXP, "results", "EXPA_GLOBAL_EXPANSION")
RESOLUTION_PATH = os.path.join(C.RESULTS_DIR, "model_resolution.json")

GOLD_TARGET = 150
GOLD_CAP = 400
WORLD_GOLD_CAP = 150
N_CORE = 125
N_REUSE = 80
GOLD_WAVE = 50
SEED = 0

EXPA_LOCKED_MID = {  # results/EXPA_GLOBAL_EXPANSION/summary_tables.json (HF prefill, locked)
    "benign_paraphrase": {"n": 150, "valid": 0.88, "poisoned": 0.0, "doubt": 0.1267, "unparsed": 0.02},
    "global_falsehood": {"n": 150, "valid": 0.36, "poisoned": 0.3733, "doubt": 0.0133, "unparsed": 0.0133},
    "one_hop_falsehood": {"n": 150, "valid": 0.64, "poisoned": 0.02, "doubt": 0.2733, "unparsed": 0.0133},
    "true_interruption": {"n": 150, "valid": 0.8667, "poisoned": 0.0, "doubt": 0.0267, "unparsed": 0.0133},
}


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def ledger():
    return A.CostLedger(os.path.join(C.RESULTS_DIR, "cost_ledger.json"))


def effective_config(key):
    cfg = A.get_config(key)
    if os.path.exists(RESOLUTION_PATH):
        res = json.load(open(RESOLUTION_PATH))
        if key in res:
            entry = res[key]
            if entry.get("excluded"):
                raise SystemExit("model %s excluded: %s" % (key, entry.get("why")))
            cfg.update(entry.get("overrides", {}))
    return cfg


def _resolution():
    return json.load(open(RESOLUTION_PATH)) if os.path.exists(RESOLUTION_PATH) else {}


def _save_resolution(res):
    C.write_json(RESOLUTION_PATH, res)


# ------------------------------------------------------------------- smoke

SMOKE_Q = ("Every yumpus is a dumpus. Dumpuses are tumpuses. Tumpuses are not bright. "
           "Sam is a yumpus.")
SMOKE_T = "Sam is not bright."


def smoke_one(cfg, led):
    msgs_gold = A.build_request_messages(cfg, SMOKE_Q, SMOKE_T, None)
    out1 = A.call_model(cfg, msgs_gold, led)
    pre = "Sam is a yumpus. Every yumpus is a dumpus. Sam is a dumpus."
    msgs_cont = A.build_request_messages(cfg, SMOKE_Q, SMOKE_T, pre)
    out2 = A.call_model(cfg, msgs_cont, led)
    return out1, out2


def cmd_smoke(args):
    led = ledger()
    res = _resolution()
    keys = [args.model] if args.model else [c["key"] for c in A.MODEL_CONFIGS]
    for key in keys:
        cfg = A.get_config(key)
        if key in res:
            cfg.update(res[key].get("overrides", {}))
        out1, out2 = smoke_one(cfg, led)
        ok = out1.get("text") and out2.get("text")
        print("== %s ok=%s" % (key, bool(ok)))
        for tag, o in (("gold", out1), ("cont", out2)):
            t = (o.get("text") or "")[:110].replace("\n", " ")
            print("   %s: err=%s text=%r" % (tag, o.get("error"), t))
        if not ok and key == "gpt-5.1_instruct":
            # substitution ladder per design: gpt-5-chat-latest, else gpt-4.1-mini
            for sub in ("gpt-5-chat-latest", "gpt-4.1-mini"):
                scfg = dict(cfg)
                scfg["model_id"] = sub
                scfg["reasoning_effort"] = None
                scfg["temperature"] = 0.0 if sub == "gpt-4.1-mini" else scfg.get("temperature")
                o1, o2 = smoke_one(scfg, led)
                if o1.get("text") and o2.get("text"):
                    res[key] = {"overrides": {"model_id": sub, "reasoning_effort": None,
                                              "temperature": scfg.get("temperature"),
                                              "gold_key": sub},
                                "why": "gpt-5.1 failed smoke (%s); substituted %s"
                                       % (out1.get("error") or out2.get("error"), sub)}
                    print("   -> substituted %s" % sub)
                    break
            else:
                res[key] = {"excluded": True,
                            "why": "gpt-5.1 and substitutes failed smoke"}
        elif not ok:
            res.setdefault(key, {})["smoke_failed"] = True
    _save_resolution(res)
    led.save()
    print("ledger total: $%.4f" % led.cost_usd())


# -------------------------------------------------------------------- gold

def load_candidate_pool():
    """Legacy pool restricted to validator-certifiable (fully in-grammar) questions.

    Faithful to the locked run: all 150 locked EXPA eligible problems are strictly
    in-grammar; composed-grammar candidates can never pass the double filter (the
    validator cannot certify their proofs by construction), and burning the
    400-attempt cap on them would starve the cohort (the locked EXPA run needed
    3,032 UNCAPPED gold attempts against the mixed pool, 5.0% eligibility).
    651/5,819 pool candidates survive this filter. Disclosed in EXPH_REPORT.md."""
    from validator import parse_world
    pool = C.read_jsonl(os.path.join(EXPA_DIR, "candidate_pool.jsonl"))
    return [c for c in pool if not parse_world(c["question"])[2]]


def gold_paths(gold_key):
    return {
        "raw": os.path.join(API_DIR, "raw_gold_%s.jsonl" % gold_key),
        "cohort": os.path.join(API_DIR, "cohort_%s.jsonl" % gold_key),
        "wraw": os.path.join(API_DIR, "raw_wgold_%s.jsonl" % gold_key),
        "wcohort": os.path.join(API_DIR, "wcohort_%s.jsonl" % gold_key),
    }


def eval_gold(cand, raw):
    """EXPA double filter on an API gold rollout."""
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
    cfg = effective_config(args.model)
    gold_key = cfg["gold_key"]
    led = ledger()
    paths = gold_paths(gold_key)
    pool = load_candidate_pool()
    pool_by_id = {c["problem_id"]: c for c in pool}

    existing_raw = {r["run_id"]: r for r in C.read_jsonl(paths["raw"])}
    cohort = {r["problem_id"]: r for r in C.read_jsonl(paths["cohort"])}
    attempts = len(cohort)
    eligible = sum(1 for r in cohort.values() if r.get("eligible"))
    idx = 0
    # advance idx past already-attempted candidates (pool order is deterministic)
    while idx < len(pool) and pool[idx]["problem_id"] in cohort:
        idx += 1
    while eligible < GOLD_TARGET and attempts < GOLD_CAP and idx < len(pool):
        wave = pool[idx: idx + min(GOLD_WAVE, GOLD_CAP - attempts)]
        idx += len(wave)
        jobs = [{"run_id": C.sha_row(["EXPH", "gold", gold_key, c["problem_id"]]),
                 "kind": "gold", "problem_id": c["problem_id"],
                 "question": c["question"], "target": c["target"], "pre_text": None}
                for c in wave]
        A.generate_batch(cfg, jobs, led, paths["raw"], existing_raw.keys())
        raw_map = {r["problem_id"]: r for r in C.read_jsonl(paths["raw"])}
        existing_raw = {r["run_id"]: r for r in C.read_jsonl(paths["raw"])}
        for c in wave:
            rec = eval_gold(c, raw_map.get(c["problem_id"]))
            cohort[c["problem_id"]] = rec
            attempts += 1
            eligible += int(bool(rec.get("eligible")))
        C.write_jsonl(paths["cohort"], list(cohort.values()))
        print("[gold %s] attempts=%d eligible=%d ledger=$%.2f"
              % (gold_key, attempts, eligible, led.cost_usd()), flush=True)

    # ---- generated worlds golds
    worlds = C.read_jsonl(os.path.join(WORLDS_DIR, "world_manifest.jsonl"))[:WORLD_GOLD_CAP]
    existing_w = {r["run_id"] for r in C.read_jsonl(paths["wraw"])}
    jobs = [{"run_id": C.sha_row(["EXPH", "wgold", gold_key, w["world_id"]]),
             "kind": "wgold", "problem_id": w["world_id"],
             "question": w["question"], "target": w["target"], "pre_text": None}
            for w in worlds]
    A.generate_batch(cfg, jobs, led, paths["wraw"], existing_w)
    wraw = {r["problem_id"]: r for r in C.read_jsonl(paths["wraw"])}
    wcoh = []
    for w in worlds:
        cand = {"problem_id": w["world_id"], "entity": w["entity"],
                "question": w["question"], "target": w["target"]}
        rec = eval_gold(cand, wraw.get(w["world_id"]))
        rec.update({"world_id": w["world_id"], "family_id": w["family_id"],
                    "family_idx": w["family_idx"], "version": w["version"]})
        wcoh.append(rec)
    C.write_jsonl(paths["wcohort"], wcoh)
    w_elig = sum(1 for r in wcoh if r.get("eligible"))

    n_solved = sum(1 for r in cohort.values() if r.get("solved"))
    funnel = {"gold_key": gold_key, "attempts": attempts,
              "solved": n_solved, "solve_rate": round(n_solved / max(1, attempts), 4),
              "eligible": eligible, "eligible_rate": round(eligible / max(1, attempts), 4),
              "world_attempts": len(wcoh), "world_eligible": w_elig,
              "reached_target": eligible >= GOLD_TARGET}
    C.write_json(os.path.join(API_DIR, "gold_funnel_%s.json" % gold_key), funnel)
    led.save()

    # ---- cost checkpoint / projection
    total = led.cost_usd()
    model_cost = led.cost_usd(cfg["model_id"])
    n_req = led.data["models"].get(cfg["model_id"], {}).get("requests", 1)
    per_req = model_cost / max(1, n_req)
    modes = sum(1 for m in A.MODEL_CONFIGS if m["gold_key"] == gold_key)
    n_pert = modes * (4 * min(N_CORE, eligible) + 2 * min(N_REUSE, w_elig))
    projection = n_pert * per_req * 1.3  # perturb prompts are longer (prefix + instruction)
    print(json.dumps({"funnel": funnel, "ledger_total_usd": round(total, 2),
                      "model_cost_usd": round(model_cost, 2),
                      "projected_perturb_usd_all_modes": round(projection, 2)}, indent=2))
    if total + projection > A.HARD_BUDGET_USD:
        raise SystemExit("HARD STOP: projected total $%.2f exceeds $%.2f"
                         % (total + projection, A.HARD_BUDGET_USD))


# ------------------------------------------------------------------ perturb

def cmd_perturb(args):
    cfg = effective_config(args.model)
    key, gold_key = cfg["key"], cfg["gold_key"]
    led = ledger()
    paths = gold_paths(gold_key)
    pool_by_id = {c["problem_id"]: c for c in load_candidate_pool()}
    worlds_by_id = {w["world_id"]: w for w in C.read_jsonl(
        os.path.join(WORLDS_DIR, "world_manifest.jsonl"))}

    manifest_path = os.path.join(API_DIR, "manifest_%s.jsonl" % key)
    audit_path = os.path.join(API_DIR, "audit_%s.jsonl" % key)
    raw_path = os.path.join(API_DIR, "raw_pert_%s.jsonl" % key)
    mrows = C.read_jsonl(manifest_path)
    if not mrows:
        pool_order = [c["problem_id"] for c in load_candidate_pool()]
        cohort = {r["problem_id"]: r for r in C.read_jsonl(paths["cohort"])}
        eligible_ids = [pid for pid in pool_order
                        if pid in cohort and cohort[pid].get("eligible")][:N_CORE]
        for pid in eligible_ids:
            cand, rec = pool_by_id[pid], cohort[pid]
            steps = rec["gold_steps"]
            injs, skips = C.build_core_injections(cand["question"], cand["target"],
                                                  cand["entity"], steps, seed=SEED)
            for s in skips:
                C.append_jsonl(audit_path, {"problem_id": pid, "stage": "core_injection", **s})
            for inj in injs:
                mrows.append({
                    "run_id": C.sha_row(["EXPH", "pert", key, pid, inj["condition"]]),
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
        wcoh = C.read_jsonl(paths["wcohort"])
        for version in ("usable", "inert"):
            taken = 0
            for rec in sorted([r for r in wcoh if r.get("version") == version and r.get("eligible")],
                              key=lambda x: x["family_idx"]):
                if taken >= N_REUSE:
                    break
                w = worlds_by_id[rec["world_id"]]
                inj, err = C.build_reuse_injection(w, rec["gold_steps"], seed=SEED)
                if inj is None:
                    C.append_jsonl(audit_path, {"world_id": rec["world_id"],
                                                "stage": "reuse_injection", **err})
                    continue
                mrows.append({
                    "run_id": C.sha_row(["EXPH", "pert", key, rec["world_id"], inj["condition"]]),
                    "model_key": key, "provider": cfg["provider"], "model_id": cfg["model_id"],
                    "mode": cfg["mode"], "arm": "reuse", "condition": inj["condition"],
                    "problem_id": rec["world_id"], "cluster_id": rec["family_id"],
                    "injection_position": "mid", "seed": SEED, "sent_idx": inj["sent_idx"],
                    "question": w["question"], "target": w["target"], "entity": w["entity"],
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
    print("[perturb %s] manifest=%d new=%d ledger=$%.2f"
          % (key, len(mrows), n, led.cost_usd()))


# ----------------------------------------------------------------- validate

def cmd_validate(args):
    key = args.model
    manifest = C.read_jsonl(os.path.join(API_DIR, "manifest_%s.jsonl" % key))
    raw = {r["run_id"]: r for r in C.read_jsonl(os.path.join(API_DIR, "raw_pert_%s.jsonl" % key))}
    rows = [C.validate_row(m, raw.get(m["run_id"])) for m in manifest]
    C.write_jsonl(os.path.join(API_DIR, "validated_%s.jsonl" % key), rows)
    print("[validate %s] rows=%d" % (key, len(rows)))


# ---------------------------------------------------------------- summarize

def model_keys_present():
    keys = []
    for cfgc in A.MODEL_CONFIGS:
        if os.path.exists(os.path.join(API_DIR, "validated_%s.jsonl" % cfgc["key"])):
            keys.append(cfgc["key"])
    return keys


def cmd_summarize(args):
    led = ledger()
    tables = {"created_at": now_iso(), "models": {}, "bridge": {}, "gold_funnels": {},
              "expa_locked_mid_anchors": EXPA_LOCKED_MID,
              "cost": {"total_usd": round(led.cost_usd(), 4),
                       "by_model": led.data.get("usd_by_model", {})}}
    for name in ("bridge_legacy_summary", "bridge_worlds_summary", "bridge_gate"):
        p = os.path.join(BRIDGE_DIR, name + ".json")
        if os.path.exists(p):
            tables["bridge"][name] = json.load(open(p))
    combined = []
    for key in model_keys_present():
        rows = C.read_jsonl(os.path.join(API_DIR, "validated_%s.jsonl" % key))
        combined.extend(rows)
        per = {}
        for cond in list(C.CORE_CONDITIONS) + list(C.REUSE_CONDITIONS):
            sub = [r for r in rows if r["condition"] == cond]
            if sub:
                per[cond] = C.summarize_cell(sub, seed=SEED)
        tables["models"][key] = per
    for cfgc in A.MODEL_CONFIGS:
        p = os.path.join(API_DIR, "gold_funnel_%s.json" % cfgc["gold_key"])
        if os.path.exists(p):
            tables["gold_funnels"][cfgc["gold_key"]] = json.load(open(p))
    if os.path.exists(RESOLUTION_PATH):
        tables["model_resolution"] = json.load(open(RESOLUTION_PATH))
    C.write_json(os.path.join(C.RESULTS_DIR, "summary_tables.json"), tables)
    C.write_jsonl(os.path.join(C.RESULTS_DIR, "validated_outputs.jsonl"), combined)
    # combined manifest + raw for the standard schema
    allm, allraw = [], []
    for key in model_keys_present():
        allm.extend(C.read_jsonl(os.path.join(API_DIR, "manifest_%s.jsonl" % key)))
        allraw.extend(C.read_jsonl(os.path.join(API_DIR, "raw_pert_%s.jsonl" % key)))
    C.write_jsonl(os.path.join(C.RESULTS_DIR, "manifest.jsonl"), allm)
    C.write_jsonl(os.path.join(C.RESULTS_DIR, "raw_generations.jsonl"), allraw)
    print("summary_tables.json written; models=%s" % list(tables["models"]))


# ------------------------------------------------------------------- report

def git_hash():
    try:
        return subprocess.check_output(["git", "-C", EXP, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def write_metadata():
    meta = {
        "experiment": "EXPH_API_MODELS",
        "status": "descriptive/exploratory arm; NOT confirmatory; no pre-registration involvement",
        "created_at": now_iso(),
        "git_commit_hash": git_hash(),
        "prompt": {"instruction": C.INSTR, "fewshot": C.FEWSHOT,
                   "continuation_instruction": C.CONT_INSTR,
                   "sha256": C.prompt_sha()},
        "modes": {
            "prefill": "user turn = INSTR+FEWSHOT+Q...A:; assistant turn pre-filled with prefix+planted step (Anthropic models that accept prefill)",
            "instruct": "single user message: INSTR+FEWSHOT+Q...A: <prefix+planted step> + continuation instruction; identical rendered content across providers",
        },
        "models": [{k: v for k, v in m.items()} for m in A.MODEL_CONFIGS],
        "decoding": {"max_tokens": A.MAX_TOKENS, "concurrency": A.CONCURRENCY},
        "caps": {"gold_target": GOLD_TARGET, "gold_cap": GOLD_CAP,
                 "world_gold_cap": WORLD_GOLD_CAP, "n_core_per_cell": N_CORE,
                 "n_reuse_per_cell": N_REUSE, "seed": SEED},
        "budget": {"hard_stop_usd": A.HARD_BUDGET_USD, "target_usd": "25-60"},
        "bridge": {"model": "Qwen/Qwen2.5-7B-Instruct",
                   "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                   "backend": "vLLM (tooling/vllm_gen.py), greedy, max_new_tokens=256"},
        "api_keys": "read from environment only (env file path documented in lab notes); never logged or stored",
        "excluded_models": {
            "claude-fable-5 / o-series / DeepSeek-R1": "deliberation-channel non-equivalence: always-on or non-disablable reasoning channel makes the continuation regime incomparable",
        },
        "python": sys.version,
    }
    C.write_json(os.path.join(C.RESULTS_DIR, "run_metadata.json"), meta)
    return meta


def _fmt(x):
    return "n.a." if x is None else ("%.3f" % x)


def cmd_report(args):
    write_metadata()
    tables = json.load(open(os.path.join(C.RESULTS_DIR, "summary_tables.json")))
    led_cost = tables["cost"]
    lines = ["# EXPH: API-model generality arm (descriptive / exploratory)", "",
             "Generated: %s" % now_iso(), "",
             "**Scope**: NOT confirmatory. No pre-registered hypotheses are scored here; "
             "rows are descriptive robustness checks of the paper's core phenomena on "
             "API models under an instruction-format continuation prompt.", ""]

    # bridge
    lines += ["## Bridge arm (Qwen2.5-7B-Instruct, instruct format, local vLLM)", ""]
    gate = tables["bridge"].get("bridge_gate", {})
    bl = tables["bridge"].get("bridge_legacy_summary", {})
    if bl:
        lines += ["Legacy EXPA rows (locked manifest, mid, n=100/cell; only the prompt format changed):", "",
                  "| cell | n | valid | deriv inj-dep | echo | stated-comp | doubt | unparsed | parse |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cond in C.CORE_CONDITIONS:
            c = bl.get(cond)
            if not c:
                continue
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                cond, c["n"], _fmt(c["valid_recovery"]["rate"]),
                _fmt(c["inj_derivational"]["rate"]), _fmt(c["inj_echo"]["rate"]),
                _fmt(c["stated_complement"]["rate"]), _fmt(c["doubt"]["rate"]),
                _fmt(c["unparsed"]["rate"]), _fmt(c["parse_rate"])))
        lines += ["", "Locked EXPA mid anchors (HF prefill): valid benign 0.880 / one-hop 0.640 / "
                       "global 0.360; poisoned global 0.373, one-hop 0.020; doubt one-hop 0.273.", ""]
    bw = tables["bridge"].get("bridge_worlds_summary", {})
    if bw:
        lines += ["Usable/inert pair on EXPD-style generated worlds (own Qwen traces, d=1, mid):", "",
                  "| cell | n | valid | deriv inj-dep | echo | stated-comp | doubt | parse |",
                  "|---|---|---|---|---|---|---|---|"]
        for cond in C.REUSE_CONDITIONS:
            c = bw.get(cond)
            if not c:
                continue
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
                cond, c["n"], _fmt(c["valid_recovery"]["rate"]),
                _fmt(c["inj_derivational"]["rate"]), _fmt(c["inj_echo"]["rate"]),
                _fmt(c["stated_complement"]["rate"]), _fmt(c["doubt"]["rate"]),
                _fmt(c["parse_rate"])))
        lines.append("")
    if gate:
        lines += ["Gate verdicts: legacy pass=%s, worlds pass=%s" % (
            gate.get("legacy", {}).get("pass"), gate.get("worlds", {}).get("pass")),
            "", "```json", json.dumps({k: gate.get(k) for k in ("legacy", "worlds")}, indent=2),
            "```", ""]

    # per-model tables
    lines += ["## API models (own-trace golds; mid injections)", ""]
    gf = tables.get("gold_funnels", {})
    if gf:
        lines += ["Gold funnels (double filter = solved AND validator-valid AND >=3 intermediate entity steps):", "",
                  "| model (gold) | attempts | solve rate | eligible | world eligible |", "|---|---|---|---|---|"]
        for k, f in gf.items():
            lines.append("| %s | %d | %.3f | %d | %d |" % (
                k, f["attempts"], f["solve_rate"], f["eligible"], f["world_eligible"]))
        lines.append("")
    for key, per in tables.get("models", {}).items():
        lines += ["### %s" % key, "",
                  "| cell | n | valid | deriv inj-dep | echo | stated-comp | doubt | unparsed | parse |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cond in list(C.CORE_CONDITIONS) + list(C.REUSE_CONDITIONS):
            c = per.get(cond)
            if not c:
                continue
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                cond, c["n"], _fmt(c["valid_recovery"]["rate"]),
                _fmt(c["inj_derivational"]["rate"]), _fmt(c["inj_echo"]["rate"]),
                _fmt(c["stated_complement"]["rate"]), _fmt(c["doubt"]["rate"]),
                _fmt(c["unparsed"]["rate"]), _fmt(c["parse_rate"])))
        lines.append("")

    # haiku prefill-vs-instruct delta
    hk_p = tables.get("models", {}).get("claude-haiku-4-5_prefill", {})
    hk_i = tables.get("models", {}).get("claude-haiku-4-5_instruct", {})
    if hk_p and hk_i:
        lines += ["## Prefill vs instruct (claude-haiku-4-5; SAME gold traces, same injections)", "",
                  "| cell | metric | prefill | instruct | delta (instruct - prefill) |", "|---|---|---|---|---|"]
        for cond in list(C.CORE_CONDITIONS) + list(C.REUSE_CONDITIONS):
            if cond not in hk_p or cond not in hk_i:
                continue
            for met in ("valid_recovery", "inj_derivational", "doubt", "stated_complement"):
                a = hk_p[cond][met]["rate"]
                b = hk_i[cond][met]["rate"]
                if a is None or b is None:
                    continue
                lines.append("| %s | %s | %.3f | %.3f | %+.3f |" % (cond, met, a, b, b - a))
        lines.append("")

    # cost ledger
    lines += ["## Cost ledger", "",
              "Total: $%.2f (hard stop $%.0f; target ~$25-60)" % (led_cost["total_usd"], A.HARD_BUDGET_USD), "",
              "| model | requests | input tok | output tok | USD |", "|---|---|---|---|---|"]
    led = ledger()
    for mid, m in sorted(led.data["models"].items()):
        lines.append("| %s | %d | %d | %d | $%.2f |" % (
            mid, m["requests"], m["input_tokens"], m["output_tokens"],
            led.data.get("usd_by_model", {}).get(mid, 0)))
    lines.append("")

    # disclosures
    lines += [
        "## Disclosures", "",
        "- **Instruction-format deviation**: legacy locked results use in-stream assistant "
        "continuation (HF/vLLM prefill). API chat models cannot reproduce that byte stream; "
        "mode `instruct` presents the proof-so-far inside the user turn plus an explicit "
        "continuation instruction; mode `prefill` (Anthropic haiku-4-5 / sonnet-4-5) uses an "
        "assistant-turn prefill, structurally closest to the legacy regime but still a chat-turn "
        "boundary rather than one token stream. The Qwen bridge arm quantifies the pure "
        "format shift on the same model/rows as the locked EXPA cells.",
        "- **Own-trace golds**: every API model is perturbed on its own gold rollouts "
        "(EXPA standard); condition availability and injected statements therefore vary by "
        "model. All planted claims are audited (global/one-hop/reuse audited FALSE; paraphrase/"
        "interruption audited TRUE); reuse cells additionally assert measured d==1 from the prefix.",
        "- **One-hop d-smearing**: as in the locked data, `one_hop_falsehood` is negation of the "
        "model's own mid step; measured refutation distance is recorded per row "
        "(`measured_local_d`) but not re-binned here (stage-0 audit found ~77%% of legacy "
        "one-hop at d==1).",
        "- **Doubt is a lexical regex** on the continuation; absolute doubt levels are known to be "
        "decoding/backend sensitive (replay_full) and API chat models differ further in verbosity; "
        "read doubt as within-model contrasts only.",
        "- **Sampling**: temperature=0 where the API allows it (haiku-4-5, sonnet-4-5, gpt-4.1); "
        "claude-sonnet-5 (thinking disabled) and claude-opus-4-8 do not accept temperature "
        "(API rejects sampling params); gpt-5.1 run with reasoning_effort='none'. Greedy "
        "determinism is therefore NOT guaranteed on those models; rows are one-sample "
        "descriptive estimates.",
        "- **max_tokens=300** per continuation (legacy used 192 HF tokens); truncated "
        "continuations surface as derailed/unparsed and are counted, not dropped.",
        "- **Post-processing**: continuations are stripped of a leading 'A:' and truncated at a "
        "regenerated 'Q:' block before validation (flagged per row); no other edits.",
        "- **Excluded models**: Claude Fable 5, OpenAI o-series, DeepSeek-R1 -- always-on / "
        "non-disablable deliberation channels make the interrupted-continuation regime "
        "non-equivalent to the paper's setting.",
        "- Cluster bootstrap CIs cluster on problem (legacy) / family (worlds).",
        "",
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(C.RESULTS_DIR, "EXPH_REPORT.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(C.RESULTS_DIR, "README.md"), "w") as fh:
        fh.write(text.replace("# EXPH:", "# EXPH_API_MODELS README:", 1))
    print("EXPH_REPORT.md written (%d lines)" % len(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "gold", "perturb", "validate",
                                      "summarize", "report", "meta"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    os.makedirs(API_DIR, exist_ok=True)
    if args.stage == "smoke":
        cmd_smoke(args)
    elif args.stage == "gold":
        cmd_gold(args)
    elif args.stage == "perturb":
        cmd_perturb(args)
    elif args.stage == "validate":
        cmd_validate(args)
    elif args.stage == "summarize":
        cmd_summarize(args)
    elif args.stage == "report":
        cmd_report(args)
    elif args.stage == "meta":
        write_metadata()


if __name__ == "__main__":
    main()
