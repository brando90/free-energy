"""EXPH bridge arm: Qwen2.5-7B-Instruct (pinned) under mode "instruct" on
(1) the 4 legacy EXPA core conditions at mid, n=100/cell, reusing the LOCKED
    EXPA manifest rows verbatim (same questions, same gold prefixes, same
    injected statements) -- the ONLY change is prompt format; and
(2) the usable/inert categorical pair on EXPD-style generated worlds
    (fresh cat-only pool, d=1, n<=60/cell) with the model's own gold traces.

Pre-stated gates (evaluated here, written to bridge/bridge_gate.json):
  legacy : valid(global) < valid(one_hop) < valid(benign);
           derivational inj-dep(global) >= 0.15; inj-dep(one_hop) <= 0.10;
           parse rate >= 0.85 per cell.
  worlds : derivational reuse usable - inert >= 0.30.

Run inside tooling/.venv-vllm with tooling/env.sh sourced and one idle GPU.
"""
import argparse
import datetime as _dt
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exph_common as C  # noqa: E402  (also inserts src/ into sys.path)

EXP = C.EXP
TOOLING = os.path.join(EXP, "improvement_plan", "tooling")
EXPD_DIR = os.path.join(EXP, "improvement_plan", "expd")
for p in (TOOLING, EXPD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import vllm_gen  # noqa: E402

QWEN = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_NEW = 256
N_LEGACY_PER_CELL = 100
N_WORLDS_PER_CELL = 60
BRIDGE_DIR = os.path.join(C.RESULTS_DIR, os.environ.get("EXPH_BRIDGE_SUBDIR", "bridge"))
WORLDS_DIR = os.path.join(C.RESULTS_DIR, "worlds")
EXPA_DIR = os.path.join(EXP, "results", "EXPA_GLOBAL_EXPANSION")

WORLD_CONFIG = {"name": "exph_cat75", "attr": [], "cat": [{"families": 75, "d_levels": [1]}]}


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def qwen_tok():
    return vllm_gen.get_tokenizer(QWEN, QWEN_REV)


def chat_ids(tok, user_content):
    return vllm_gen._chat_ids(tok, [{"role": "user", "content": user_content}])


# --------------------------------------------------------------- legacy arm

def load_legacy_rows():
    """Locked EXPA mid rows; first N_LEGACY_PER_CELL problems (manifest order)
    that carry all four core conditions at mid."""
    rows = [r for r in C.read_jsonl(os.path.join(EXPA_DIR, "manifest.jsonl"))
            if r["injection_position"] == "mid"]
    by_problem = {}
    order = []
    for r in rows:
        if r["problem_id"] not in by_problem:
            order.append(r["problem_id"])
        by_problem.setdefault(r["problem_id"], {})[r["condition"]] = r
    picked = [pid for pid in order
              if all(c in by_problem[pid] for c in C.CORE_CONDITIONS)][:N_LEGACY_PER_CELL]
    out = []
    for pid in picked:
        for cond in C.CORE_CONDITIONS:
            out.append(by_problem[pid][cond])
    return picked, out


def run_legacy():
    os.makedirs(BRIDGE_DIR, exist_ok=True)
    picked, expa_rows = load_legacy_rows()
    print("[bridge/legacy] problems=%d rows=%d" % (len(picked), len(expa_rows)), flush=True)
    tok = qwen_tok()
    jobs, mrows = [], []
    for r in expa_rows:
        pre = C.prefix_text(r["prefix_steps"], r["injected_statement"])
        user = C.instruct_user_content(r["question"], r["target"], pre)
        jobs.append({"prompt_token_ids": chat_ids(tok, user)})
        mrows.append({
            "run_id": C.sha_row(["EXPH_BRIDGE_LEGACY", r["run_id"]]),
            "expa_run_id": r["run_id"],
            "model_key": "qwen2.5-7b_bridge_instruct", "provider": "local_vllm",
            "model_id": QWEN, "mode": "instruct", "arm": "core",
            "condition": r["condition"], "problem_id": r["problem_id"],
            "cluster_id": r["problem_id"], "injection_position": "mid",
            "seed": 0, "sent_idx": r["sent_idx"],
            "question": r["question"], "target": r["target"], "entity": r["entity"],
            "prefix_steps": r["prefix_steps"],
            "injected_statement": r["injected_statement"],
            "audited_truth_status": r["audited_truth_status"],
            "measured_local_d": None,
            "poisoning_measurable": r["condition"] == "global_falsehood",
            "original_proof_validated": True,
        })
    outs = vllm_gen.generate_continuations(QWEN, jobs, max_new_tokens=MAX_NEW, revision=QWEN_REV)
    raw_rows, val_rows = [], []
    for m, o in zip(mrows, outs):
        raw = {"run_id": m["run_id"], "condition": m["condition"], "problem_id": m["problem_id"],
               "failed_generation": o["text"] is None, "continuation": o["text"],
               "finish_reason": o.get("finish_reason"), "created_at": now_iso()}
        raw_rows.append(raw)
        val_rows.append(C.validate_row(m, raw))
    C.write_jsonl(os.path.join(BRIDGE_DIR, "bridge_legacy_raw.jsonl"), raw_rows)
    C.write_jsonl(os.path.join(BRIDGE_DIR, "bridge_legacy_manifest.jsonl"), mrows)
    C.write_jsonl(os.path.join(BRIDGE_DIR, "bridge_legacy_validated.jsonl"), val_rows)
    summary = {}
    for cond in C.CORE_CONDITIONS:
        summary[cond] = C.summarize_cell([v for v in val_rows if v["condition"] == cond])
    C.write_json(os.path.join(BRIDGE_DIR, "bridge_legacy_summary.json"), summary)
    return summary


def legacy_gate(summary):
    v = {c: summary[c]["valid_recovery"]["rate"] for c in C.CORE_CONDITIONS}
    d = {c: summary[c]["inj_derivational"]["rate"] for c in C.CORE_CONDITIONS}
    p = {c: summary[c]["parse_rate"] for c in C.CORE_CONDITIONS}
    checks = {
        "ordering_valid_global_lt_onehop": v["global_falsehood"] < v["one_hop_falsehood"],
        "ordering_valid_onehop_lt_benign": v["one_hop_falsehood"] < v["benign_paraphrase"],
        "injdep_global_ge_0.15": d["global_falsehood"] >= 0.15,
        "injdep_onehop_le_0.10": d["one_hop_falsehood"] <= 0.10,
        "parse_rate_ge_0.85_all_cells": all((x or 0) >= 0.85 for x in p.values()),
    }
    return {"values": {"valid": v, "inj_derivational": d, "parse_rate": p},
            "checks": checks, "pass": all(checks.values())}


# --------------------------------------------------------------- worlds arm

def ensure_worlds():
    manifest = os.path.join(WORLDS_DIR, "world_manifest.jsonl")
    if os.path.exists(manifest):
        return C.read_jsonl(manifest)
    import gen_worlds_expd as genw
    genw.CONFIGS["exph_cat75"] = WORLD_CONFIG
    doc = genw.generate_and_write(WORLDS_DIR, 0, "exph_cat75")
    print("[bridge/worlds] generated worlds=%d audit_failures=%d"
          % (doc["world_count"], len(doc["audit_failures"])), flush=True)
    return C.read_jsonl(manifest)


def qwen_world_golds(worlds):
    """Instruct-format golds (plain question->proof) for every world; cached."""
    path = os.path.join(BRIDGE_DIR, "qwen_worlds_gold_raw.jsonl")
    if os.path.exists(path):
        return {r["world_id"]: r for r in C.read_jsonl(path)}
    tok = qwen_tok()
    jobs = [{"prompt_token_ids": chat_ids(tok, C.gold_user_content(w["question"], w["target"]))}
            for w in worlds]
    outs = vllm_gen.generate_continuations(QWEN, jobs, max_new_tokens=MAX_NEW, revision=QWEN_REV)
    rows = []
    for w, o in zip(worlds, outs):
        rows.append({"world_id": w["world_id"], "family_id": w["family_id"],
                     "version": w["version"], "condition": "gold",
                     "failed_generation": o["text"] is None,
                     "continuation": o["text"], "finish_reason": o.get("finish_reason"),
                     "created_at": now_iso()})
    C.write_jsonl(path, rows)
    return {r["world_id"]: r for r in rows}


def build_world_cohort(worlds, golds, out_path):
    cohort = []
    for w in worlds:
        g = golds.get(w["world_id"])
        rec = {"world_id": w["world_id"], "family_id": w["family_id"],
               "family_idx": w["family_idx"], "version": w["version"],
               "entity": w["entity"], "target": w["target"]}
        if g is None or g.get("failed_generation"):
            rec.update({"eligible": False, "reason": "gold_missing_or_failed"})
            cohort.append(rec)
            continue
        text = C.clean_continuation(g.get("continuation", ""))
        steps = C.split_sentences(text)
        is_solved = C.solved(text, w["target"])
        from validator import validate_continuation
        v = validate_continuation(w["question"], [], None, text, w["target"], w["entity"])
        pts = C.expc_injection_points(steps, w["entity"])
        eligible = is_solved and v["class"] == "valid_rederivation" and pts is not None
        rec.update({"solved": is_solved, "gold_class": v["class"], "gold_steps": steps,
                    "eligible": eligible,
                    "reason": ("eligible" if eligible else
                               "gold_not_solved" if not is_solved else
                               "gold_not_validator_valid" if v["class"] != "valid_rederivation"
                               else "too_few_intermediate_entity_steps")})
        cohort.append(rec)
    C.write_jsonl(out_path, cohort)
    return cohort


def run_worlds():
    os.makedirs(BRIDGE_DIR, exist_ok=True)
    worlds = ensure_worlds()
    by_id = {w["world_id"]: w for w in worlds}
    golds = qwen_world_golds(worlds)
    cohort = build_world_cohort(worlds, golds,
                                os.path.join(BRIDGE_DIR, "qwen_worlds_gold_cohort.jsonl"))
    n_eligible = sum(1 for c in cohort if c.get("eligible"))
    print("[bridge/worlds] eligible=%d/%d" % (n_eligible, len(cohort)), flush=True)

    tok = qwen_tok()
    jobs, mrows, rejects = [], [], []
    for version in ("usable", "inert"):
        taken = 0
        for rec in sorted([c for c in cohort if c["version"] == version and c.get("eligible")],
                          key=lambda x: x["family_idx"]):
            if taken >= N_WORLDS_PER_CELL:
                break
            w = by_id[rec["world_id"]]
            inj, err = C.build_reuse_injection(w, rec["gold_steps"], seed=0)
            if inj is None:
                rejects.append({"world_id": w["world_id"], **err})
                continue
            si = inj["sent_idx"]
            pre = C.prefix_text(rec["gold_steps"][:si], inj["injected_statement"])
            user = C.instruct_user_content(w["question"], w["target"], pre)
            jobs.append({"prompt_token_ids": chat_ids(tok, user)})
            mrows.append({
                "run_id": C.sha_row(["EXPH_BRIDGE_WORLDS", w["world_id"], inj["condition"]]),
                "model_key": "qwen2.5-7b_bridge_instruct", "provider": "local_vllm",
                "model_id": QWEN, "mode": "instruct", "arm": "reuse",
                "condition": inj["condition"], "problem_id": w["world_id"],
                "cluster_id": w["family_id"], "injection_position": "mid",
                "seed": 0, "sent_idx": si,
                "question": w["question"], "target": w["target"], "entity": w["entity"],
                "prefix_steps": rec["gold_steps"][:si],
                "injected_statement": inj["injected_statement"],
                "audited_truth_status": "false", "measured_local_d": 1,
                "poisoning_measurable": version == "usable",
                "original_proof_validated": True,
            })
            taken += 1
    outs = vllm_gen.generate_continuations(QWEN, jobs, max_new_tokens=MAX_NEW, revision=QWEN_REV)
    raw_rows, val_rows = [], []
    for m, o in zip(mrows, outs):
        raw = {"run_id": m["run_id"], "condition": m["condition"], "problem_id": m["problem_id"],
               "failed_generation": o["text"] is None, "continuation": o["text"],
               "finish_reason": o.get("finish_reason"), "created_at": now_iso()}
        raw_rows.append(raw)
        val_rows.append(C.validate_row(m, raw))
    C.write_jsonl(os.path.join(BRIDGE_DIR, "bridge_worlds_raw.jsonl"), raw_rows)
    C.write_jsonl(os.path.join(BRIDGE_DIR, "bridge_worlds_manifest.jsonl"), mrows)
    C.write_jsonl(os.path.join(BRIDGE_DIR, "bridge_worlds_validated.jsonl"), val_rows)
    C.write_jsonl(os.path.join(BRIDGE_DIR, "bridge_worlds_rejects.jsonl"), rejects)
    summary = {}
    for cond in C.REUSE_CONDITIONS:
        summary[cond] = C.summarize_cell([v for v in val_rows if v["condition"] == cond])
    C.write_json(os.path.join(BRIDGE_DIR, "bridge_worlds_summary.json"), summary)
    return summary


def worlds_gate(summary):
    du = summary["cat_false_usable_d1"]["inj_derivational"]["rate"] or 0
    di = summary["cat_false_inert_d1"]["inj_derivational"]["rate"] or 0
    checks = {"derivational_reuse_gap_ge_0.30": (du - di) >= 0.30}
    return {"values": {"usable_derivational": du, "inert_derivational": di,
                       "gap": round(du - di, 4)},
            "checks": checks, "pass": all(checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["legacy", "worlds", "all"])
    args = ap.parse_args()
    gate_path = os.path.join(BRIDGE_DIR, "bridge_gate.json")
    gate = json.load(open(gate_path)) if os.path.exists(gate_path) else {}
    if args.stage in ("legacy", "all"):
        s = run_legacy()
        gate["legacy"] = legacy_gate(s)
        gate["legacy"]["evaluated_at"] = now_iso()
        print(json.dumps(gate["legacy"], indent=2))
    if args.stage in ("worlds", "all"):
        s = run_worlds()
        gate["worlds"] = worlds_gate(s)
        gate["worlds"]["evaluated_at"] = now_iso()
        print(json.dumps(gate["worlds"], indent=2))
    gate["qwen_model"] = QWEN
    gate["qwen_revision"] = QWEN_REV
    gate["prompt_sha256"] = C.prompt_sha()
    C.write_json(gate_path, gate)
    print("GATE_SUMMARY " + json.dumps({k: gate[k]["pass"] for k in ("legacy", "worlds")
                                        if k in gate and isinstance(gate[k], dict) and "pass" in gate[k]}))


if __name__ == "__main__":
    main()
