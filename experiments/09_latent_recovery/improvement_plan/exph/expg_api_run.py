"""EXPH2 Experiment A: second-regime (EXPG program-trace) frontier check.

Prefill mode, Anthropic only: claude-haiku-4-5 + claude-sonnet-4-5 on the EXPG
execution-trace regime, reusing the EXPG generator / trace format / injector /
two-world validator verbatim (improvement_plan/expg/). Own-trace golds per model
(pilot Stage-0 final knobs; SAME seed bases as the pilot so kr8/kc1/kc5 pools
are supersets/equal of the pilot pools), 7 pilot cells x mid x n<=75.

Descriptive rows only. Question: does frontier rejection generalize to the
execution regime (esp. opfree cells where Qwen next-read absorption was 1.00)?

Usage: expg_api_run.py {prepare,parity,gold,perturb,validate,summarize} [--model KEY]
"""
import argparse
import datetime as _dt
import json
import os
import sys
from collections import Counter

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXPG = os.path.abspath(os.path.join(HERE, "..", "expg"))
for p in (HERE, EXPG):
    if p not in sys.path:
        sys.path.insert(0, p)

import exph_common as C  # noqa: E402  (io helpers + EXP path; PrOntoQA bits unused here)
import api_gen as A  # noqa: E402
import gen_programs as gp  # noqa: E402
import inject as inj_mod  # noqa: E402
import trace_format as tf  # noqa: E402
import validate as val_mod  # noqa: E402
from interp import parse_program  # noqa: E402

EXP = C.EXP
R2_DIR = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS", "regime2")
EXPH_LEDGER = os.path.join(EXP, "results", "EXPH_API_MODELS", "cost_ledger.json")
SEED = 0
MAX_GOLD = 512
MAX_CONT = 384
CELL_CAP = 75

# Pilot constants (expg_progtrace.py / EXPG_PROGTRACE_PILOT run_metadata)
CELL_SHAPE = {"benign_paraphrase": "kr1", "true_interruption": "kr1",
              "adjacent_contradiction": "kr1", "opfree_kr1": "kr1",
              "opfree_kr8": "kr8", "onehop_kc1": "kc1", "deep_kc5": "kc5"}
CELLS = ("benign_paraphrase", "true_interruption", "adjacent_contradiction",
         "opfree_kr1", "opfree_kr8", "onehop_kc1", "deep_kc5")
SHAPE_SEED_BASE = {"kr1": 100000, "kr8": 200000, "kc1": 300000, "kc5": 400000}
PILOT_KNOBS = {"const_max": 45, "const_min": 2, "deep_const_max": 9,
               "filler_const_prob": 0.35, "forward_use_gap": 2, "loop_prob": 0.0,
               "max_attempts_per_program": 400, "mult_prob": 0.15,
               "read_const_max": 20, "value_max": 999, "value_min": 1}
# gold attempts cap 250 total; kr1 hosts 4 cells -> weighted allocation
SHAPE_ALLOC = {"kr1": 70, "kr8": 60, "kc1": 60, "kc5": 60}

MODELS = [
    {"key": "claude-haiku-4-5_prefill_r2", "provider": "anthropic",
     "model_id": "claude-haiku-4-5", "mode": "prefill", "temperature": 0.0},
    {"key": "claude-sonnet-4-5_prefill_r2", "provider": "anthropic",
     "model_id": "claude-sonnet-4-5", "mode": "prefill", "temperature": 0.0},
]


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def ledger():
    extras = [EXPH_LEDGER]
    base = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS")
    for sub in ("ladder", "instruct_panel", "completions_probe"):
        extras.append(os.path.join(base, sub, "cost_ledger.json"))
    return A.CostLedger(os.path.join(R2_DIR, "cost_ledger.json"), extra_paths=extras)


def get_cfg(key):
    for m in MODELS:
        if m["key"] == key:
            return dict(m)
    raise KeyError(key)


def prog_path():
    return os.path.join(R2_DIR, "programs.jsonl")


def cmd_prepare(_args):
    os.makedirs(R2_DIR, exist_ok=True)
    if os.path.exists(prog_path()):
        print("programs.jsonl exists (%d rows)" % len(C.read_jsonl(prog_path())))
        return
    funnels = {}
    for shape in gp.SHAPES:
        progs, funnel = gp.generate_shape_programs(shape, SHAPE_ALLOC[shape],
                                                   SHAPE_SEED_BASE[shape], PILOT_KNOBS)
        funnels[shape] = {"attempts": funnel["attempts"], "accepted": len(progs)}
        for p in progs:
            row = dict(p)
            row["listing_text"] = tf.make_listing_text(p["stmt_texts"])
            C.append_jsonl(prog_path(), row)
    C.write_json(os.path.join(R2_DIR, "generator_funnels.json"),
                 {"funnels": funnels, "knobs": PILOT_KNOBS,
                  "seed_bases": SHAPE_SEED_BASE, "alloc": SHAPE_ALLOC,
                  "note": "pilot knobs + pilot seed bases (kr8/kc1/kc5 pools == pilot pools)"})
    print(json.dumps(funnels, indent=1))


def gold_rid(cfg, program_id):
    return C.sha_row(["EXPH2_R2", cfg["model_id"], "gold", program_id, SEED])


def pert_rid(cfg, program_id, cell):
    return C.sha_row(["EXPH2_R2", cfg["model_id"], cell, program_id, SEED])


def gold_paths(cfg):
    return (os.path.join(R2_DIR, "raw_gold_%s.jsonl" % cfg["model_id"]),
            os.path.join(R2_DIR, "cohort_%s.json" % cfg["model_id"]))


def api_jobs(cfg, jobs, led, raw_path, max_tokens):
    """jobs: (base_row, user_msg, prefill). Prefill trailing-newline handling:
    try exact parity; Anthropic rejects trailing whitespace in the final
    assistant turn, so a trailing '\\n' is stripped and flagged per row."""
    cfg = dict(cfg)
    cfg["max_tokens"] = max_tokens
    existing = {r["run_id"] for r in C.read_jsonl(raw_path)}
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    pending = [j for j in jobs if j[0]["run_id"] not in existing]

    def work(job):
        base, user_msg, prefill = job
        stripped = prefill.endswith("\n") if prefill else False
        pf = prefill.rstrip("\n") if prefill else prefill
        messages = [{"role": "user", "content": user_msg}]
        if pf is not None:
            messages.append({"role": "assistant", "content": pf})
        out = A.call_model(cfg, messages, led)
        rec = dict(base)
        rec.update({"failed_generation": out.get("text") is None,
                    "continuation": out.get("text"), "error": out.get("error"),
                    "stop_reason": out.get("stop_reason"),
                    "prefill_trailing_newline_stripped": stripped,
                    "input_tokens": out.get("input_tokens"),
                    "output_tokens": out.get("output_tokens"),
                    "created_at": now_iso()})
        with lock:
            C.append_jsonl(raw_path, rec)
        return rec

    with ThreadPoolExecutor(max_workers=A.CONCURRENCY) as ex:
        list(ex.map(work, pending))
    led.check_budget()
    led.save()
    return len(pending)


def cmd_gold(args):
    cfg = get_cfg(args.model)
    led = ledger()
    programs = C.read_jsonl(prog_path())
    raw_path, coh_path = gold_paths(cfg)
    jobs = []
    for p in programs:
        base = {"run_id": gold_rid(cfg, p["program_id"]), "program_id": p["program_id"],
                "condition": "gold", "shape": p["shape"], "model_id": cfg["model_id"]}
        jobs.append((base, tf.make_user_message(p["listing_text"]), tf.GOLD_PREFILL))
    n = api_jobs(cfg, jobs, led, raw_path, MAX_GOLD)
    raw = {r["program_id"]: r for r in C.read_jsonl(raw_path)}
    per_prog = []
    for p in programs:
        g = raw.get(p["program_id"])
        if g is None or g.get("failed_generation"):
            per_prog.append({"program_id": p["program_id"], "shape": p["shape"],
                             "solved": False, "eligible": False,
                             "reason": "gold_missing_or_failed"})
            continue
        ev = inj_mod.gold_solve_eval(p, tf.GOLD_PREFILL + g["continuation"])
        per_prog.append({"program_id": p["program_id"], "shape": p["shape"],
                         "solved": ev["solved"], "parse_ok": ev["parse_ok"],
                         "eligible": ev["solved"],
                         "reason": "eligible" if ev["solved"]
                         else (ev["first_error"] or {}).get("type", "unknown")})
    summary = {"n_programs": len(per_prog), "new_generations": n,
               "solve_rate": round(sum(r["solved"] for r in per_prog) / len(per_prog), 4),
               "parse_rate": round(sum(bool(r.get("parse_ok")) for r in per_prog) / len(per_prog), 4),
               "failure_taxonomy": dict(Counter(r["reason"] for r in per_prog if not r["solved"])),
               "by_shape": {}}
    for shape in gp.SHAPES:
        rows = [r for r in per_prog if r["shape"] == shape]
        summary["by_shape"][shape] = {
            "n": len(rows),
            "solve_rate": round(sum(r["solved"] for r in rows) / len(rows), 4),
            "eligible": sum(r["eligible"] for r in rows)}
    C.write_json(coh_path, {"per_program": per_prog, "summary": summary})
    print(json.dumps({"model": cfg["model_id"], "summary": summary,
                      "cumulative_usd": round(led.cost_usd() + led.extra_cost_usd(), 2)}))


def cmd_perturb(args):
    cfg = get_cfg(args.model)
    led = ledger()
    programs = {p["program_id"]: p for p in C.read_jsonl(prog_path())}
    raw_gold_path, coh_path = gold_paths(cfg)
    cohort = json.load(open(coh_path))
    gold_raw = {r["program_id"]: r for r in C.read_jsonl(raw_gold_path)}
    elig = {s: [] for s in gp.SHAPES}
    for r in cohort["per_program"]:
        if r["eligible"]:
            elig[r["shape"]].append(r["program_id"])
    man_path = os.path.join(R2_DIR, "manifest_%s.jsonl" % cfg["model_id"])
    audit_path = os.path.join(R2_DIR, "audit_%s.jsonl" % cfg["model_id"])
    mrows = C.read_jsonl(man_path)
    if not mrows:
        for cell in CELLS:
            shape = CELL_SHAPE[cell]
            for pid in elig[shape][:CELL_CAP]:
                p = programs[pid]
                gold_full = tf.GOLD_PREFILL + gold_raw[pid]["continuation"]
                try:
                    b = inj_mod.build_injection(p, cell, gold_full)
                except inj_mod.InjectError as e:
                    C.append_jsonl(audit_path, {"stage": "injection_audit", "cell": cell,
                                                "program_id": pid, "rejected": True,
                                                "reason": e.reason})
                    continue
                row = {"run_id": pert_rid(cfg, pid, cell), "cell": cell,
                       "condition": cell, "program_id": pid, "shape": shape,
                       "position": "mid", "L": p["L"], "site_line": p["site_line"],
                       "site_var": p["site_var"], "k_r": p["k_r"], "k_c": p["k_c"],
                       "r1": p["r1"], "r1_var": p["r1_var"],
                       "model_id": cfg["model_id"], "model_key": cfg["key"],
                       "mode": "prefill", "seed": SEED, "created_at": now_iso()}
                row.update({k: b[k] for k in
                            ("family", "injected_trace_line", "original_trace_line",
                             "prefix_text", "step_replaced", "expected_lines_from",
                             "cf", "planted_var", "planted_value", "true_value",
                             "force_after_line", "delta_policy")})
                mrows.append(row)
        C.write_jsonl(man_path, mrows)
    raw_path = os.path.join(R2_DIR, "raw_pert_%s.jsonl" % cfg["model_id"])
    jobs = []
    for m in mrows:
        p = programs[m["program_id"]]
        base = {"run_id": m["run_id"], "program_id": m["program_id"],
                "condition": m["cell"], "shape": m["shape"], "model_id": cfg["model_id"]}
        jobs.append((base, tf.make_user_message(p["listing_text"]), m["prefix_text"]))
    n = api_jobs(cfg, jobs, led, raw_path, MAX_CONT)
    print("[r2 perturb %s] manifest=%d new=%d cells=%s cum=$%.2f"
          % (cfg["model_id"], len(mrows), n,
             json.dumps(dict(Counter(m["cell"] for m in mrows))),
             led.cost_usd() + led.extra_cost_usd()))


def cmd_validate(args):
    cfg = get_cfg(args.model)
    programs = {p["program_id"]: p for p in C.read_jsonl(prog_path())}
    manifest = C.read_jsonl(os.path.join(R2_DIR, "manifest_%s.jsonl" % cfg["model_id"]))
    raw = {r["run_id"]: r for r in
           C.read_jsonl(os.path.join(R2_DIR, "raw_pert_%s.jsonl" % cfg["model_id"]))}
    out_path = os.path.join(R2_DIR, "validated_%s.jsonl" % cfg["model_id"])
    rows = []
    for m in manifest:
        p = programs[m["program_id"]]
        r = raw.get(m["run_id"])
        rec = dict(m)
        if r is None or r.get("failed_generation"):
            rec.update({"generation_failed": True, "class": "generation_failed"})
            rows.append(rec)
            continue
        inj = {k: m[k] for k in ("cf", "planted_var", "planted_value",
                                 "force_after_line", "expected_lines_from")}
        metrics = val_mod.classify_run(parse_program(p["stmt_texts"]), p, inj,
                                       r["continuation"])
        rec.update(metrics)
        rec["generation_failed"] = False
        rec["continuation"] = r["continuation"]
        rec["first_char_digit"] = bool((r["continuation"] or "").lstrip("\n")[:1].isdigit()) \
            if (r["continuation"] or "").strip() else False
        rows.append(rec)
    C.write_jsonl(out_path, rows)
    print("[r2 validate %s] rows=%d" % (cfg["model_id"], len(rows)))


R2_METRICS = ("trace_valid", "final_output_valid", "next_read_absorbed",
              "final_output_absorbed", "repair_event", "injection_dependent",
              "doubt_lex", "doubt_broad", "unparsed", "generation_failed",
              "first_char_digit")


def _r2val(r, metric):
    if metric == "generation_failed":
        return 1 if r.get("generation_failed") else 0
    if r.get("generation_failed"):
        return 0  # house convention: failed runs stay in denominators
    v = r.get(metric)
    if v is None:
        return None  # structurally not measurable (non-cf cells)
    return int(bool(v))


def summarize_r2_cell(rows, seed=0):
    out = {"n": len(rows)}
    for metric in R2_METRICS:
        vals = [(_r2val(r, metric), r) for r in rows]
        meas = [(v, r) for v, r in vals if v is not None]
        if not meas:
            out[metric] = {"rate": None, "n_measured": 0}
            continue
        k = sum(v for v, _ in meas)
        n = len(meas)
        sub_rows = [dict(r, _mv=v) for v, r in meas]
        out[metric] = {
            "count": k, "n_measured": n, "rate": round(k / n, 4),
            "wilson95": C.wilson(k, n),
            "program_cluster_bootstrap95": C.cluster_boot_rate(
                sub_rows, lambda r: r["_mv"], seed=seed, cluster_key="program_id"),
        }
    skips = [r.get("line_skip_rate") for r in rows
             if not r.get("generation_failed") and r.get("line_skip_rate") is not None]
    out["mean_line_skip_rate"] = round(sum(skips) / len(skips), 4) if skips else None
    return out


def cmd_summarize(_args):
    out = {"created_at": now_iso(), "models": {}, "cells": list(CELLS),
           "qwen_pilot_reference": "results/EXPG_PROGTRACE_PILOT/REPORT.md (n<=35/cell)"}
    for m in MODELS:
        p = os.path.join(R2_DIR, "validated_%s.jsonl" % m["model_id"])
        if not os.path.exists(p):
            continue
        rows = C.read_jsonl(p)
        per = {}
        for cell in CELLS:
            sub = [r for r in rows if r["condition"] == cell]
            if sub:
                per[cell] = summarize_r2_cell(sub, seed=SEED)
        out["models"][m["model_id"]] = per
        cp = os.path.join(R2_DIR, "cohort_%s.json" % m["model_id"])
        if os.path.exists(cp):
            out.setdefault("gold_funnels", {})[m["model_id"]] = json.load(open(cp))["summary"]
    led = ledger()
    out["cumulative_usd"] = round(led.cost_usd() + led.extra_cost_usd(), 4)
    C.write_json(os.path.join(R2_DIR, "summary_regime2.json"), out)
    print("summary_regime2.json written; cumulative=$%.2f" % out["cumulative_usd"])


def cmd_parity(_args):
    """Byte-parity spot check of the API request vs the EXPG prompt builder."""
    programs = C.read_jsonl(prog_path())[:3]
    checks = []
    for p in programs:
        user = tf.make_user_message(p["listing_text"])
        checks.append({
            "program_id": p["program_id"],
            "user_equals_expg_builder": user == tf.make_user_message(
                tf.make_listing_text(p["stmt_texts"])),
            "gold_prefill": tf.GOLD_PREFILL,
            "gold_prefill_trailing_ok": not tf.GOLD_PREFILL[-1].isspace(),
        })
    doc = {"pass": all(c["user_equals_expg_builder"] for c in checks), "checks": checks,
           "note": ("user turn == tf.make_user_message(listing) byte-identical; gold prefill "
                    "'line 1:' has no trailing whitespace; continuation prefill == "
                    "manifest prefix_text with its single trailing newline stripped "
                    "(Anthropic rejects trailing whitespace in the final assistant turn) -- "
                    "stripped flag recorded per raw row; first_char_digit measured per "
                    "validated row to bound digit-gluing risk")}
    C.write_json(os.path.join(R2_DIR, "prefill_parity_check.json"), doc)
    print(json.dumps(doc, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "parity", "gold", "perturb",
                                      "validate", "summarize"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    os.makedirs(R2_DIR, exist_ok=True)
    {"prepare": cmd_prepare, "parity": cmd_parity, "gold": cmd_gold,
     "perturb": cmd_perturb, "validate": cmd_validate,
     "summarize": cmd_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
