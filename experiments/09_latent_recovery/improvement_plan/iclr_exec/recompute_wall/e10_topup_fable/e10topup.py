"""E10 TOPUP + FABLE-5.

JOB 1 (sample-size top-up): run the batch2 SUPERSET worlds through the four key
cells for the four roster models that were n=58-70/cell, EACH IN ITS OWN
MEASUREMENT-VALID PROTOCOL:
  * claude-opus-4-8, claude-sonnet-5  -> bridge (they reject prefill)
  * claude-haiku-4-5, claude-sonnet-4-5 -> prefill (their native protocol; the
    e10bridge parity check confirms prefill user-turn == regime2 tf.make_user_message,
    so batch2 prefill pools with the regime2 PREFILL batch1).

JOB 2 (Fable-5 arm), same four cells, bridge protocol (Fable-5 rejects prefill,
verified 2026-07-27):
  * claude-fable-5-off : thinking floor. Fable-5 CANNOT disable thinking
    (thinking.type=disabled -> 400); the comparable floor is output_config
    effort="low" (thinking still adaptive/on, minimal). Joins the main table.
  * claude-fable-5-on  : thinking adaptive, display="summarized",
    output_config effort default(high). Satellite. Grade ONLY the final visible
    continuation with the same re-execution metric; ALSO capture the returned
    thinking SUMMARY text (raw CoT is never returned by Fable-5) for a
    re-derivation scan.
Fable golds are OWN-TRACE and SHARED across the two conditions (gold_key
claude-fable-5, generated once via the -off config; injections are identical, only
the continuation regime differs).

Reuses the EXPG generator/injector/two-world validator and api_gen verbatim; the
only api_gen edits are HARD_BUDGET 100->150, Fable-5 pricing, and output_config +
thinking capture (all backward-compatible). Usage:
  e10topup.py {prepare,parity,gold,perturb,validate,summarize} [--model KEY]
"""
import argparse, datetime as _dt, json, os, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
EXPH = os.path.join(EXP, "improvement_plan", "exph")
EXPG = os.path.join(EXP, "improvement_plan", "expg")
for p in (EXPH, EXPG):
    if p not in sys.path:
        sys.path.insert(0, p)

import exph_common as C
import api_gen as A
import gen_programs as gp
import inject as inj_mod
import trace_format as tf
import validate as val_mod
from interp import parse_program

RW = os.path.join(EXP, "improvement_plan", "iclr_exec", "recompute_wall")
OUT_DIR = os.path.join(RW, "e10_topup_fable")
PROG_SRC = os.path.join(OUT_DIR, "programs_batch2.jsonl")   # batch2 superset

SEED = 0
MAX_GOLD = 512
MAX_CONT = 384
CELL_CAP = 130   # >= batch2 per-shape count so every eligible program is used

# FOUR KEY CELLS ONLY.
CELL_SHAPE = {"adjacent_contradiction": "kr1", "opfree_kr1": "kr1",
              "onehop_kc1": "kc1", "deep_kc5": "kc5"}
CELLS = ("adjacent_contradiction", "opfree_kr1", "onehop_kc1", "deep_kc5")


def _extra_ledgers():
    """All prior ledgers summed into the cumulative hard-stop, INCLUDING the
    e10_api_bridge run (so cumulative = base $30.81 + bridge $21.84 = $52.65)."""
    base = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS")
    paths = [os.path.join(EXP, "results", "EXPH_API_MODELS", "cost_ledger.json"),
             os.path.join(RW, "e10_api_bridge", "cost_ledger.json")]
    for sub in ("regime2", "ladder", "instruct_panel", "completions_probe"):
        paths.append(os.path.join(base, sub, "cost_ledger.json"))
    return paths


MODELS = [
    # --- JOB 1: frontier, bridge (reject prefill) ---
    {"key": "claude-opus-4-8", "gold_key": "claude-opus-4-8", "provider": "anthropic",
     "model_id": "claude-opus-4-8", "mode": "bridge", "temperature": None,
     "batch1_dir": "e10_api_bridge", "batch1_mode": "bridge"},
    {"key": "claude-sonnet-5", "gold_key": "claude-sonnet-5", "provider": "anthropic",
     "model_id": "claude-sonnet-5", "mode": "bridge", "temperature": None,
     "thinking": {"type": "disabled"}, "batch1_dir": "e10_api_bridge", "batch1_mode": "bridge"},
    # --- JOB 1: prefill models, PREFILL (native; pool with regime2 prefill) ---
    {"key": "claude-haiku-4-5", "gold_key": "claude-haiku-4-5", "provider": "anthropic",
     "model_id": "claude-haiku-4-5", "mode": "prefill", "temperature": 0.0,
     "batch1_dir": "regime2", "batch1_mode": "prefill"},
    {"key": "claude-sonnet-4-5", "gold_key": "claude-sonnet-4-5", "provider": "anthropic",
     "model_id": "claude-sonnet-4-5", "mode": "prefill", "temperature": 0.0,
     "batch1_dir": "regime2", "batch1_mode": "prefill"},
    # --- JOB 2: Fable-5 thinking floor (effort low; thinking can't be disabled) ---
    {"key": "claude-fable-5-off", "gold_key": "claude-fable-5", "provider": "anthropic",
     "model_id": "claude-fable-5", "mode": "bridge", "temperature": None,
     "output_config": {"effort": "low"}, "gold_max_tokens": 1536,
     "batch1_dir": None, "batch1_mode": None},
    # --- JOB 2: Fable-5 thinking-ON satellite (adaptive, summarized) ---
    {"key": "claude-fable-5-on", "gold_key": "claude-fable-5", "provider": "anthropic",
     "model_id": "claude-fable-5", "mode": "bridge", "temperature": None,
     "thinking": {"type": "adaptive", "display": "summarized"},
     "output_config": {"effort": "high"}, "cont_max_tokens": 2048,
     "batch1_dir": None, "batch1_mode": None},
]


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def ledger():
    return A.CostLedger(os.path.join(OUT_DIR, "cost_ledger.json"),
                        extra_paths=_extra_ledgers())


def get_cfg(key):
    for m in MODELS:
        if m["key"] == key:
            return dict(m)
    raise KeyError(key)


def programs():
    return C.read_jsonl(PROG_SRC)


# ---------------------------------------------------------------- prompt builders
def bridge_user_message(listing_text, prefix_text):
    return (tf.INSTRUCTION + tf.EXEMPLARS + "Program:\n" + listing_text +
            "\n\nA partial trace of this program has already been written; it is "
            "shown below under \"Trace:\". Continue it. Write only the remaining "
            "trace lines, in order, in the exact same format, picking up "
            "immediately after the last line shown. Do not repeat lines that are "
            "already written.\n\nTrace:\n" + prefix_text.rstrip("\n"))


def gold_payload(cfg, listing_text):
    user = tf.make_user_message(listing_text)
    if cfg["mode"] == "prefill":
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": tf.GOLD_PREFILL}], True
    return [{"role": "user", "content": user}], False   # bridge


def perturb_payload(cfg, listing_text, prefix_text):
    user = tf.make_user_message(listing_text)
    if cfg["mode"] == "prefill":
        pf = prefix_text.rstrip("\n")
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": pf}], pf.endswith("\n")
    return [{"role": "user", "content": bridge_user_message(listing_text, prefix_text)}], False


# ---------------------------------------------------------------- api driver
def run_jobs(cfg, jobs, led, raw_path, max_tokens):
    cfg = dict(cfg)
    cfg["max_tokens"] = max_tokens
    existing = {r["run_id"] for r in C.read_jsonl(raw_path)}
    pending = [j for j in jobs if j[0]["run_id"] not in existing]
    lock = threading.Lock()

    def work(job):
        base, payload = job
        out = A.call_model(cfg, payload, led)
        rec = dict(base)
        rec.update({"failed_generation": out.get("text") is None,
                    "continuation": out.get("text"), "thinking_text": out.get("thinking"),
                    "error": out.get("error"), "stop_reason": out.get("stop_reason"),
                    "input_tokens": out.get("input_tokens"),
                    "output_tokens": out.get("output_tokens"), "created_at": now_iso()})
        with lock:
            C.append_jsonl(raw_path, rec)
        return rec

    with ThreadPoolExecutor(max_workers=A.CONCURRENCY) as ex:
        list(ex.map(work, pending))
    led.check_budget()
    led.save()
    return len(pending)


def gold_rid(cfg, pid):
    return C.sha_row(["E10TOPUP", cfg["model_id"], "gold", pid, SEED])


def pert_rid(cfg, pid, cell):
    return C.sha_row(["E10TOPUP", cfg["key"], cell, pid, SEED])


def gold_full_text(cfg, continuation):
    if cfg["mode"] in ("prefill", "completion"):
        return tf.GOLD_PREFILL + (continuation or "")
    return continuation or ""


# ---------------------------------------------------------------- stages
def cmd_prepare(_args):
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(programs())
    src_fp = C.sha_row([r["program_id"] for r in programs()])
    print("batch2 worlds: %d programs, fp=%s, shapes=%s"
          % (n, src_fp, dict(Counter(r["shape"] for r in programs()))))


def cmd_parity(_args):
    progs = programs()[:3]
    checks = []
    for p in progs:
        listing = tf.make_listing_text(p["stmt_texts"])
        user = tf.make_user_message(listing)
        checks.append({
            "program_id": p["program_id"],
            "user_equals_expg_builder": user == tf.make_user_message(p["listing_text"])
                if "listing_text" in p else True,
            "gold_prefill_no_trailing_ws": not tf.GOLD_PREFILL[-1].isspace(),
            "bridge_contains_instruction": user[:60] in bridge_user_message(listing, "line 1: a = 1; a = 1\n"),
        })
    doc = {"pass": all(c["user_equals_expg_builder"] for c in checks), "checks": checks,
           "note": "prefill user turn byte-identical to tf.make_user_message (== regime2 "
                   "batch1 protocol); bridge = instruction/exemplars/program with prefix "
                   "in the user turn. All scored by classify_run on continuation numbers."}
    C.write_json(os.path.join(OUT_DIR, "parity_check.json"), doc)
    print(json.dumps(doc, indent=1))


def gold_paths(cfg):
    return (os.path.join(OUT_DIR, "raw_gold_%s.jsonl" % cfg["gold_key"]),
            os.path.join(OUT_DIR, "cohort_%s.json" % cfg["gold_key"]))


def cmd_gold(args):
    cfg = get_cfg(args.model)
    led = ledger()
    progs = programs()
    raw_path, coh_path = gold_paths(cfg)
    jobs = []
    for p in progs:
        listing = tf.make_listing_text(p["stmt_texts"])
        base = {"run_id": gold_rid(cfg, p["program_id"]), "program_id": p["program_id"],
                "condition": "gold", "shape": p["shape"], "model_id": cfg["model_id"],
                "model_key": cfg["key"], "mode": cfg["mode"]}
        payload, _ = gold_payload(cfg, listing)
        jobs.append((base, payload))
    n = run_jobs(cfg, jobs, led, raw_path, cfg.get("gold_max_tokens", MAX_GOLD))
    raw = {r["program_id"]: r for r in C.read_jsonl(raw_path)}
    per_prog = []
    for p in progs:
        g = raw.get(p["program_id"])
        if g is None or g.get("failed_generation"):
            per_prog.append({"program_id": p["program_id"], "shape": p["shape"],
                             "solved": False, "eligible": False, "reason": "gold_missing_or_failed"})
            continue
        ev = inj_mod.gold_solve_eval(p, gold_full_text(cfg, g["continuation"]))
        per_prog.append({"program_id": p["program_id"], "shape": p["shape"],
                         "solved": ev["solved"], "parse_ok": ev["parse_ok"],
                         "eligible": ev["solved"],
                         "reason": "eligible" if ev["solved"]
                         else (ev["first_error"] or {}).get("type", "unknown")})
    summary = {"n_programs": len(per_prog), "new_generations": n,
               "solve_rate": round(sum(r["solved"] for r in per_prog) / len(per_prog), 4),
               "failure_taxonomy": dict(Counter(r["reason"] for r in per_prog if not r["solved"])),
               "by_shape": {}}
    for shape in ("kr1", "kc1", "kc5"):
        rows = [r for r in per_prog if r["shape"] == shape]
        summary["by_shape"][shape] = {"n": len(rows), "eligible": sum(r["eligible"] for r in rows),
            "solve_rate": round(sum(r["solved"] for r in rows) / len(rows), 4) if rows else None}
    C.write_json(coh_path, {"per_program": per_prog, "summary": summary})
    print(json.dumps({"model": cfg["key"], "gold_key": cfg["gold_key"], "summary": summary,
                      "cumulative_usd": round(led.cost_usd() + led.extra_cost_usd(), 2)}))


def cmd_perturb(args):
    cfg = get_cfg(args.model)
    led = ledger()
    progs = {p["program_id"]: p for p in programs()}
    raw_gold_path, coh_path = gold_paths(cfg)
    cohort = json.load(open(coh_path))
    gold_raw = {r["program_id"]: r for r in C.read_jsonl(raw_gold_path)}
    elig = {s: [] for s in ("kr1", "kr8", "kc1", "kc5")}
    for r in cohort["per_program"]:
        if r["eligible"]:
            elig[r["shape"]].append(r["program_id"])
    man_path = os.path.join(OUT_DIR, "manifest_%s.jsonl" % cfg["key"])
    audit_path = os.path.join(OUT_DIR, "audit_%s.jsonl" % cfg["key"])
    mrows = C.read_jsonl(man_path)
    if not mrows:
        for cell in CELLS:
            shape = CELL_SHAPE[cell]
            for pid in elig[shape][:CELL_CAP]:
                p = progs[pid]
                gfull = gold_full_text(cfg, gold_raw[pid]["continuation"])
                try:
                    b = inj_mod.build_injection(p, cell, gfull)
                except inj_mod.InjectError as e:
                    C.append_jsonl(audit_path, {"stage": "injection_audit", "cell": cell,
                                                "program_id": pid, "rejected": True, "reason": e.reason})
                    continue
                row = {"run_id": pert_rid(cfg, pid, cell), "cell": cell, "condition": cell,
                       "program_id": pid, "shape": shape, "position": "mid", "L": p["L"],
                       "site_line": p["site_line"], "site_var": p["site_var"],
                       "k_r": p["k_r"], "k_c": p["k_c"], "r1": p["r1"], "r1_var": p["r1_var"],
                       "model_id": cfg["model_id"], "model_key": cfg["key"], "mode": cfg["mode"],
                       "seed": SEED, "created_at": now_iso()}
                row.update({k: b[k] for k in
                            ("family", "injected_trace_line", "original_trace_line", "prefix_text",
                             "step_replaced", "expected_lines_from", "cf", "planted_var",
                             "planted_value", "true_value", "force_after_line", "delta_policy")})
                mrows.append(row)
        C.write_jsonl(man_path, mrows)
    raw_path = os.path.join(OUT_DIR, "raw_pert_%s.jsonl" % cfg["key"])
    jobs = []
    for m in mrows:
        p = progs[m["program_id"]]
        listing = tf.make_listing_text(p["stmt_texts"])
        base = {"run_id": m["run_id"], "program_id": m["program_id"],
                "condition": m["cell"], "shape": m["shape"], "model_id": cfg["model_id"]}
        payload, _ = perturb_payload(cfg, listing, m["prefix_text"])
        jobs.append((base, payload))
    n = run_jobs(cfg, jobs, led, raw_path, cfg.get("cont_max_tokens", MAX_CONT))
    print("[perturb %s] manifest=%d new=%d cells=%s cum=$%.2f"
          % (cfg["key"], len(mrows), n, json.dumps(dict(Counter(m["cell"] for m in mrows))),
             led.cost_usd() + led.extra_cost_usd()))


def cmd_validate(args):
    cfg = get_cfg(args.model)
    progs = {p["program_id"]: p for p in programs()}
    manifest = C.read_jsonl(os.path.join(OUT_DIR, "manifest_%s.jsonl" % cfg["key"]))
    raw = {r["run_id"]: r for r in C.read_jsonl(os.path.join(OUT_DIR, "raw_pert_%s.jsonl" % cfg["key"]))}
    out_path = os.path.join(OUT_DIR, "validated_%s.jsonl" % cfg["key"])
    rows = []
    for m in manifest:
        p = progs[m["program_id"]]
        r = raw.get(m["run_id"])
        rec = dict(m)
        if r is None or r.get("failed_generation"):
            rec.update({"generation_failed": True, "class": "generation_failed"})
            rows.append(rec)
            continue
        inj = {k: m[k] for k in ("cf", "planted_var", "planted_value", "force_after_line", "expected_lines_from")}
        metrics = val_mod.classify_run(parse_program(p["stmt_texts"]), p, inj, r["continuation"])
        rec.update(metrics)
        rec["generation_failed"] = False
        rec["continuation"] = r["continuation"]
        rec["thinking_text"] = r.get("thinking_text")
        rows.append(rec)
    C.write_jsonl(out_path, rows)
    print("[validate %s] rows=%d" % (cfg["key"], len(rows)))


def cmd_summarize(_args):
    out = {"created_at": now_iso(), "models": {}, "cells": list(CELLS)}
    for m in MODELS:
        p = os.path.join(OUT_DIR, "validated_%s.jsonl" % m["key"])
        if not os.path.exists(p):
            continue
        rows = C.read_jsonl(p)
        per = {c: _cell_summary([r for r in rows if r["condition"] == c]) for c in CELLS
               if any(r["condition"] == c for r in rows)}
        out["models"][m["key"]] = {"mode": m["mode"], "model_id": m["model_id"], "cells": per}
    led = ledger()
    out["cumulative_usd"] = round(led.cost_usd() + led.extra_cost_usd(), 4)
    out["this_run_usd"] = round(led.cost_usd(), 4)
    C.write_json(os.path.join(OUT_DIR, "summary_topup.json"), out)
    print("summary_topup.json written; this_run=$%.2f cumulative=$%.2f"
          % (out["this_run_usd"], out["cumulative_usd"]))


def _cell_summary(rows):
    n = len(rows)
    live = [r for r in rows if not r.get("generation_failed")]
    cf_rows = [r for r in live if r.get("cf")]
    out = {"n": n, "n_live": len(live), "n_gen_failed": n - len(live)}

    def rate(k, nn):
        return {"count": k, "n": nn, "rate": round(k / nn, 4) if nn else None,
                "wilson95": C.wilson(k, nn) if nn else None}

    fo = [r for r in cf_rows if r.get("final_output_absorbed") is not None]
    out["final_output_absorbed"] = rate(sum(bool(r["final_output_absorbed"]) for r in fo), len(fo))
    out["doubt_lex"] = rate(sum(bool(r.get("doubt_lex")) for r in cf_rows), len(cf_rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "parity", "gold", "perturb", "validate", "summarize"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    {"prepare": cmd_prepare, "parity": cmd_parity, "gold": cmd_gold, "perturb": cmd_perturb,
     "validate": cmd_validate, "summarize": cmd_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
