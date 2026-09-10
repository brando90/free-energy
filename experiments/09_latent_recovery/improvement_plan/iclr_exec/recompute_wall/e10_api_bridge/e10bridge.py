"""E10 substrate-gap bridge: EXPG program-trace substrate on the OpenAI roster
and the NEWEST Anthropic models (claude-opus-4-8, claude-sonnet-5).

Closes the gap flagged in E10_API_REPORT.md: those models previously had only
NL-substrate data (non-computed plants), so their rows could not speak to the
recompute wall. Here they are run on the SAME EXPG program-trace worlds as the
two prefill models already reported (regime2/programs.jsonl -- identical world
substrate, kr1:70 kr8:60 kc1:60 kc5:60), so opus-4-8 / sonnet-5 are directly
comparable to claude-haiku-4-5 / claude-sonnet-4-5 on true computed cells.

Three continuation protocols, all scored by the SAME EXPG two-world validator
(classify_run, re-execution of the continuation's numbers -- format-robust):
  * prefill  (Anthropic): assistant-turn continuation, EXPH2 regime2 harness
             semantics UNCHANGED (byte-parity user turn; prefix as assistant).
  * completion (OpenAI legacy v1/completions): user_message + prefix as a single
             token stream -- protocol-identical to prefill, no chat boundary.
  * bridge   (OpenAI chat): prefix presented in the USER turn with an explicit
             "continue this partial trace" instruction. DISCLOSED format shift:
             chat re-presentation can substitute re-planning for continuation.
             Anchor metric (final_output_absorbed via re-execution) is
             format-robust because it reads the continuation's numbers.

Own-trace golds per model. Usage:
  e10bridge.py {prepare,parity,gold,perturb,validate,summarize} [--model KEY]
"""
import argparse
import datetime as _dt
import json
import os
import sys
import threading
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

import exph_common as C          # noqa: E402
import api_gen as A              # noqa: E402
import gen_programs as gp        # noqa: E402
import inject as inj_mod         # noqa: E402
import trace_format as tf        # noqa: E402
import validate as val_mod       # noqa: E402
from interp import parse_program  # noqa: E402

RW = os.path.join(EXP, "improvement_plan", "iclr_exec", "recompute_wall")
OUT_DIR = os.path.join(RW, "e10_api_bridge")
PROG_SRC = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS", "regime2",
                        "programs.jsonl")  # identical worlds as the prefill models

SEED = 0
MAX_GOLD = 512
MAX_CONT = 384
CELL_CAP = 75

CELL_SHAPE = {"benign_paraphrase": "kr1", "true_interruption": "kr1",
              "adjacent_contradiction": "kr1", "opfree_kr1": "kr1",
              "opfree_kr8": "kr8", "onehop_kc1": "kc1", "deep_kc5": "kc5"}
CELLS = ("benign_paraphrase", "true_interruption", "adjacent_contradiction",
         "opfree_kr1", "opfree_kr8", "onehop_kc1", "deep_kc5")

# Cumulative-budget ledgers summed into the hard-stop check (read-only extras).
def _extra_ledgers():
    base = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS")
    paths = [os.path.join(EXP, "results", "EXPH_API_MODELS", "cost_ledger.json")]
    for sub in ("regime2", "ladder", "instruct_panel", "completions_probe"):
        paths.append(os.path.join(base, sub, "cost_ledger.json"))
    return paths


MODELS = [
    # --- newest Anthropic. NB: claude-opus-4-8 and claude-sonnet-5 REJECT
    #     assistant-message prefill via the API ("model does not support
    #     assistant message prefill; conversation must end with a user
    #     message"), verified 2026-07-27. The true-prefill harness therefore
    #     CANNOT be run on them; the only API path to the program-trace
    #     substrate is the chat bridge (prefix in the user turn, disclosed). ---
    {"key": "claude-opus-4-8_bridge", "provider": "anthropic",
     "model_id": "claude-opus-4-8", "mode": "bridge", "temperature": None},
    {"key": "claude-sonnet-5_bridge", "provider": "anthropic",
     "model_id": "claude-sonnet-5", "mode": "bridge", "temperature": None,
     "thinking": {"type": "disabled"}},
    # --- old Anthropic prefill models, run in BRIDGE mode as calibration
    #     anchors: they already have TRUE-PREFILL numbers in regime2 on these
    #     exact worlds, so the prefill->bridge delta here measures how much the
    #     regime shift alone moves absorption, letting us read opus/sonnet-5. ---
    {"key": "claude-haiku-4-5_bridge", "provider": "anthropic",
     "model_id": "claude-haiku-4-5", "mode": "bridge", "temperature": 0.0},
    {"key": "claude-sonnet-4-5_bridge", "provider": "anthropic",
     "model_id": "claude-sonnet-4-5", "mode": "bridge", "temperature": 0.0},
    # --- OpenAI legacy completions: protocol-identical single stream ---
    {"key": "gpt-3.5-turbo-instruct_completion", "provider": "openai",
     "model_id": "gpt-3.5-turbo-instruct", "mode": "completion", "temperature": 0.0},
    # --- OpenAI chat: bridged (prefix in user turn; disclosed) ---
    {"key": "gpt-4.1_bridge", "provider": "openai", "model_id": "gpt-4.1",
     "mode": "bridge", "temperature": 0.0},
    {"key": "gpt-4o_bridge", "provider": "openai", "model_id": "gpt-4o",
     "mode": "bridge", "temperature": 0.0},
    {"key": "gpt-5.1_bridge", "provider": "openai", "model_id": "gpt-5.1",
     "mode": "bridge", "temperature": None, "reasoning_effort": "none"},
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
    """Chat-mode bridge: same instruction + exemplars + program, then the
    partial trace (with the plant) presented for continuation. Disclosed shift."""
    return (tf.INSTRUCTION + tf.EXEMPLARS + "Program:\n" + listing_text +
            "\n\nA partial trace of this program has already been written; it is "
            "shown below under \"Trace:\". Continue it. Write only the remaining "
            "trace lines, in order, in the exact same format, picking up "
            "immediately after the last line shown. Do not repeat lines that are "
            "already written.\n\nTrace:\n" + prefix_text.rstrip("\n"))


def gold_payload(cfg, listing_text):
    """Return the payload for a gold generation (mode-specific) and a flag saying
    whether GOLD_PREFILL must be prepended to the continuation to reconstruct the
    full gold trace."""
    user = tf.make_user_message(listing_text)
    if cfg["mode"] == "prefill":
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": tf.GOLD_PREFILL}], True
    if cfg["mode"] == "completion":
        return user + "\n" + tf.GOLD_PREFILL, True
    # bridge (chat): plain trace request, model writes the whole trace itself
    return [{"role": "user", "content": user}], False


def perturb_payload(cfg, listing_text, prefix_text):
    user = tf.make_user_message(listing_text)
    if cfg["mode"] == "prefill":
        pf = prefix_text.rstrip("\n")  # Anthropic rejects trailing whitespace
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": pf}], pf.endswith("\n")
    if cfg["mode"] == "completion":
        return user + "\n" + prefix_text, False
    return [{"role": "user", "content": bridge_user_message(listing_text, prefix_text)}], False


# ---------------------------------------------------------------- api driver
def run_jobs(cfg, jobs, led, raw_path, max_tokens):
    """jobs: list of (base_row, payload). payload is a messages list or a raw
    string (completion). Resume-safe on run_id."""
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
                    "continuation": out.get("text"), "error": out.get("error"),
                    "stop_reason": out.get("stop_reason"),
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


def gold_rid(cfg, pid):
    return C.sha_row(["E10BRIDGE", cfg["model_id"], "gold", pid, SEED])


def pert_rid(cfg, pid, cell):
    return C.sha_row(["E10BRIDGE", cfg["model_id"], cell, pid, SEED])


def gold_full_text(cfg, continuation):
    if cfg["mode"] in ("prefill", "completion"):
        return tf.GOLD_PREFILL + (continuation or "")
    return continuation or ""  # bridge: model wrote the whole trace


# ---------------------------------------------------------------- stages
def cmd_prepare(_args):
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(programs())
    src_fp = C.sha_row([r["program_id"] for r in programs()])
    C.write_json(os.path.join(OUT_DIR, "worlds_provenance.json"),
                 {"source": PROG_SRC, "n_programs": n,
                  "program_id_fingerprint": src_fp,
                  "shapes": dict(Counter(r["shape"] for r in programs())),
                  "note": "identical worlds as regime2 prefill models "
                          "(claude-haiku-4-5, claude-sonnet-4-5)"})
    print("worlds: %d programs, fp=%s" % (n, src_fp))


def cmd_parity(_args):
    """Confirm the bridge/completion user turns are byte-anchored to the EXPG
    builder, and that prefill semantics match the regime2 harness."""
    progs = programs()[:3]
    checks = []
    for p in progs:
        listing = tf.make_listing_text(p["stmt_texts"])
        user = tf.make_user_message(listing)
        checks.append({
            "program_id": p["program_id"],
            "user_equals_expg_builder": user == tf.make_user_message(p["listing_text"]),
            "gold_prefill_no_trailing_ws": not tf.GOLD_PREFILL[-1].isspace(),
            "bridge_contains_instruction": user[:60] in bridge_user_message(listing, "line 1: a = 1; a = 1\n"),
        })
    doc = {"pass": all(c["user_equals_expg_builder"] for c in checks),
           "checks": checks,
           "note": "prefill user turn byte-identical to tf.make_user_message; "
                   "completion = user + '\\n' + prefix (single stream); bridge = "
                   "same instruction/exemplars/program with prefix in the user "
                   "turn (disclosed). All three scored by classify_run on the "
                   "continuation's numbers (format-robust)."}
    C.write_json(os.path.join(OUT_DIR, "parity_check.json"), doc)
    print(json.dumps(doc, indent=1))


def gold_paths(cfg):
    return (os.path.join(OUT_DIR, "raw_gold_%s.jsonl" % cfg["model_id"]),
            os.path.join(OUT_DIR, "cohort_%s.json" % cfg["model_id"]))


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
    n = run_jobs(cfg, jobs, led, raw_path, MAX_GOLD)
    raw = {r["program_id"]: r for r in C.read_jsonl(raw_path)}
    per_prog = []
    for p in progs:
        g = raw.get(p["program_id"])
        if g is None or g.get("failed_generation"):
            per_prog.append({"program_id": p["program_id"], "shape": p["shape"],
                             "solved": False, "eligible": False,
                             "reason": "gold_missing_or_failed"})
            continue
        ev = inj_mod.gold_solve_eval(p, gold_full_text(cfg, g["continuation"]))
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
            "solve_rate": round(sum(r["solved"] for r in rows) / len(rows), 4) if rows else None,
            "eligible": sum(r["eligible"] for r in rows)}
    C.write_json(coh_path, {"per_program": per_prog, "summary": summary})
    print(json.dumps({"model": cfg["model_id"], "summary": summary,
                      "cumulative_usd": round(led.cost_usd() + led.extra_cost_usd(), 2)}))


def cmd_perturb(args):
    cfg = get_cfg(args.model)
    led = ledger()
    progs = {p["program_id"]: p for p in programs()}
    raw_gold_path, coh_path = gold_paths(cfg)
    cohort = json.load(open(coh_path))
    gold_raw = {r["program_id"]: r for r in C.read_jsonl(raw_gold_path)}
    elig = {s: [] for s in gp.SHAPES}
    for r in cohort["per_program"]:
        if r["eligible"]:
            elig[r["shape"]].append(r["program_id"])
    man_path = os.path.join(OUT_DIR, "manifest_%s.jsonl" % cfg["model_id"])
    audit_path = os.path.join(OUT_DIR, "audit_%s.jsonl" % cfg["model_id"])
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
                                                "program_id": pid, "rejected": True,
                                                "reason": e.reason})
                    continue
                row = {"run_id": pert_rid(cfg, pid, cell), "cell": cell,
                       "condition": cell, "program_id": pid, "shape": shape,
                       "position": "mid", "L": p["L"], "site_line": p["site_line"],
                       "site_var": p["site_var"], "k_r": p["k_r"], "k_c": p["k_c"],
                       "r1": p["r1"], "r1_var": p["r1_var"],
                       "model_id": cfg["model_id"], "model_key": cfg["key"],
                       "mode": cfg["mode"], "seed": SEED, "created_at": now_iso()}
                row.update({k: b[k] for k in
                            ("family", "injected_trace_line", "original_trace_line",
                             "prefix_text", "step_replaced", "expected_lines_from",
                             "cf", "planted_var", "planted_value", "true_value",
                             "force_after_line", "delta_policy")})
                mrows.append(row)
        C.write_jsonl(man_path, mrows)
    raw_path = os.path.join(OUT_DIR, "raw_pert_%s.jsonl" % cfg["model_id"])
    jobs = []
    for m in mrows:
        p = progs[m["program_id"]]
        listing = tf.make_listing_text(p["stmt_texts"])
        base = {"run_id": m["run_id"], "program_id": m["program_id"],
                "condition": m["cell"], "shape": m["shape"], "model_id": cfg["model_id"]}
        payload, _ = perturb_payload(cfg, listing, m["prefix_text"])
        jobs.append((base, payload))
    n = run_jobs(cfg, jobs, led, raw_path, MAX_CONT)
    print("[perturb %s] manifest=%d new=%d cells=%s cum=$%.2f"
          % (cfg["model_id"], len(mrows), n,
             json.dumps(dict(Counter(m["cell"] for m in mrows))),
             led.cost_usd() + led.extra_cost_usd()))


def cmd_validate(args):
    cfg = get_cfg(args.model)
    progs = {p["program_id"]: p for p in programs()}
    manifest = C.read_jsonl(os.path.join(OUT_DIR, "manifest_%s.jsonl" % cfg["model_id"]))
    raw = {r["run_id"]: r for r in
           C.read_jsonl(os.path.join(OUT_DIR, "raw_pert_%s.jsonl" % cfg["model_id"]))}
    out_path = os.path.join(OUT_DIR, "validated_%s.jsonl" % cfg["model_id"])
    rows = []
    for m in manifest:
        p = progs[m["program_id"]]
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
        rows.append(rec)
    C.write_jsonl(out_path, rows)
    print("[validate %s] rows=%d" % (cfg["model_id"], len(rows)))


def cmd_summarize(_args):
    out = {"created_at": now_iso(), "models": {}, "cells": list(CELLS)}
    for m in MODELS:
        p = os.path.join(OUT_DIR, "validated_%s.jsonl" % m["model_id"])
        if not os.path.exists(p):
            continue
        rows = C.read_jsonl(p)
        per = {}
        for cell in CELLS:
            sub = [r for r in rows if r["condition"] == cell]
            if sub:
                per[cell] = _cell_summary(sub)
        out["models"][m["model_id"]] = {"mode": m["mode"], "cells": per}
        cp = os.path.join(OUT_DIR, "cohort_%s.json" % m["model_id"])
        if os.path.exists(cp):
            out.setdefault("gold_funnels", {})[m["model_id"]] = json.load(open(cp))["summary"]
    led = ledger()
    out["cumulative_usd"] = round(led.cost_usd() + led.extra_cost_usd(), 4)
    C.write_json(os.path.join(OUT_DIR, "summary_bridge.json"), out)
    print("summary_bridge.json written; cumulative=$%.2f" % out["cumulative_usd"])


def _cell_summary(rows):
    """3-way outcome (absorbed / silently_corrected / flagged) on cf cells, plus
    the EXPG channels needed for the readable-vs-recompute margin."""
    n = len(rows)
    live = [r for r in rows if not r.get("generation_failed")]
    cf_rows = [r for r in live if r.get("cf")]
    out = {"n": n, "n_live": len(live), "n_gen_failed": n - len(live)}

    def rate(k, nn):
        return {"count": k, "n": nn, "rate": round(k / nn, 4) if nn else None,
                "wilson95": C.wilson(k, nn) if nn else None}

    # final_output_absorbed on cf cells (the wall's primary channel)
    fo = [r for r in cf_rows if r.get("final_output_absorbed") is not None]
    out["final_output_absorbed"] = rate(sum(bool(r["final_output_absorbed"]) for r in fo), len(fo))
    nr = [r for r in cf_rows if r.get("next_read_absorbed") is not None]
    out["next_read_absorbed"] = rate(sum(bool(r["next_read_absorbed"]) for r in nr), len(nr))

    # 3-way, format-robust, on cf rows with an output claim decision
    absorbed = silent = flagged = other = 0
    for r in cf_rows:
        fl = bool(r.get("doubt_lex"))
        if fl:
            flagged += 1
        elif r.get("final_output_absorbed"):
            absorbed += 1
        elif r.get("final_output_valid"):
            silent += 1
        else:
            other += 1
    m = len(cf_rows)
    out["three_way"] = {
        "n_cf": m,
        "absorbed": rate(absorbed, m),
        "silently_corrected": rate(silent, m),
        "flagged": rate(flagged, m),
        "other": rate(other, m),
    }
    out["doubt_lex"] = rate(sum(bool(r.get("doubt_lex")) for r in cf_rows), m)
    out["repair_event"] = rate(sum(bool(r.get("repair_event")) for r in cf_rows), m)
    out["mean_line_skip"] = round(
        sum(r.get("line_skip_rate") or 0 for r in live) / len(live), 4) if live else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "parity", "gold", "perturb",
                                      "validate", "summarize"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    {"prepare": cmd_prepare, "parity": cmd_parity, "gold": cmd_gold,
     "perturb": cmd_perturb, "validate": cmd_validate,
     "summarize": cmd_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
