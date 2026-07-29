"""ARM A -- gpt-5.1 reasoning ON vs OFF on the recompute-wall computed cells.

Reuses the E10 bridge harness worlds EXACTLY: the manifest_gpt-5.1.jsonl rows
(same programs.jsonl worlds, same planted values, same prefix_text) for the three
directive cells: adjacent_contradiction (readable), onehop_kc1 (1-op computed),
deep_kc5 (5-op computed). The partial trace (with the plant) is presented in the
USER turn with the same "continue this trace" instruction as the bridge run
(e10bridge.bridge_user_message), so the ONLY difference vs the bridge reasoning-off
gpt-5.1 row is the reasoning setting.

Endpoint: OpenAI Responses API (the only gpt-5.1 path that returns reasoning
SUMMARIES). Two conditions, within-endpoint, R=4 reps each:
  OFF : reasoning={"effort":"none"}                (reasoning_tokens == 0)
  ON  : reasoning={"effort":"medium","summary":"auto"}

Grading: the FINAL visible output (response.output_text) is scored by the same
EXPG re-execution validator (classify_run) -> final_output_absorbed 3-way.
For ON, the reasoning summary is stored and mechanically scanned (rederive_scan)
for re-derivation of the planted variable's true value.

NB: OpenAI reasoning summaries are SUMMARIZED, not the raw CoT -> the ARM A
re-derivation scan is a LOWER BOUND on true re-derivations (disclosed).

Usage: arm_a.py {run,validate,summarize} [--effort-on medium] [--reps 4]
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
RW = os.path.join(EXP, "improvement_plan", "iclr_exec", "recompute_wall")
BRIDGE = os.path.join(RW, "e10_api_bridge")
OUT_DIR = os.path.join(RW, "reasoning_arm", "arm_a_gpt51")
for p in (EXPH, EXPG, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import exph_common as C          # noqa: E402
import api_gen as A              # noqa: E402
import trace_format as tf        # noqa: E402
import validate as val_mod       # noqa: E402
from interp import parse_program  # noqa: E402
import rederive_scan as RS        # noqa: E402
import openai                     # noqa: E402

MODEL_ID = "gpt-5.1"
CELLS = ("adjacent_contradiction", "onehop_kc1", "deep_kc5")
PROG_SRC = os.path.join(EXP, "results", "EXPH2_FRONTIER_FOLLOWUPS", "regime2",
                        "programs.jsonl")
MANIFEST = os.path.join(BRIDGE, "manifest_%s.jsonl" % MODEL_ID)
MAX_OUT = 4000
CONCURRENCY = 8

_CLIENT = None
_CLOCK = threading.Lock()


def client():
    global _CLIENT
    with _CLOCK:
        if _CLIENT is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY not set (source keys env first)")
            _CLIENT = openai.OpenAI(max_retries=0)
        return _CLIENT


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def programs():
    return {p["program_id"]: p for p in C.read_jsonl(PROG_SRC)}


def bridge_user_message(listing_text, prefix_text):
    return (tf.INSTRUCTION + tf.EXEMPLARS + "Program:\n" + listing_text +
            "\n\nA partial trace of this program has already been written; it is "
            "shown below under \"Trace:\". Continue it. Write only the remaining "
            "trace lines, in order, in the exact same format, picking up "
            "immediately after the last line shown. Do not repeat lines that are "
            "already written.\n\nTrace:\n" + prefix_text.rstrip("\n"))


def ledger():
    # cumulative hard-stop across the whole program; extra ledgers summed in.
    base = os.path.join(EXP, "results")
    extra = [os.path.join(base, "EXPH_API_MODELS", "cost_ledger.json")]
    fb = os.path.join(base, "EXPH2_FRONTIER_FOLLOWUPS")
    for sub in ("regime2", "ladder", "instruct_panel", "completions_probe"):
        extra.append(os.path.join(fb, sub, "cost_ledger.json"))
    for sub in ("e10_api_bridge", "e10_topup_fable", "e11_run/results/E11_haiku",
                "e11_run/results/E11_haiku_r8", "e12_fixes/A0", "e12_fixes/A1",
                "e12_fixes/A2", "e12_fixes/A3"):
        extra.append(os.path.join(RW, sub, "cost_ledger.json"))
    return A.CostLedger(os.path.join(OUT_DIR, "cost_ledger.json"), extra_paths=extra)


def call_responses(effort, input_text, led):
    """One Responses API call. Returns dict with text, reasoning summary, usage."""
    reasoning = {"effort": effort}
    if effort != "none":
        reasoning["summary"] = "auto"
    last = None
    for attempt in range(6):
        try:
            r = client().responses.create(model=MODEL_ID, input=input_text,
                                          reasoning=reasoning, max_output_tokens=MAX_OUT)
            u = r.usage.model_dump()
            summ_parts = []
            for it in r.output:
                if getattr(it, "type", None) == "reasoning":
                    for s in (getattr(it, "summary", None) or []):
                        t = getattr(s, "text", None)
                        if t:
                            summ_parts.append(t)
            led.add(MODEL_ID, u.get("input_tokens"), u.get("output_tokens"))
            return {"text": r.output_text, "reasoning_summary": "\n".join(summ_parts),
                    "status": r.status,
                    "input_tokens": u.get("input_tokens"),
                    "output_tokens": u.get("output_tokens"),
                    "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                    "error": None}
        except Exception as e:  # noqa: BLE001
            last = e
            name = type(e).__name__
            retry = name in ("RateLimitError", "InternalServerError", "APIConnectionError",
                             "APITimeoutError") or getattr(e, "status_code", None) in (429, 500, 502, 503, 504)
            if not retry or attempt == 5:
                return {"text": None, "reasoning_summary": None, "status": "error",
                        "error": "%s: %s" % (name, str(e)[:200])}
            import time, random
            time.sleep(min(30, 2 ** attempt + random.random()))
    return {"text": None, "error": "%s" % last}


def jobs_from_manifest(reps):
    man = [m for m in C.read_jsonl(MANIFEST) if m["cell"] in CELLS]
    progs = programs()
    jobs = []
    for m in man:
        p = progs[m["program_id"]]
        listing = tf.make_listing_text(p["stmt_texts"])
        input_text = bridge_user_message(listing, m["prefix_text"])
        for cond, effort in (("off", "none"), ("on", None)):  # effort filled at runtime
            for rep in range(reps):
                rid = C.sha_row(["ARMA", MODEL_ID, m["cell"], m["program_id"], cond, rep])
                jobs.append({"run_id": rid, "cond": cond, "rep": rep,
                             "cell": m["cell"], "program_id": m["program_id"],
                             "shape": m["shape"], "cf": m["cf"],
                             "planted_var": m["planted_var"], "planted_value": m["planted_value"],
                             "true_value": m["true_value"], "force_after_line": m["force_after_line"],
                             "expected_lines_from": m["expected_lines_from"],
                             "input_text": input_text})
    return jobs


def cmd_run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    led = ledger()
    raw_path = os.path.join(OUT_DIR, "raw_arm_a.jsonl")
    existing = {r["run_id"] for r in C.read_jsonl(raw_path)}
    jobs = [j for j in jobs_from_manifest(args.reps) if j["run_id"] not in existing]
    print("[run] pending=%d (of %d cells x worlds x 2cond x %dreps)"
          % (len(jobs), len(CELLS), args.reps))
    lock = threading.Lock()
    counter = {"n": 0}

    def work(j):
        effort = "none" if j["cond"] == "off" else args.effort_on
        out = call_responses(effort, j["input_text"], led)
        rec = {k: v for k, v in j.items() if k != "input_text"}
        rec.update({"effort": effort, "continuation": out.get("text"),
                    "reasoning_summary": out.get("reasoning_summary"),
                    "status": out.get("status"), "error": out.get("error"),
                    "input_tokens": out.get("input_tokens"),
                    "output_tokens": out.get("output_tokens"),
                    "reasoning_tokens": out.get("reasoning_tokens"),
                    "created_at": now_iso()})
        with lock:
            C.append_jsonl(raw_path, rec)
            counter["n"] += 1
            if counter["n"] % 40 == 0:
                led.check_budget()
                print("  ... %d done, cum=$%.2f" % (counter["n"], led.cost_usd() + led.extra_cost_usd()))
        return rec

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(work, jobs))
    led.check_budget()
    led.save()
    print("[run] done. this-run gpt-5.1 spend=$%.2f cumulative=$%.2f"
          % (led.cost_usd(MODEL_ID), led.cost_usd() + led.extra_cost_usd()))


def cmd_validate(_args):
    progs = programs()
    raw = C.read_jsonl(os.path.join(OUT_DIR, "raw_arm_a.jsonl"))
    out_path = os.path.join(OUT_DIR, "validated_arm_a.jsonl")
    rows = []
    for r in raw:
        p = progs[r["program_id"]]
        rec = dict(r)
        if r.get("continuation") is None:
            rec.update({"generation_failed": True, "class": "generation_failed"})
            rows.append(rec)
            continue
        inj = {"cf": r["cf"], "planted_var": r["planted_var"],
               "planted_value": r["planted_value"], "force_after_line": r["force_after_line"],
               "expected_lines_from": r["expected_lines_from"]}
        metrics = val_mod.classify_run(parse_program(p["stmt_texts"]), p, inj, r["continuation"])
        rec.update(metrics)
        rec["generation_failed"] = False
        # re-derivation scan on the reasoning summary (ON condition only has one)
        if r.get("reasoning_summary"):
            sc = RS.scan(r["reasoning_summary"], r["planted_var"], r["planted_value"], r["true_value"])
            rec["rederive"] = sc
            if r["cf"]:
                rec["rederive_bucket"] = RS.taxonomy_bucket(sc, rec.get("final_output_absorbed"))
                rec["rederive_bucket_loose"] = RS.taxonomy_bucket_loose(sc, rec.get("final_output_absorbed"))
        rows.append(rec)
    C.write_jsonl(out_path, rows)
    print("[validate] rows=%d written to %s" % (len(rows), out_path))


def _rate(k, n):
    return {"count": k, "n": n, "rate": round(k / n, 4) if n else None,
            "wilson95": C.wilson(k, n) if n else None}


def cmd_summarize(_args):
    rows = C.read_jsonl(os.path.join(OUT_DIR, "validated_arm_a.jsonl"))
    out = {"created_at": now_iso(), "model": MODEL_ID, "cells": list(CELLS),
           "by_cell_cond": {}, "rederive_taxonomy": {}}
    for cell in CELLS:
        out["by_cell_cond"][cell] = {}
        for cond in ("off", "on"):
            sub = [r for r in rows if r["cell"] == cell and r["cond"] == cond
                   and not r.get("generation_failed")]
            cf_rows = [r for r in sub if r.get("cf")]
            fo = [r for r in cf_rows if r.get("final_output_absorbed") is not None]
            # 3-way format-robust
            absorbed = silent = flagged = other = 0
            for r in cf_rows:
                if r.get("doubt_lex"):
                    flagged += 1
                elif r.get("final_output_absorbed"):
                    absorbed += 1
                elif r.get("final_output_valid"):
                    silent += 1
                else:
                    other += 1
            m = len(cf_rows)
            d = {"n": len(sub), "n_cf": m,
                 "final_output_absorbed": _rate(sum(bool(r.get("final_output_absorbed")) for r in fo), len(fo)),
                 "absorbed": _rate(absorbed, m), "silently_corrected": _rate(silent, m),
                 "flagged": _rate(flagged, m), "other": _rate(other, m),
                 "doubt_lex": _rate(sum(bool(r.get("doubt_lex")) for r in cf_rows), m),
                 "mean_reasoning_tokens": (round(sum(r.get("reasoning_tokens") or 0 for r in sub) / len(sub), 1)
                                           if sub else None)}
            # readable cell (adjacent_contradiction) is cf too; final_output_absorbed applies
            out["by_cell_cond"][cell][cond] = d
    # re-derivation taxonomy (ON, cf cells)
    for cell in CELLS:
        on_cf = [r for r in rows if r["cell"] == cell and r["cond"] == "on"
                 and r.get("cf") and not r.get("generation_failed") and "rederive_bucket" in r]
        strict = Counter(r["rederive_bucket"] for r in on_cf)
        loose = Counter(r.get("rederive_bucket_loose") for r in on_cf)
        n = len(on_cf)
        out["rederive_taxonomy"][cell] = {
            "n": n,
            "strict_assign": {k: {"count": strict.get(k, 0), "rate": round(strict.get(k, 0) / n, 4) if n else None}
                              for k in ("rederive_and_correct", "rederive_and_absorb", "never_rederive")},
            "loose_token": {k: {"count": loose.get(k, 0), "rate": round(loose.get(k, 0) / n, 4) if n else None}
                            for k in ("rederive_and_correct", "rederive_and_absorb", "never_rederive")},
            "any_summary_present": sum(1 for r in on_cf if (r.get("reasoning_summary") or "").strip()),
        }
    led = ledger()
    out["cumulative_usd"] = round(led.cost_usd() + led.extra_cost_usd(), 4)
    out["arm_a_gpt51_usd"] = round(led.cost_usd(MODEL_ID), 4)
    C.write_json(os.path.join(OUT_DIR, "summary_arm_a.json"), out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "validate", "summarize"])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--effort-on", default="medium")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    {"run": cmd_run, "validate": cmd_validate, "summarize": cmd_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
