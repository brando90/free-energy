"""E11 haiku (Anthropic) TRUE-PREFILL arm. Key-gated: requires ANTHROPIC_API_KEY
in the environment (source your keys env file first) -- this script never asks
for or stores a credential.

Reuses:
  * e11_run.py          worlds load, build_injection_e11, 3-way DV, wilson,
                        cluster bootstrap, cont/gold run ids (single source of
                        truth for the injection + scoring seams).
  * exph/api_gen.py     CostLedger (hard-stop + cumulative extra_paths), the
                        Anthropic client, _anthropic_call, PRICES.

Prefill regime: messages = [user(make_user_message(listing)),
assistant(prefix_text)]; the model continues the trace it is handed, exactly
the EXPH prefill semantics. Gold prefill = "line 1:".

Sampling: rollout 0 greedy (temperature 0.0); rollouts 1..R at temperature
--temp (Anthropic has no n>1, so R sequential calls/prompt). Cost-bounded:
hard stop at api_gen.HARD_BUDGET_USD AND a per-run soft cap (--usd-cap).

Usage:
  source <your keys env>            # sets ANTHROPIC_API_KEY
  python e11_api.py --out-dir <D> --n-target 60 --R 3 --usd-cap 55
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e11_run as R                              # noqa: E402
import trace_format as tf                        # noqa: E402
import inject as inj_mod                          # noqa: E402
import validate as val_mod                        # noqa: E402
from interp import parse_program                  # noqa: E402

EXPH_DIR = os.environ.get("E11_EXPH_DIR") or \
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "exph"))
sys.path.insert(0, EXPH_DIR)
import api_gen                                    # noqa: E402

MODEL_ID = "claude-haiku-4-5"
CFG_GREEDY = {"model_id": MODEL_ID, "mode": "prefill", "temperature": 0.0,
              "max_tokens": 384}


def _cfg(temp):
    return {"model_id": MODEL_ID, "mode": "prefill", "temperature": temp,
            "max_tokens": 384}


def _call(cfg, user_msg, prefill, ledger):
    # Anthropic rejects a final assistant message ending in whitespace; strip the
    # trailing ws (the trace's line-terminating "\n") for the request, then re-prepend
    # it to the returned continuation so the reconstructed/scored trace is unchanged.
    stripped = prefill.rstrip()
    trailing = prefill[len(stripped):]
    msgs = [{"role": "user", "content": user_msg},
            {"role": "assistant", "content": stripped}]
    out = api_gen._anthropic_call(cfg, msgs)
    if out.get("text") is not None and trailing:
        out["text"] = trailing + out["text"]
    ledger.add(MODEL_ID, out.get("input_tokens"), out.get("output_tokens"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--programs", default=None,
                    help="programs.jsonl to reuse (default: generate fresh worlds)")
    ap.add_argument("--world-pool", type=int, default=180)
    ap.add_argument("--n-target", type=int, default=60)
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-gold", type=int, default=512)
    ap.add_argument("--usd-cap", type=float, default=55.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Source your keys env file, then rerun. "
                         "This script will not prompt for a credential.")
    os.makedirs(a.out_dir, exist_ok=True)
    P = R.paths(a.out_dir)
    ledger = api_gen.CostLedger(os.path.join(a.out_dir, "cost_ledger.json"),
                                extra_paths=_sibling_ledgers(a.out_dir))

    def budget_ok():
        try:
            total = ledger.check_budget()
        except api_gen.BudgetExceeded as e:
            print("[E11-api] %s" % e)
            return False
        if ledger.cost_usd(MODEL_ID) > a.usd_cap:
            print("[E11-api] soft cap hit: this-run $%.2f > $%.2f"
                  % (ledger.cost_usd(MODEL_ID), a.usd_cap))
            return False
        return True

    # ---- worlds (reuse open-model programs.jsonl if given -> identical worlds)
    if a.programs and os.path.exists(a.programs):
        programs = R.read_jsonl(a.programs)
        import shutil
        shutil.copy(a.programs, P["programs"])
    else:
        for k in R.KS:
            progs, _ = R.gd.generate_cell(k, a.world_pool, 700000 + k * 911,
                                          audit_fn=R._audit_fn_for(k))
            for p in progs:
                R.append_jsonl(P["programs"], p)
        programs = R.read_jsonl(P["programs"])
    prog_by_id = {p["program_id"]: p for p in programs}

    existing = {r["run_id"] for r in R.read_jsonl(P["raw"])}
    # ---- gold (own-trace, prefill "line 1:")
    by_k = defaultdict(list)
    for p in programs:
        if not budget_ok():
            break
        rid = R.gold_run_id(p["program_id"], a.seed)
        if rid in existing:
            g = next((r for r in R.read_jsonl(P["raw"]) if r["run_id"] == rid), None)
        else:
            out = _call({**CFG_GREEDY, "max_tokens": a.max_new_gold},
                        tf.make_user_message(tf.make_listing_text(p["stmt_texts"])),
                        tf.GOLD_PREFILL, ledger)
            g = {"run_id": rid, "program_id": p["program_id"], "k": p["k"],
                 "condition": "gold", "continuation": out.get("text"),
                 "failed_generation": out.get("text") is None}
            R.append_jsonl(P["raw"], g)
            existing.add(rid)
        if g and not g.get("failed_generation"):
            ev = inj_mod.gold_solve_eval(p, tf.GOLD_PREFILL + g["continuation"])
            if ev["solved"]:
                by_k[p["k"]].append((p, g))
    ledger.save()

    # ---- cohort + manifest + continuations
    validated = []
    gold_solve = {}
    for k in R.KS:
        elig = sorted(by_k[k], key=lambda pg: pg[0]["program_id"])[:a.n_target]
        gold_solve[str(k)] = {"eligible": len(by_k[k]), "kept": len(elig)}
        for p, g in elig:
            stmts = parse_program(p["stmt_texts"])
            for fam in R.families_of(p):
                try:
                    b = R.build_injection_e11(p, fam, tf.GOLD_PREFILL + g["continuation"])
                except inj_mod.InjectError:
                    continue
                kk, opac = R.cell_of(p, fam)
                inj = {x: b[x] for x in ("cf", "planted_var", "planted_value",
                                         "force_after_line", "expected_lines_from")}
                um = tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))
                for roll in range(0, a.R + 1):
                    if not budget_ok():
                        break
                    cfg = CFG_GREEDY if roll == 0 else _cfg(a.temp)
                    out = _call(cfg, um, b["prefix_text"], ledger)
                    rec = {"program_id": p["program_id"], "family": fam, "k": kk,
                           "opacity": opac, "rollout": roll,
                           "planted_value": b["planted_value"], "true_value": b["true_value"]}
                    if out.get("text") is None:
                        rec.update({"generation_failed": True, "label": "unresolved"})
                    else:
                        met = val_mod.classify_run(stmts, p, inj, out["text"])
                        rec.update({x: met[x] for x in ("final_output_absorbed",
                                    "final_output_valid", "repair_event", "trace_valid",
                                    "next_read_absorbed", "doubt_lex", "unparsed")})
                        rec["judge_reject"] = None
                        rec["label"] = R.three_way_label(rec)
                    validated.append(rec)
                    R.append_jsonl(P["validated"], rec)
        ledger.save()

    # ---- summary (per k x opacity, 3-way rates + contrasts)
    cells = {}
    for k in R.KS:
        for opac in (["bare"] if k == 0 else R.OPACITIES):
            crows = [r for r in validated if r["k"] == k and r["opacity"] == opac]
            if crows:
                cells["k%d_%s" % (k, opac)] = R._cell_summary(crows, a.seed)
    summ = {"experiment": "E11_DEPTH_OPACITY_haiku_prefill", "model": MODEL_ID,
            "created_at": R.now_iso(), "cells": cells, "gold_cohort": gold_solve,
            "ledger_this_run_usd": round(ledger.cost_usd(MODEL_ID), 4),
            "flag_channel": "doubt_lex (frozen lexicon; no 32B judge on API arm)",
            "note": "haiku pilot deriv-absorption was 0.000; DV question = does any "
                    "k/opacity move absorbed off the floor."}
    R.write_json(P["summary"], summ)
    ledger.save()
    print(json.dumps({"gold_cohort": gold_solve,
                      "absorbed_by_cell": {n: c["absorbed"]["rate"]
                                           for n, c in cells.items()},
                      "this_run_usd": summ["ledger_this_run_usd"]}, indent=2))


def _sibling_ledgers(out_dir):
    """Sum prior API ledgers into the cumulative hard-stop check."""
    base = os.path.abspath(os.path.join(out_dir, "..", "..", ".."))
    found = []
    for root, _dirs, files in os.walk(base):
        if ".venv" in root or "__pycache__" in root:
            continue
        for f in files:
            if f == "cost_ledger.json":
                found.append(os.path.join(root, f))
    return found


if __name__ == "__main__":
    main()
