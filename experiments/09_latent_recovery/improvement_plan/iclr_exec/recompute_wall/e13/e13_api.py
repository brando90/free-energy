"""E13 haiku-4-5 (Anthropic) TRUE-PREFILL driver over all new E13 cells.

ADDITIVE runner, copy-derived from e11_api.py: api_gen CostLedger (cumulative
hard stop) + _anthropic_call; injection dispatch + new transforms via e13_lib
(contract RW/e13/README.md); scoring via validate.classify_run (existing
path). Worlds: the dry-run-audited pools in RW/e13/worlds/.

Sampling matches the E11 frontier arm: rollout 0 greedy (T=0.0) + rollouts
1..R at --temp (default R=3, temp 0.7; Anthropic has no n>1 so R sequential
calls/prompt).

k1_probe: SAME perturbed prefix as depth_k1_bare (prefix_family), but the
trace-continuation request is REPLACED by the direct question, delivered as a
follow-up user turn after the closed assistant turn holding the perturbed
partial trace (no prefill on the probe call; the conversation legally ends
with a user message). Raw answers stored verbatim in raw_generations.jsonl.

Budget: api_gen.HARD_BUDGET_USD cumulative hard stop over every
cost_ledger.json in the recompute_wall tree + EXPH ledgers, plus a per-run
soft --usd-cap. Projected spend per arm is printed BEFORE any live call.
NO LIVE CALLS until the Review agent issues GO.

Usage:
  python e13_api.py --out-dir results/E13_haiku --dry          # no API use
  source <keys env>  &&  python e13_api.py --out-dir results/E13_haiku \
      --n-target 60 --R 3 --usd-cap 12
"""
import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e13_lib as L                        # noqa: E402
import e11_run as e11                      # noqa: E402
import trace_format as tf                  # noqa: E402
import inject as inj_mod                   # noqa: E402

try:
    import api_gen                         # noqa: E402
except Exception:
    api_gen = None

MODEL_ID = "claude-haiku-4-5"
PRICE_IN, PRICE_OUT = 1.00, 5.00           # api_gen.PRICES[MODEL_ID]


def cfg_for(temp, max_tokens=384):
    return {"model_id": MODEL_ID, "mode": "prefill", "temperature": temp,
            "max_tokens": max_tokens}


def _call_prefill(cfg, user_msg, prefill, ledger):
    """e11_api semantics: strip trailing ws for the request, re-prepend after."""
    stripped = prefill.rstrip()
    trailing = prefill[len(stripped):]
    msgs = [{"role": "user", "content": user_msg},
            {"role": "assistant", "content": stripped}]
    out = api_gen._anthropic_call(cfg, msgs)
    if out.get("text") is not None and trailing:
        out["text"] = trailing + out["text"]
    ledger.add(MODEL_ID, out.get("input_tokens"), out.get("output_tokens"))
    return out


def _call_probe(cfg, user_msg, prefix_text, question, ledger):
    msgs = L.probe_messages(user_msg, prefix_text, question)
    out = api_gen._anthropic_call(cfg, msgs)
    ledger.add(MODEL_ID, out.get("input_tokens"), out.get("output_tokens"))
    return out


def grid(cell, pid, seed):
    return e11.sha(["E13api", MODEL_ID, cell, pid, "gold", seed])


def rid(cell, pid, fam, roll, seed):
    return e11.sha(["E13api", MODEL_ID, cell, pid, fam, roll, seed])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", default=",".join(L.ALL_CELLS))
    ap.add_argument("--n-target", type=int, default=60)
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-gold", type=int, default=512)
    ap.add_argument("--max-new-cont", type=int, default=384)
    ap.add_argument("--usd-cap", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true",
                    help="assemble prompts + projections; NO API calls")
    a = ap.parse_args()
    a.cells = [c for c in a.cells.split(",") if c]
    for c in a.cells:
        if c not in L.CELL_FAMILIES:
            raise SystemExit("unknown cell %s" % c)
    if not a.dry:
        if api_gen is None:
            raise SystemExit("api_gen (exph) not importable; API arm needs it")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set. Source the keys env "
                             "file, then rerun. This script never prompts for "
                             "or stores a credential.")
    os.makedirs(a.out_dir, exist_ok=True)
    P = {k: os.path.join(a.out_dir, v) for k, v in {
        "raw": "raw_generations.jsonl", "manifest": "manifest.jsonl",
        "validated": "validated_outputs.jsonl",
        "summary": "summary_tables.json", "run_manifest": "run_manifest.json",
        "audit": "generation_audit.jsonl", "examples": "example_prompts.json",
        "ledger": "cost_ledger.json", "projection": "spend_projection.json",
        "status": "queue_status.json"}.items()}

    worlds, world_meta = {}, {}
    for cell in a.cells:
        rows, wpath = L.load_worlds(cell)
        worlds[cell] = rows
        world_meta[cell] = {"path": wpath, "n_worlds": len(rows),
                            "sha256": L.sha256_file(wpath)}

    e11.write_json(P["run_manifest"], {
        "experiment": "E13_MATCHED_CONTROLS_haiku_prefill",
        "runner": "e13_api.py", "model": MODEL_ID,
        "created_at": e11.now_iso(), "dry": bool(a.dry), "seed": a.seed,
        "cells": a.cells, "families": {c: L.CELL_FAMILIES[c] for c in a.cells},
        "n_target_per_cell": a.n_target, "R": a.R, "temp": a.temp,
        "decoding": {"greedy_rollout": 0, "sampled_rollouts": a.R,
                     "max_new_gold": a.max_new_gold,
                     "max_new_cont": a.max_new_cont,
                     "probe_max_tokens": L.PROBE_MAX_TOKENS},
        "worlds": world_meta,
        "prompt_sha256": tf.prompt_sha256(), "gold_prefill": tf.GOLD_PREFILL,
        "injection_contract": "RW/e13/README.md (Build 1/2, binding)",
        "probe_note": ("prefix_family=depth_k1_bare (interpretation choice "
                       "flagged for Review); question as follow-up user turn; "
                       "raw answers retained"),
        "conditional_gate": ("families %s CONDITIONAL on parse-sanity gate in "
                             "summary" % (L.CONDITIONAL_FAMILIES,)),
        "gold_policy": ("live: own-trace haiku golds regenerated in this run "
                        "(prefill 'line 1:'); dry: interpreter-true stand-in, "
                        "labelled"),
        "flag_channel": "doubt_lex frozen lexicon (no 32B judge on API arm)"})

    ledger = None
    if not a.dry:
        ledger = api_gen.CostLedger(P["ledger"],
                                    extra_paths=L.all_ledgers(P["ledger"]))

    def budget_ok():
        if a.dry:
            return True
        try:
            ledger.check_budget()
        except api_gen.BudgetExceeded as e:
            print("[e13_api] %s" % e)
            return False
        if ledger.cost_usd(MODEL_ID) > a.usd_cap:
            print("[e13_api] soft cap hit: this-run $%.2f > $%.2f"
                  % (ledger.cost_usd(MODEL_ID), a.usd_cap))
            return False
        return True

    existing = {r["run_id"] for r in e11.read_jsonl(P["raw"])}
    raw_by_id = {r["run_id"]: r for r in e11.read_jsonl(P["raw"])}
    lock = threading.Lock()
    stop = {"halt": False}

    def check_stop():
        with lock:
            if stop["halt"]:
                return True
        if not budget_ok():
            with lock:
                stop["halt"] = True
            return True
        return False

    # -------------------------------------------------- golds (live only)
    golds = {}
    for cell in a.cells:
        if a.dry:
            for p in worlds[cell]:
                golds[(cell, p["program_id"])] = {
                    "continuation": L.mock_gold_full(p), "mock": True}
            continue

        def gold_work(p, cell=cell):
            if check_stop():
                return
            r = grid(cell, p["program_id"], a.seed)
            if r in existing:
                return
            um = tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))
            out = _call_prefill(cfg_for(0.0, a.max_new_gold), um,
                                tf.GOLD_PREFILL, ledger)
            g = {"run_id": r, "program_id": p["program_id"], "cell": cell,
                 "condition": "gold", "continuation": out.get("text"),
                 "failed_generation": out.get("text") is None,
                 "output_tokens": out.get("output_tokens"),
                 "created_at": e11.now_iso()}
            with lock:
                e11.append_jsonl(P["raw"], g)
                existing.add(r)
                raw_by_id[r] = g
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(gold_work, worlds[cell]))
        ledger.save()
        for p in worlds[cell]:
            g = raw_by_id.get(grid(cell, p["program_id"], a.seed))
            if g and not g.get("failed_generation"):
                golds[(cell, p["program_id"])] = g

    # ------------------------------- gold filter + manifest + projections
    manifest_rows = e11.read_jsonl(P["manifest"])
    man_ids = {m["run_id"] for m in manifest_rows}
    cohort = {}
    examples = defaultdict(list)
    jobs_by_arm = defaultdict(list)          # family -> [(tok_est, n_calls)]
    prog_by = {}
    for cell in a.cells:
        elig = []
        for p in sorted(worlds[cell], key=lambda x: x["program_id"]):
            prog_by[(cell, p["program_id"])] = p
            g = golds.get((cell, p["program_id"]))
            if g is None:
                continue
            gfull = g["continuation"] if g.get("mock") \
                else tf.GOLD_PREFILL + g["continuation"]
            if inj_mod.gold_solve_eval(p, gfull)["solved"]:
                elig.append((p, gfull))
        cohort[cell] = {"tested": len(worlds[cell]), "eligible": len(elig),
                        "kept": min(len(elig), a.n_target)}
        for p, gfull in elig[:a.n_target]:
            um = tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))
            for fam in L.CELL_FAMILIES[cell]:
                mid = e11.sha(["E13apiman", cell, p["program_id"], fam, a.seed])
                if mid in man_ids:
                    continue
                try:
                    b = L.build_injection_e13(p, fam, gfull)
                except inj_mod.InjectError as e:
                    e11.append_jsonl(P["audit"], {
                        "stage": "injection", "cell": cell, "family": fam,
                        "program_id": p["program_id"], "rejected": True,
                        "reason": e.reason, "created_at": e11.now_iso()})
                    continue
                row = {"run_id": mid, "cell": cell, "family": fam,
                       "program_id": p["program_id"], "seed": a.seed,
                       "site_line": p["site_line"],
                       "created_at": e11.now_iso()}
                for k in ("prefix_text", "expected_lines_from", "cf",
                          "planted_var", "planted_value", "true_value",
                          "force_after_line", "injected_trace_line",
                          "original_trace_line"):
                    row[k] = b.get(k)
                if b.get("probe"):
                    row.update({"probe": True,
                                "probe_question": b["probe_question"],
                                "probe_answer": b["probe_answer"],
                                "prefix_family": b["prefix_family"]})
                e11.append_jsonl(P["manifest"], row)
                manifest_rows.append(row)
                man_ids.add(mid)
                tok = L.est_tokens(um) + L.est_tokens(b["prefix_text"])
                jobs_by_arm[fam].append((tok, 1 + a.R))
                if len(examples[fam]) < 2:
                    if b.get("probe"):
                        examples[fam].append({
                            "program_id": p["program_id"], "cell": cell,
                            "api_messages": L.probe_messages(
                                um, b["prefix_text"], b["probe_question"]),
                            "probe_answer": b["probe_answer"]})
                    else:
                        examples[fam].append({
                            "program_id": p["program_id"], "cell": cell,
                            "api_messages": [
                                {"role": "user", "content": um},
                                {"role": "assistant (prefill, trailing ws "
                                         "stripped for the request)",
                                 "content": b["prefix_text"]}]})
    L.dump_examples(P["examples"], dict(examples))
    print("[e13_api] cohort: %s" % json.dumps(cohort))
    print("[e13_api] manifest: %d rows by-family=%s" %
          (len(manifest_rows),
           json.dumps(dict(Counter(m["family"] for m in manifest_rows)))))
    # gold projection (live regenerates golds; in dry we project the full pool)
    gold_jobs = {"__gold__": [(L.est_tokens(
        tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))), 1)
        for cell in a.cells for p in worlds[cell]]}
    proj_g = L.project_spend("e13_api golds", gold_jobs, PRICE_IN, PRICE_OUT,
                             450)
    proj_c = L.project_spend("e13_api continuations (greedy + R=%d)" % a.R,
                             jobs_by_arm, PRICE_IN, PRICE_OUT, 300)
    e11.write_json(P["projection"], {"golds": proj_g, "continuations": proj_c,
                                     "total_usd_est": round(
                                         proj_g["total_usd_est"] +
                                         proj_c["total_usd_est"], 2)})
    if a.dry:
        e11.write_json(P["status"], {"dry": True, "cohort": cohort,
                                     "manifest_rows": len(manifest_rows),
                                     "updated_at": e11.now_iso()})
        print("[e13_api:dry] DONE (zero API calls made)")
        return

    # -------------------------------------------------- continuations
    cont_jobs = []
    for m in manifest_rows:
        p = prog_by[(m["cell"], m["program_id"])]
        um = tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))
        for roll in range(0, a.R + 1):
            r = rid(m["cell"], m["program_id"], m["family"], roll, a.seed)
            if r not in existing:
                cont_jobs.append((m, um, roll, r))

    def cont_work(job):
        m, um, roll, r = job
        if check_stop():
            return
        temp = 0.0 if roll == 0 else a.temp
        if m.get("probe"):
            out = _call_probe(cfg_for(temp, L.PROBE_MAX_TOKENS), um,
                              m["prefix_text"], m["probe_question"], ledger)
        else:
            out = _call_prefill(cfg_for(temp, a.max_new_cont), um,
                                m["prefix_text"], ledger)
        rec = {"run_id": r, "program_id": m["program_id"], "cell": m["cell"],
               "family": m["family"], "condition": "cont", "rollout": roll,
               "continuation": out.get("text"),
               "failed_generation": out.get("text") is None,
               "output_tokens": out.get("output_tokens"),
               "created_at": e11.now_iso()}
        with lock:
            e11.append_jsonl(P["raw"], rec)
            existing.add(r)
            raw_by_id[r] = rec
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(cont_work, cont_jobs))
    ledger.save()

    # -------------------------------------------------- validate + summarize
    if os.path.exists(P["validated"]):
        os.remove(P["validated"])
    validated = []
    for m in manifest_rows:
        p = prog_by[(m["cell"], m["program_id"])]
        b = dict(m)
        b["probe"] = bool(m.get("probe"))
        for roll in range(0, a.R + 1):
            r = raw_by_id.get(rid(m["cell"], m["program_id"], m["family"],
                                  roll, a.seed))
            rec = {"program_id": m["program_id"], "cell": m["cell"],
                   "family": m["family"], "rollout": roll,
                   "planted_value": m["planted_value"],
                   "true_value": m["true_value"], "model": MODEL_ID,
                   "run_id": (r or {}).get("run_id"),
                   "output_tokens": (r or {}).get("output_tokens")}
            if r is None or r.get("failed_generation"):
                rec.update({"generation_failed": True, "label": "unresolved"})
                if b["probe"]:
                    rec["probe_correct"] = False
            else:
                rec["generation_failed"] = False
                rec.update(L.validate_row(p, b, r["continuation"]))
                if b["probe"]:
                    rec["probe_raw_answer"] = r["continuation"]
            validated.append(rec)
            e11.append_jsonl(P["validated"], rec)

    cells = {}
    for cell in a.cells:
        crows = [r for r in validated if r["cell"] == cell]
        cells[cell] = {
            "all": L.summarize_families(crows, seed=a.seed),
            "greedy": L.summarize_families(
                [r for r in crows if r["rollout"] == 0], seed=a.seed),
            "sampled": L.summarize_families(
                [r for r in crows if r["rollout"] > 0], seed=a.seed)}
    summ = {"experiment": "E13_MATCHED_CONTROLS_haiku_prefill",
            "model": MODEL_ID, "created_at": e11.now_iso(),
            "cohort": cohort, "cells": cells,
            "this_run_usd": round(ledger.cost_usd(MODEL_ID), 4),
            "stats_note": ("Wilson95 + program-cluster bootstrap95 "
                           "(n_boot=1000, cluster=program_id); greedy=rollout "
                           "0, sampled=1..%d at T=%.1f" % (a.R, a.temp))}
    e11.write_json(P["summary"], summ)
    ledger.save()
    e11.write_json(P["status"], {"done": True, "cohort": cohort,
                                 "this_run_usd": summ["this_run_usd"],
                                 "updated_at": e11.now_iso()})
    print(json.dumps({"cohort": cohort,
                      "this_run_usd": summ["this_run_usd"]}, indent=2))


if __name__ == "__main__":
    main()
