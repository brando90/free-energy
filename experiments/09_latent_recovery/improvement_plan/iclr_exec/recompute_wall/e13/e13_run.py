"""E13 open-model vLLM driver (llama8b + qwen7b), TRUE token-level prefill.

ADDITIVE runner: reuses e11_run.VllmBackend / vllm_gen / inject / validate BY
IMPORT; injection dispatch and new-transform mechanics live in e13_lib
(contract: RW/e13/README.md). Worlds are the dry-run-audited pools in
RW/e13/worlds/ (never regenerated here).

Design (matches the E11 open arms for Fig-3 comparability):
  * gold: own-trace per world, prefill "line 1:", greedy
  * gold double-filter (inject.gold_solve_eval), first --n-target eligible
    worlds kept per cell
  * continuations: rollout 0 greedy + rollouts 1..R at T=--temp (default
    R=8, temp 0.7 -- R=9 total rollouts, E11 spec)
  * probe (k1_probe): same perturbed prefix (prefix_family=depth_k1_bare),
    trace-continuation request REPLACED by the direct question as a follow-up
    user turn; scored by first-integer match; raw replies retained
  * scoring: validate.classify_run (existing path) via e13_lib.validate_row
  * summaries: Wilson + program-cluster bootstrap (e11_run helpers)

GPUs: --gpus is REQUIRED for live runs and is written into
CUDA_VISIBLE_DEVICES before any vllm import. Launch policy for this batch:
GPUs 3, 6, 7 ONLY (0,1,2,4,5 carry another user's vLLM servers). tp=1: one
model fits one GPU; parallelize models by separate invocations, e.g.
--gpus 3 for llama8b and --gpus 6 for qwen7b.

Usage (one model per invocation; resume-safe on run_id):
  python e13_run.py --model llama8b --gpus 3 --out-dir results/E13_open_llama8b
  python e13_run.py --model qwen7b  --gpus 6 --out-dir results/E13_open_qwen7b
  python e13_run.py --model llama8b --dry --out-dir results/E13_dry_llama8b
--dry assembles every prompt (interpreter-true stand-in golds, labelled) and
writes example_prompts.json + projected call counts; NO model/GPU use.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e13_lib as L                        # noqa: E402  (sets e11_run/expg/exph paths)
import e11_run as e11                      # noqa: E402
import trace_format as tf                  # noqa: E402
import inject as inj_mod                   # noqa: E402

MODEL_MAP = {
    # llama8b: NousResearch mirror -- meta-llama/Llama-3.1-8B-Instruct in the
    # local HF cache is a gated stub; the E11 runs used the NousResearch mirror
    # (pinned revision in e11_run.PINNED). Cache: $HF_HOME (tooling/env.sh).
    "llama8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}


def rid(model_tag, cell, pid, family, roll, seed):
    return e11.sha(["E13", model_tag, cell, pid, family, roll, seed])


def grid(model_tag, cell, pid, seed):
    return e11.sha(["E13", model_tag, cell, pid, "gold", seed])


def paths(out_dir):
    return {k: os.path.join(out_dir, v) for k, v in {
        "raw": "raw_generations.jsonl",
        "manifest": "manifest.jsonl",
        "validated": "validated_outputs.jsonl",
        "summary": "summary_tables.json",
        "run_manifest": "run_manifest.json",
        "audit": "generation_audit.jsonl",
        "examples": "example_prompts.json",
        "status": "queue_status.json"}.items()}


def write_run_manifest(P, a, model_id, revision, world_meta):
    man = {
        "experiment": "E13_MATCHED_CONTROLS_open",
        "runner": "e13_run.py", "created_at": e11.now_iso(),
        "model_tag": a.model, "model_id": model_id, "revision": revision,
        "backend": "vllm true token-level prefill (vllm_gen; prompt_token_ids)",
        "gpus_cuda_visible_devices": a.gpus, "dry": bool(a.dry),
        "cells": a.cells, "families": {c: L.CELL_FAMILIES[c] for c in a.cells},
        "n_target_per_cell": a.n_target, "R": a.R, "temp": a.temp,
        "seed": a.seed,
        "decoding": {"greedy_rollout": 0, "sampled_rollouts": a.R,
                     "max_new_gold": a.max_new_gold,
                     "max_new_cont": a.max_new_cont,
                     "probe_max_tokens": L.PROBE_MAX_TOKENS},
        "worlds": world_meta,
        "prompt_sha256": tf.prompt_sha256(), "gold_prefill": tf.GOLD_PREFILL,
        "injection_contract": "RW/e13/README.md (Build 1/2, binding)",
        "probe_note": ("k1_probe prefix_family=depth_k1_bare (bare, NOT "
                       "full_sum) -- interpretation choice flagged for Review; "
                       "question delivered as a follow-up user turn after the "
                       "closed assistant turn holding the perturbed prefix"),
        "conditional_gate": ("families %s are CONDITIONAL on the parse-sanity "
                             "gate reported per-family in summary_tables.json "
                             "(spec E13a-grid)" % (L.CONDITIONAL_FAMILIES,)),
        "gold_policy": ("dry mode uses interpreter-true stand-in golds "
                        "(labelled); live runs regenerate own-trace golds per "
                        "model exactly like E11"),
        "flag_channel": "doubt_lex frozen lexicon (validate.DOUBT_LEX_RE)",
    }
    e11.write_json(P["run_manifest"], man)
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_MAP))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpus", default=None,
                    help="CUDA_VISIBLE_DEVICES value (REQUIRED live; this "
                         "batch: pick from 3,6,7 ONLY)")
    ap.add_argument("--cells", default=",".join(L.ALL_CELLS))
    ap.add_argument("--n-target", type=int, default=150)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-gold", type=int, default=512)
    ap.add_argument("--max-new-cont", type=int, default=384)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true",
                    help="assemble all prompts, write examples, NO model/GPU")
    a = ap.parse_args()
    a.cells = [c for c in a.cells.split(",") if c]
    for c in a.cells:
        if c not in L.CELL_FAMILIES:
            raise SystemExit("unknown cell %s" % c)
    model_id = MODEL_MAP[a.model]
    revision = e11.PINNED.get(model_id)

    if not a.dry:
        if not a.gpus:
            raise SystemExit("--gpus is required for live runs "
                             "(this batch: 3, 6 or 7 ONLY)")
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus
        os.environ.setdefault("EXPG_LLM_MAX_MODEL_LEN", "8192")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    os.makedirs(a.out_dir, exist_ok=True)
    P = paths(a.out_dir)

    worlds, world_meta = {}, {}
    for cell in a.cells:
        rows, wpath = L.load_worlds(cell)
        worlds[cell] = rows
        world_meta[cell] = {"path": wpath, "n_worlds": len(rows),
                            "sha256": L.sha256_file(wpath)}
    write_run_manifest(P, a, model_id, revision, world_meta)

    backend = None
    if not a.dry:
        backend = e11.VllmBackend(model_id, revision)

    existing = {r["run_id"] for r in e11.read_jsonl(P["raw"])}

    def emit(row):
        e11.append_jsonl(P["raw"], row)
        existing.add(row["run_id"])

    # ------------------------------------------------ phase 1: golds per cell
    golds = {}                                  # (cell, pid) -> gold row
    for cell in a.cells:
        jobs = []
        for p in worlds[cell]:
            r = grid(a.model, cell, p["program_id"], a.seed)
            um = tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))
            jobs.append((p, r, um))
        pending = [(p, r, um) for p, r, um in jobs if r not in existing]
        if a.dry:
            print("[e13_run:dry] %s golds: %d prompts (stand-in interpreter "
                  "gold used downstream)" % (cell, len(jobs)))
        else:
            for i in range(0, len(pending), a.batch):
                chunk = pending[i:i + a.batch]
                ids = [backend.render_ids(um, tf.GOLD_PREFILL)
                       for _p, _r, um in chunk]
                outs = backend.greedy(ids, a.max_new_gold)
                for (p, r, _um), texts in zip(chunk, outs):
                    emit({"run_id": r, "program_id": p["program_id"],
                          "cell": cell, "condition": "gold", "rollout": 0,
                          "continuation": texts[0],
                          "failed_generation": texts[0] is None,
                          "created_at": e11.now_iso()})
                print("[e13_run] %s gold %d/%d" % (cell, i + len(chunk),
                                                   len(pending)), flush=True)
        raw = {r["run_id"]: r for r in e11.read_jsonl(P["raw"])}
        for p in worlds[cell]:
            if a.dry:
                golds[(cell, p["program_id"])] = {
                    "continuation": L.mock_gold_full(p), "mock": True}
            else:
                g = raw.get(grid(a.model, cell, p["program_id"], a.seed))
                if g and not g.get("failed_generation"):
                    golds[(cell, p["program_id"])] = g

    # ------------------------------------- phase 2: gold filter + manifest
    manifest_rows = e11.read_jsonl(P["manifest"])
    man_ids = {m["run_id"] for m in manifest_rows}
    cohort = {}
    examples = defaultdict(list)
    for cell in a.cells:
        elig = []
        for p in sorted(worlds[cell], key=lambda x: x["program_id"]):
            g = golds.get((cell, p["program_id"]))
            if g is None:
                continue
            gfull = g["continuation"] if g.get("mock") \
                else tf.GOLD_PREFILL + g["continuation"]
            ev = inj_mod.gold_solve_eval(p, gfull)
            if ev["solved"]:
                elig.append((p, gfull))
        cohort[cell] = {"tested": len(worlds[cell]), "eligible": len(elig),
                        "kept": min(len(elig), a.n_target)}
        for p, gfull in elig[:a.n_target]:
            for fam in L.CELL_FAMILIES[cell]:
                mid = e11.sha(["E13man", a.model, cell, p["program_id"], fam,
                               a.seed])
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
                       "program_id": p["program_id"],
                       "site_line": p["site_line"], "seed": a.seed,
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
                if len(examples[fam]) < 2:
                    um = tf.make_user_message(
                        tf.make_listing_text(p["stmt_texts"]))
                    if b.get("probe"):
                        examples[fam].append({
                            "program_id": p["program_id"], "cell": cell,
                            "messages": L.probe_messages(
                                um, b["prefix_text"], b["probe_question"]),
                            "probe_answer": b["probe_answer"]})
                    else:
                        examples[fam].append({
                            "program_id": p["program_id"], "cell": cell,
                            "messages": [
                                {"role": "user", "content": um},
                                {"role": "assistant (token-level prefill)",
                                 "content": b["prefix_text"]}]})
    L.dump_examples(P["examples"], dict(examples))
    print("[e13_run] cohort: %s" % json.dumps(cohort))
    print("[e13_run] manifest: %d rows by-family=%s" %
          (len(manifest_rows),
           json.dumps(dict(Counter(m["family"] for m in manifest_rows)))))

    n_calls = len(manifest_rows) * (1 + a.R)
    print("[e13_run] projected GPU calls (this model): %d gold + %d cont "
          "(greedy+R=%d) = %d total prompts; $0 API spend (local vLLM)"
          % (sum(cohort[c]["tested"] for c in a.cells),
             n_calls, a.R,
             n_calls + sum(cohort[c]["tested"] for c in a.cells)))
    if a.dry:
        e11.write_json(P["status"], {
            "dry": True, "cohort": cohort,
            "manifest_rows": len(manifest_rows),
            "projected_prompts": n_calls, "updated_at": e11.now_iso()})
        print("[e13_run:dry] DONE (no model/GPU/API use)")
        return

    # ------------------------------------------------ phase 3: continuations
    prog_by = {(c, p["program_id"]): p for c in a.cells for p in worlds[c]}

    def cont_ids(m):
        p = prog_by[(m["cell"], m["program_id"])]
        um = tf.make_user_message(tf.make_listing_text(p["stmt_texts"]))
        if m.get("probe"):
            msgs = L.probe_messages(um, m["prefix_text"], m["probe_question"])
            return backend.vg._chat_ids(backend.tok, msgs)
        return backend.render_ids(um, m["prefix_text"])

    # greedy rollout 0
    todo = [m for m in manifest_rows
            if rid(a.model, m["cell"], m["program_id"], m["family"], 0,
                   a.seed) not in existing]
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        ids = [cont_ids(m) for m in chunk]
        maxn = [L.PROBE_MAX_TOKENS if m.get("probe") else a.max_new_cont
                for m in chunk]
        outs = backend.greedy(ids, max(maxn))
        for m, texts in zip(chunk, outs):
            emit({"run_id": rid(a.model, m["cell"], m["program_id"],
                                m["family"], 0, a.seed),
                  "program_id": m["program_id"], "cell": m["cell"],
                  "family": m["family"], "condition": "cont", "rollout": 0,
                  "continuation": texts[0],
                  "failed_generation": texts[0] is None,
                  "created_at": e11.now_iso()})
        print("[e13_run] cont-greedy %d/%d" % (i + len(chunk), len(todo)),
              flush=True)

    # sampled rollouts 1..R (n=R in one SamplingParams, E11 pattern)
    if a.R > 0:
        todo = [m for m in manifest_rows
                if rid(a.model, m["cell"], m["program_id"], m["family"], 1,
                       a.seed) not in existing]
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            ids = [cont_ids(m) for m in chunk]
            outs = backend.sampled(ids, a.max_new_cont, a.R, a.temp, a.seed)
            for m, texts in zip(chunk, outs):
                for roll in range(1, a.R + 1):
                    t = texts[roll - 1] if roll - 1 < len(texts) else None
                    emit({"run_id": rid(a.model, m["cell"], m["program_id"],
                                        m["family"], roll, a.seed),
                          "program_id": m["program_id"], "cell": m["cell"],
                          "family": m["family"], "condition": "cont",
                          "rollout": roll, "continuation": t,
                          "failed_generation": t is None,
                          "created_at": e11.now_iso()})
            print("[e13_run] cont-sampled %d/%d" % (i + len(chunk), len(todo)),
                  flush=True)

    # ------------------------------------------------ phase 4: validate
    raw = {r["run_id"]: r for r in e11.read_jsonl(P["raw"])}
    if os.path.exists(P["validated"]):
        os.remove(P["validated"])
    validated = []
    for m in manifest_rows:
        p = prog_by[(m["cell"], m["program_id"])]
        b = dict(m)
        b["probe"] = bool(m.get("probe"))
        for roll in range(0, a.R + 1):
            r = raw.get(rid(a.model, m["cell"], m["program_id"], m["family"],
                            roll, a.seed))
            rec = {"program_id": m["program_id"], "cell": m["cell"],
                   "family": m["family"], "rollout": roll,
                   "planted_value": m["planted_value"],
                   "true_value": m["true_value"], "model": a.model,
                   "run_id": (r or {}).get("run_id")}
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

    # ------------------------------------------------ phase 5: summarize
    cells = {}
    for cell in a.cells:
        crows = [r for r in validated if r["cell"] == cell]
        cells[cell] = {
            "all": L.summarize_families(crows, seed=a.seed),
            "greedy": L.summarize_families(
                [r for r in crows if r["rollout"] == 0], seed=a.seed),
            "sampled": L.summarize_families(
                [r for r in crows if r["rollout"] > 0], seed=a.seed)}
    summ = {"experiment": "E13_MATCHED_CONTROLS_open", "model_tag": a.model,
            "model_id": model_id, "created_at": e11.now_iso(),
            "cohort": cohort, "cells": cells,
            "stats_note": ("Wilson95 + program-cluster bootstrap95 "
                           "(n_boot=1000, cluster=program_id); greedy=rollout "
                           "0, sampled=rollouts 1..%d at T=%.1f"
                           % (a.R, a.temp))}
    e11.write_json(P["summary"], summ)
    e11.write_json(P["status"], {"done": True, "cohort": cohort,
                                 "updated_at": e11.now_iso()})
    print("[e13_run] DONE -> %s" % P["summary"])


if __name__ == "__main__":
    main()
