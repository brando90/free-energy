"""EXPE gold-cohort expansion (cohort prep only -- runs NO perturbation arms).

Purpose: the legacy gold cohort (results/gold, 168 gold-validated own-trace
rollouts) admits only 27 problems for EXPE's six-arm paired design at mid.
This script expands the own-trace gold cohort from data/prontoqa_ood
(ProofsOnly + Composed files, the EXPA/EXPB source convention) until >= 150
problems admit all six EXPE arms at mid, re-using the EXPE runner's own
eligibility ladder and fail-closed audits for the availability decision.

Protocol identical to the legacy golds (collect_gold.py / common.py):
chat prompt INSTR + FEWSHOT + "Q: {question} Prove: {target}\nA:", NO prefill,
greedy, max_new=256, Qwen2.5-7B-Instruct pinned revision; double filter =
solved AND validator valid_rederivation, entity = first word of the trace.

Screening (disclosed; structural necessary conditions + a quality ordering):
  hard screen  : question not in the legacy pilot set; question unique;
                 entity + parsable target; dataset CoT has >= 4 derived steps
                 (EXPB's own-cohort threshold); >= 2 categories unentailed for
                 the entity (a trace-independent NECESSARY condition for the
                 six-arm set: F + in-vocab FREQ antecedent).
  ordering     : unentailed-pool size DESC, then problem id -- richer worlds
                 first, to maximize tier-A/tier-B vocabulary composition.

Stages:
  screen | generate --n N | validate | availability | report | run-all
Never writes to results/gold/, src/, or data/. Output:
  $EXP/results/EXPE_GOLD_EXPANSION/ + this directory's logs.
"""
import argparse
import datetime as _dt
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
EXPE_DIR = os.environ.get("LR_EXPE_DIR") or os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, EXPE_DIR)

from expe_evidence_mover import (  # noqa: E402
    ARMS,
    EXP,
    FEWSHOT,
    INSTR,
    TOOLING,
    append_jsonl,
    build_problem_arms,
    injection_points,
    norm,
    now_iso,
    question_world,
    read_jsonl,
    resolve_model_revision,
    split_sentences,
    world_categories,
    write_json,
)
from validator import derivable, parse_fact, validate_continuation  # noqa: E402

DATA_DIR = os.path.join(EXP, "data", "prontoqa_ood")
PILOT = os.path.join(EXP, "data", "pilot.jsonl")
DEFAULT_OUT = os.path.join(EXP, "results", "EXPE_GOLD_EXPANSION")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 256          # legacy gold convention (common.greedy default)
TARGET_AVAILABLE = 150 # stop condition: problems admitting all six arms at mid
MIN_UNENTAILED = 2     # necessary condition: F + in-vocab FREQ antecedent
MIN_DERIVED_STEPS = 4  # EXPB own-cohort threshold (proxy for injectable trace)


def out_paths(out_dir):
    return {
        "screened": os.path.join(out_dir, "screened_candidates.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "validated": os.path.join(out_dir, "validated.jsonl"),
        "cohort": os.path.join(out_dir, "cohort.jsonl"),
        "avail_audit": os.path.join(out_dir, "availability_audit.jsonl"),
        "availability": os.path.join(out_dir, "availability.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "report": os.path.join(out_dir, "GOLD_EXPANSION_REPORT.md"),
    }


def derived_steps(cot, entity):
    return [i for i, s in enumerate(cot)
            if s.split() and s.split()[0] == entity]


def iter_pool():
    files = (sorted(glob.glob(os.path.join(DATA_DIR, "*ProofsOnly*.json")))
             + sorted(glob.glob(os.path.join(DATA_DIR, "*Composed*.json"))))
    for path in files:
        data = json.load(open(path))
        for key, ex in data.items():
            for sub in ["test_example"] + [f"in_context_example{i}" for i in range(8)]:
                q = ex.get(sub)
                if not q or "chain_of_thought" not in q:
                    continue
                yield (f"{os.path.basename(path)}::{key}::{sub}",
                       os.path.basename(path), q)


def screen(args):
    os.makedirs(args.out_dir, exist_ok=True)
    paths = out_paths(args.out_dir)
    legacy_qs = {r["question"] for r in read_jsonl(PILOT)}
    seen = set()
    ladder = Counter()
    rows = []
    for pid, fname, q in iter_pool():
        ladder["pool_rows"] += 1
        question = q["question"]
        if question in legacy_qs:
            ladder["excluded_legacy_pilot_question"] += 1
            continue
        if question in seen:
            ladder["duplicate_question"] += 1
            continue
        seen.add(question)
        target = q["query"].replace("Prove:", "").strip()
        cot = [" ".join(str(s).split()) for s in q["chain_of_thought"] if str(s).strip()]
        tf = parse_fact(target)
        entity = tf[0] if tf else (cot[0].split()[0] if cot else None)
        if not entity or not tf:
            ladder["no_entity_or_unparsable_target"] += 1
            continue
        if len(derived_steps(cot, entity)) < MIN_DERIVED_STEPS:
            ladder["too_few_dataset_derived_steps"] += 1
            continue
        w = question_world(question, entity)
        # The world must be FULLY in-grammar for the validator: disjunctive
        # Composed variants ("Everything that is an X or a Y ...") unparse, and
        # unparsable rules also silently inflate the unentailed-category count
        # (round-1 lesson: 0/600 gold-valid without this screen).
        if w["unparsed"]:
            ladder["question_not_fully_parsable"] += 1
            continue
        gv = validate_continuation(question, [], None, " ".join(cot), target, entity)
        if gv["class"] != "valid_rederivation":
            ladder["dataset_cot_not_validator_valid"] += 1
            continue
        unent = [c for c in world_categories(question)
                 if not derivable(("cat", c), w["premises"], w["reach"])]
        if len(unent) < MIN_UNENTAILED:
            ladder["too_few_unentailed_categories"] += 1
            continue
        ladder["screened_in"] += 1
        rows.append({"problem_id": pid, "source_file": fname, "question": question,
                     "target": target, "entity": entity,
                     "n_dataset_derived_steps": len(derived_steps(cot, entity)),
                     "n_unentailed_categories": len(unent),
                     "n_categories": len(world_categories(question))})
    rows.sort(key=lambda r: (-r["n_unentailed_categories"], r["problem_id"]))
    with open(paths["screened"], "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    meta = {"experiment": "EXPE_GOLD_EXPANSION", "created_at": now_iso(),
            "model": args.model, "max_new_tokens": MAX_NEW,
            "prompt_convention": "collect_gold.py / common.py (no prefill, greedy)",
            "screen": {"min_unentailed": MIN_UNENTAILED,
                       "min_derived_steps": MIN_DERIVED_STEPS,
                       "ordering": "n_unentailed_categories DESC (tier-A preference)",
                       "ladder": dict(ladder)},
            "gpu_seconds_generate": 0.0,
            "note": "cohort prep only; no perturbation arms run; legacy results/gold untouched"}
    write_json(paths["metadata"], meta)
    print(json.dumps({"screen_ladder": dict(ladder),
                      "screened_candidates": len(rows),
                      "unentailed_histogram": dict(sorted(Counter(
                          r["n_unentailed_categories"] for r in rows).items()))},
                     indent=2))


def generate(args):
    paths = out_paths(args.out_dir)
    cands = read_jsonl(paths["screened"])
    done = {r["problem_id"] for r in read_jsonl(paths["raw"])}
    todo = [c for c in cands if c["problem_id"] not in done][: args.n]
    print(f"[gold-expansion generate] todo={len(todo)} (already done={len(done)})", flush=True)
    if not todo:
        return
    sys.path.insert(0, TOOLING)
    import vllm_gen
    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise RuntimeError(f"could not pin model revision: {err}")
    tok = vllm_gen.get_tokenizer(args.model, revision)
    rows = [{"prompt_token_ids": vllm_gen.render_prompt_ids(
        tok, c["question"], c["target"], None, INSTR, FEWSHOT)} for c in todo]
    t0 = time.time()
    outs = vllm_gen.generate_continuations(args.model, rows,
                                           max_new_tokens=MAX_NEW, revision=revision)
    gpu_s = time.time() - t0
    eos = set(vllm_gen.eos_token_ids(args.model, tok, revision))
    for c, r, o in zip(todo, rows, outs):
        toks = list(o["token_ids"])
        while toks and toks[-1] in eos:
            toks = toks[:-1]
        append_jsonl(paths["raw"], {
            "problem_id": c["problem_id"], "source_file": c["source_file"],
            "model": args.model, "model_revision": revision, "backend": "vllm",
            "gen_text": o["text"].strip(), "finish_reason": o["finish_reason"],
            "prompt_token_count": len(r["prompt_token_ids"]),
            "gen_token_count": len(toks),
            "decoding": {"do_sample": False, "temperature": 0.0,
                         "max_new_tokens": MAX_NEW},
            "created_at": now_iso()})
    meta = json.load(open(paths["metadata"]))
    meta["gpu_seconds_generate"] = round(meta.get("gpu_seconds_generate", 0.0) + gpu_s, 1)
    meta["model_revision"] = revision
    write_json(paths["metadata"], meta)
    print(f"DONE generate n={len(todo)} gpu_batch_seconds={gpu_s:.1f}", flush=True)


def validate(args):
    paths = out_paths(args.out_dir)
    cands = {c["problem_id"]: c for c in read_jsonl(paths["screened"])}
    ladder = Counter()
    with open(paths["validated"], "w") as vh, open(paths["cohort"], "w") as ch:
        for r in read_jsonl(paths["raw"]):
            c = cands.get(r["problem_id"])
            if c is None:
                ladder["superseded_by_rescreen"] += 1
                continue
            steps = split_sentences(r["gen_text"])
            ladder["rollouts"] += 1
            is_solved = bool(steps) and norm(steps[-1]) == norm(c["target"])
            entity = steps[0].split()[0] if steps and steps[0].split() else None
            rec = {"problem_id": r["problem_id"], "source_file": r["source_file"],
                   "solved": is_solved, "entity": entity,
                   "gen_token_count": r.get("gen_token_count"),
                   "finish_reason": r.get("finish_reason")}
            if not is_solved:
                ladder["not_solved"] += 1
                rec["gold_valid"] = False
                rec["reason"] = "not_solved"
            else:
                gv = validate_continuation(c["question"], [], None, r["gen_text"],
                                           c["target"], entity)
                rec["validator_class"] = gv["class"]
                rec["gold_valid"] = gv["class"] == "valid_rederivation"
                if rec["gold_valid"]:
                    ladder["gold_valid"] += 1
                    ch.write(json.dumps({
                        "problem_id": r["problem_id"],
                        "source_file": r["source_file"],
                        "question": c["question"], "target": c["target"],
                        "entity": entity, "gen_text": r["gen_text"],
                        "model": r["model"], "model_revision": r["model_revision"],
                        "n_unentailed_categories": c["n_unentailed_categories"],
                    }, sort_keys=True) + "\n")
                else:
                    ladder["not_validator_valid"] += 1
                    rec["reason"] = gv["class"]
            vh.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps({"double_filter_ladder": dict(ladder)}, indent=2))


def availability(args):
    paths = out_paths(args.out_dir)
    cohort = read_jsonl(paths["cohort"])
    ladder = Counter()
    tiers = Counter()
    per_file = Counter()
    with open(paths["avail_audit"], "w") as fh:
        for row in cohort:
            ladder["gold_valid_cohort"] += 1
            steps = split_sentences(row["gen_text"])
            pts = injection_points(steps, row["entity"])
            if pts is None:
                ladder["too_few_injection_points"] += 1
                fh.write(json.dumps({"problem_id": row["problem_id"],
                                     "eligible": False,
                                     "reason": "too_few_intermediate_entity_steps"}) + "\n")
                continue
            si = pts["mid"]
            built, diags = build_problem_arms(row["question"], steps, row["entity"],
                                              row["target"], si, row["gen_text"])
            if diags["n_falsehood_candidates"] == 0:
                ladder["no_falsehood_candidate"] += 1
                fh.write(json.dumps({"problem_id": row["problem_id"],
                                     "eligible": False,
                                     "reason": "no_cwa_false_unrefuted_category"}) + "\n")
                continue
            if built is None:
                ladder["not_all_arms_available"] += 1
                missing = [a for a in ARMS
                           if a not in diags["arm_available_under_some_F"]]
                fh.write(json.dumps({"problem_id": row["problem_id"],
                                     "eligible": False,
                                     "reason": "no_single_F_admits_all_arms",
                                     "arms_never_available": missing}) + "\n")
                continue
            ladder["admits_all_arms"] += 1
            tiers[built["vocab_tier"]] += 1
            per_file[row["source_file"]] += 1
            if built["poisoning_measurable"]:
                ladder["admits_all_arms_measurable_poisoning"] += 1
            fh.write(json.dumps({"problem_id": row["problem_id"], "eligible": True,
                                 "fcat": built["fcat"],
                                 "vocab_tier": built["vocab_tier"],
                                 "poisoning_measurable": built["poisoning_measurable"]},
                                sort_keys=True) + "\n")
    n_avail = ladder.get("admits_all_arms", 0)
    out = {
        "created_at": now_iso(),
        "position": "mid",
        "cohort_ladder": dict(ladder),
        "paired_availability_new_cohort": n_avail,
        "paired_availability_including_legacy27": n_avail + 27,
        "vocab_tier_composition": dict(tiers),
        "availability_by_source_file": dict(per_file),
        "target": TARGET_AVAILABLE,
        "target_reached": n_avail >= TARGET_AVAILABLE,
    }
    write_json(paths["availability"], out)
    print(json.dumps(out, indent=2))


def report(args):
    paths = out_paths(args.out_dir)
    meta = json.load(open(paths["metadata"]))
    avail = json.load(open(paths["availability"]))
    validated = read_jsonl(paths["validated"])
    n_roll = len(validated)
    n_valid = sum(r.get("gold_valid", False) for r in validated)
    lines = [
        "# EXPE Gold-Cohort Expansion Report",
        "",
        f"Generated: {now_iso()}",
        "",
        "Cohort prep only: NO perturbation arms were run; the legacy",
        "results/gold/ directory was not touched. Gold protocol identical to",
        "collect_gold.py (no-prefill greedy chat prompt, max_new=256) but under",
        "the vLLM backend -- new suites run wholly under vLLM (VLLM_PORT_REPORT).",
        "",
        "## Screen",
        "",
        f"- {json.dumps(meta.get('screen', {}).get('ladder', {}))}",
        f"- Hard screen: unentailed categories >= {MIN_UNENTAILED}, dataset derived "
        f"steps >= {MIN_DERIVED_STEPS}, target parses, question not in legacy pilot.",
        "- Ordering: unentailed-pool size DESC (maximizes tier-A/B vocabulary).",
        "",
        "## Rollouts and double filter",
        "",
        f"- Rollouts generated: {n_roll}",
        f"- Gold-valid (solved AND validator valid_rederivation): {n_valid} "
        f"({(n_valid / n_roll):.2f})" if n_roll else "- none",
        f"- GPU seconds (generation batches): {meta.get('gpu_seconds_generate')}",
        f"- Model: {meta.get('model')} @ {meta.get('model_revision', 'unpinned')}",
        "",
        "## Availability (EXPE six-arm ladder at mid, same audits as the runner)",
        "",
        f"- Ladder: {json.dumps(avail['cohort_ladder'])}",
        f"- PAIRED AVAILABILITY (new cohort): {avail['paired_availability_new_cohort']} "
        f"(target {avail['target']}; reached: {avail['target_reached']})",
        f"- Including the 27 legacy problems: "
        f"{avail['paired_availability_including_legacy27']}",
        f"- Vocabulary tiers: {json.dumps(avail['vocab_tier_composition'])}",
        f"- By source file: {json.dumps(avail['availability_by_source_file'])}",
        "",
        "## Notes",
        "",
        "- cohort.jsonl holds the expanded own-trace gold cohort (question, target,",
        "  entity, model trace); the EXPE runner can consume it after the PLAN2",
        "  amendment that re-scopes the EXPE cohort is timestamped.",
        "- The screen conditions are structural necessary conditions for the",
        "  six-arm set plus a disclosed richness ordering; they do not condition",
        "  on model behavior beyond the pre-registered double filter.",
        "",
    ]
    with open(paths["report"], "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EXPE gold cohort expansion (prep only)")
    p.add_argument("stage", choices=["screen", "generate", "validate",
                                     "availability", "report", "run-all"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=600, help="rollouts per generate round")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default="auto")
    p.add_argument("--allow-unpinned-model", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stages = {"screen": screen, "generate": generate, "validate": validate,
              "availability": availability, "report": report}
    if args.stage == "run-all":
        for s in ("screen", "generate", "validate", "availability", "report"):
            stages[s](args)
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()
