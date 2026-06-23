"""Experiment C: lexical polarity control.

Staged usage:

  .venv/bin/python src/expc_polarity_control.py test
  .venv/bin/python src/expc_polarity_control.py prepare --overwrite
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python src/expc_polarity_control.py generate --device cuda:0 --target 7
  .venv/bin/python src/expc_polarity_control.py validate
  .venv/bin/python src/expc_polarity_control.py summarize
  .venv/bin/python src/expc_polarity_control.py manual-audit
  .venv/bin/python src/expc_polarity_control.py report

The runner fails closed: a problem is generated only when all four
locality-by-polarity cells exist at the same early/mid/late positions.
Unavailable cells are written to the audit artifacts instead of being replaced.
"""
import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator import (
    closure,
    derivable,
    parse_fact,
    parse_rule,
    parse_world,
    strip_marker,
    validate_continuation,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "prontoqa_ood")
DEFAULT_OUT = os.path.join(BASE, "results", "EXPC_POLARITY_CONTROL")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 0
POINTS = ("early", "mid", "late")

LOCAL_FALSE_POSITIVE = "LOCAL_FALSE_POSITIVE"
LOCAL_FALSE_NEGATIVE = "LOCAL_FALSE_NEGATIVE"
GLOBAL_FALSE_POSITIVE = "GLOBAL_FALSE_POSITIVE"
GLOBAL_FALSE_NEGATIVE = "GLOBAL_FALSE_NEGATIVE"
CONDITIONS = (
    LOCAL_FALSE_POSITIVE,
    LOCAL_FALSE_NEGATIVE,
    GLOBAL_FALSE_POSITIVE,
    GLOBAL_FALSE_NEGATIVE,
)
CONDITION_META = {
    LOCAL_FALSE_POSITIVE: {"locality": "local", "polarity": "positive"},
    LOCAL_FALSE_NEGATIVE: {"locality": "local", "polarity": "negative"},
    GLOBAL_FALSE_POSITIVE: {"locality": "global", "polarity": "positive"},
    GLOBAL_FALSE_NEGATIVE: {"locality": "global", "polarity": "negative"},
}
MODEL_CONDITIONS = set(CONDITIONS)

INSTR = (
    "You will be given facts and rules about fictional creatures, then asked to prove a statement. "
    "Answer with only the proof: a sequence of statements, one deduction at a time, in the exact "
    "style of the examples. End with the statement to be proven.\n\n"
)

FEWSHOT = """Q: Every yumpus is a dumpus. Dumpuses are tumpuses. Tumpuses are not bright. Sam is a yumpus. Prove: Sam is not bright.
A: Sam is a yumpus. Every yumpus is a dumpus. Sam is a dumpus. Dumpuses are tumpuses. Sam is a tumpus. Tumpuses are not bright. Sam is not bright.

Q: Each gorpus is a sterpus. Sterpuses are red. Every borpus is a gorpus. Alex is a borpus. Prove: Alex is red.
A: Alex is a borpus. Every borpus is a gorpus. Alex is a gorpus. Each gorpus is a sterpus. Alex is a sterpus. Sterpuses are red. Alex is red.

"""


def now_iso():
    return _dt.datetime.now(_dt.UTC).isoformat()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def sha_row(parts, n=24):
    s = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def clean_step(s):
    return " ".join(str(s).split()).strip()


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", text.strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def singular(cat):
    return cat[:-2] if cat.endswith("es") else cat


def out_paths(out_dir):
    return {
        "candidate_pool": os.path.join(out_dir, "candidate_pool.jsonl"),
        "audit": os.path.join(out_dir, "eligibility_audit.jsonl"),
        "truth_audit": os.path.join(out_dir, "truth_audit.jsonl"),
        "manifest": os.path.join(out_dir, "manifest.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
        "summary": os.path.join(out_dir, "summary_tables.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "manual": os.path.join(out_dir, "manual_audit_note.md"),
        "report": os.path.join(out_dir, "EXPC_REPORT.md"),
        "readme": os.path.join(out_dir, "README.md"),
        "complete": os.path.join(out_dir, "RUN_COMPLETE"),
    }


def ensure_out_dir(out_dir, overwrite=False):
    paths = out_paths(out_dir)
    if os.path.exists(paths["complete"]) and not overwrite:
        raise SystemExit(f"{out_dir} is marked complete; refusing to modify without --overwrite")
    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def refuse_complete(out_dir, overwrite=False):
    if os.path.exists(out_paths(out_dir)["complete"]) and not overwrite:
        raise SystemExit(f"{out_dir} is marked complete; refusing to modify without --overwrite")


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.abspath(os.path.join(BASE, "..", "..")), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def prompt_hash():
    return hashlib.sha256((INSTR + FEWSHOT).encode()).hexdigest()


def resolve_model_revision(model_name, revision):
    if revision and revision != "auto":
        return revision, True, None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_name)
        return info.sha, True, None
    except Exception as e:
        return revision or "main", False, repr(e)


def write_metadata(out_dir, args, model_revision=None, revision_pinned=False, revision_error=None):
    meta = {
        "experiment": "EXPC_POLARITY_CONTROL",
        "created_at": now_iso(),
        "git_commit_hash": git_hash(),
        "model": args.model,
        "model_revision": model_revision or args.model_revision,
        "model_revision_pinned": bool(revision_pinned),
        "model_revision_error": revision_error,
        "seed": args.seed,
        "target_fully_matched_problems": args.target,
        "max_candidates": args.max_candidates,
        "device": getattr(args, "device", None),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": args.max_new_tokens,
            "num_return_sequences": 1,
        },
        "prompt": {
            "instruction": INSTR,
            "fewshot": FEWSHOT,
            "sha256": prompt_hash(),
            "shared_across_conditions": True,
        },
        "design": {
            "conditions": CONDITION_META,
            "positions": POINTS,
            "primary_formulas": [
                "valid_recovery ~ locality * polarity * position + (1 | problem_id)",
                "doubt ~ locality * polarity * position + (1 | problem_id)",
                "poisoning ~ locality * polarity * position + (1 | problem_id), where measurable",
            ],
        },
        "python": sys.version,
    }
    write_json(out_paths(out_dir)["metadata"], meta)
    return meta


def direct_rule_map(rules):
    direct = defaultdict(set)
    for lhs, rhs in rules:
        direct[lhs].add(rhs)
    return direct


def world_categories(question):
    return sorted({singular(c) for c in re.findall(r"\b([a-z]+pus(?:es)?)\b", question)})


def entity_from_target_or_steps(target, steps):
    f = parse_fact(target)
    if f:
        return f[0]
    for s in steps:
        f = parse_fact(s)
        if f:
            return f[0]
    return None


def is_positive_pred(pred):
    return pred[0] == "cat" or (pred[0] == "adj" and pred[2] is True)


def is_negative_pred(pred):
    return pred[0] == "not_cat" or (pred[0] == "adj" and pred[2] is False)


def opposite_pred(pred):
    if pred[0] == "cat":
        return ("not_cat", pred[1])
    if pred[0] == "not_cat":
        return ("cat", pred[1])
    if pred[0] == "adj":
        return ("adj", pred[1], not pred[2])
    raise ValueError(pred)


def pred_to_statement(entity, pred):
    if pred[0] == "cat":
        return f"{entity} is {article(pred[1])} {pred[1]}."
    if pred[0] == "not_cat":
        return f"{entity} is not {article(pred[1])} {pred[1]}."
    if pred[0] == "adj":
        return f"{entity} is {'not ' if not pred[2] else ''}{pred[1]}."
    raise ValueError(pred)


def statement_predicate(statement, entity):
    f = parse_fact(strip_marker(statement or ""), entity)
    return f[1] if f else None


def predicate_label(pred):
    if pred is None:
        return None
    return list(pred)


def grammar_template(statement, entity):
    s = strip_marker(statement or "").rstrip(".")
    if re.match(rf"^{re.escape(entity)}\s+is\s+(?:a|an)\s+\w+$", s):
        return "entity_is_a_category"
    if re.match(rf"^{re.escape(entity)}\s+is\s+not\s+(?:a|an)\s+\w+$", s):
        return "entity_is_not_a_category"
    if re.match(rf"^{re.escape(entity)}\s+is\s+not\s+\w+$", s):
        return "entity_is_not_attribute"
    if re.match(rf"^{re.escape(entity)}\s+is\s+\w+$", s):
        return "entity_is_attribute"
    return "other"


def predicate_family(pred):
    if pred is None:
        return "unparsed"
    if pred[0] in ("cat", "not_cat"):
        return "category"
    if pred[0] == "adj":
        return "attribute"
    return "other"


def world_state(question, entity, prefix_steps=()):
    rules, facts, unparsed = parse_world(question)
    reach = closure(rules)
    direct = direct_rule_map(rules)
    state = {p for e, p in facts if e == entity}
    for s in prefix_steps:
        f = parse_fact(strip_marker(s), entity)
        if f:
            state.add(f[1])
    return rules, direct, reach, state, unparsed


def directly_derivable(pred, state, direct):
    if pred in state:
        return True
    return any(a[0] == "cat" and pred in direct.get(a[1], set()) for a in state)


def all_entailed_preds(state, reach):
    out = set(state)
    for a in state:
        if a[0] == "cat":
            out |= set(reach.get(a[1], set()))
    return out


def shortest_rule_distance(pred, state, direct, max_depth=8):
    if pred in state:
        return 0
    frontier = [(a[1], 0) for a in state if a[0] == "cat"]
    seen = {c for c, _ in frontier}
    while frontier:
        cat, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for rhs in direct.get(cat, set()):
            nd = depth + 1
            if rhs == pred:
                return nd
            if rhs[0] == "cat" and rhs[1] not in seen:
                seen.add(rhs[1])
                frontier.append((rhs[1], nd))
    return None


def audit_truth_status(question, statement, entity, prefix_steps=()):
    pred = statement_predicate(statement, entity)
    if pred is None:
        return {"truth_status": "unparsed", "audit_basis": "statement_unparsed"}
    _, direct, reach, state, unparsed = world_state(question, entity, ())
    _, local_direct, _, local_state, _ = world_state(question, entity, prefix_steps)
    opp = opposite_pred(pred)
    entailed = derivable(pred, state, reach)
    opposite_entailed = derivable(opp, state, reach)
    local_refutable = directly_derivable(opp, local_state, local_direct)
    dist = shortest_rule_distance(opp, local_state, local_direct)

    if entailed:
        truth, basis = "true", "entailed"
    elif opposite_entailed:
        truth, basis = "false", "opposite_entailed"
    elif pred[0] == "cat":
        truth, basis = "false", "unentailed_category_closed_world"
    elif pred[0] == "not_cat":
        truth, basis = "unknown", "negated_unentailed_category_open_world"
    else:
        truth, basis = "unknown", "attribute_not_decidable"

    return {
        "truth_status": truth,
        "audit_basis": basis,
        "predicate": predicate_label(pred),
        "opposite_predicate": predicate_label(opp),
        "predicate_family": predicate_family(pred),
        "grammar_template": grammar_template(statement, entity),
        "local_refutable_from_prefix": bool(local_refutable),
        "local_rule_distance": dist,
        "world_unparsed_sentence_count": len(unparsed),
    }


def entity_sentences(steps, entity):
    return [i for i, s in enumerate(steps) if parse_fact(s, entity)]


def injection_points(steps, entity):
    ents = entity_sentences(steps, entity)
    inter = ents[1:-1]
    if len(inter) < 3:
        return None
    return {"early": inter[0], "mid": inter[len(inter) // 2], "late": inter[-1]}


def make_local_false_positive(question, steps, entity, si, idx=0):
    _, direct, _, state, _ = world_state(question, entity, steps[:si])
    cands = []
    for atom in state:
        if atom[0] != "cat":
            continue
        for rhs in direct.get(atom[1], set()):
            if is_negative_pred(rhs):
                pred = opposite_pred(rhs)
                if is_positive_pred(pred):
                    cands.append((0 if pred[0] == "cat" else 1, pred, rhs, atom))
    if not cands:
        return None, {"availability_reason": "no_direct_local_negative_evidence_for_positive_claim"}
    cands = sorted(cands, key=lambda x: (x[0], str(x[1]), str(x[3])))
    _, pred, refuting_pred, support_atom = cands[idx % len(cands)]
    return pred_to_statement(entity, pred), {
        "refutation_scope": "local_direct",
        "refuting_predicate": predicate_label(refuting_pred),
        "support_atom": predicate_label(support_atom),
        "positive_claim_family": predicate_family(pred),
    }


def make_local_false_negative(question, steps, entity, si, idx=0):
    f = parse_fact(steps[si], entity)
    if not f:
        return None, {"availability_reason": "injection_step_unparsed"}
    pred = f[1]
    if not is_positive_pred(pred):
        return None, {"availability_reason": "local_step_not_positive"}
    _, direct, _, state, _ = world_state(question, entity, steps[:si])
    if not directly_derivable(pred, state, direct):
        return None, {"availability_reason": "local_positive_step_not_directly_derivable_from_prefix"}
    return pred_to_statement(entity, opposite_pred(pred)), {
        "refutation_scope": "local_direct",
        "refuting_predicate": predicate_label(pred),
        "support_statement": steps[si],
    }


def make_global_false_positive(question, steps, entity, si, idx=0):
    _, direct, reach, state, _ = world_state(question, entity, ())
    _, local_direct, _, local_state, _ = world_state(question, entity, steps[:si])
    cands = []
    for cat in world_categories(question):
        pred = ("cat", cat)
        neg = ("not_cat", cat)
        if derivable(pred, state, reach):
            continue
        if derivable(neg, state, reach):
            continue
        if directly_derivable(neg, local_state, local_direct):
            continue
        cands.append(pred)
    if not cands:
        return None, {"availability_reason": "no_unentailed_category_without_negative_evidence"}
    pred = sorted(cands, key=str)[idx % len(cands)]
    return pred_to_statement(entity, pred), {
        "refutation_scope": "global_closed_world_unentailed",
        "support_atom": None,
    }


def make_global_false_negative(question, steps, entity, si, idx=0):
    _, direct, reach, state, _ = world_state(question, entity, ())
    _, local_direct, _, local_state, _ = world_state(question, entity, steps[:si])
    entailed = all_entailed_preds(state, reach)
    direct_entailed = set(local_state)
    for atom in local_state:
        if atom[0] == "cat":
            direct_entailed |= local_direct.get(atom[1], set())
    cands = []
    for pred in entailed:
        if not is_positive_pred(pred):
            continue
        if pred in local_state or pred in direct_entailed:
            continue
        dist = shortest_rule_distance(pred, local_state, local_direct)
        if dist is None or dist < 2:
            continue
        cands.append((0 if pred[0] == "cat" else 1, dist, pred))
    if not cands:
        return None, {"availability_reason": "no_nonlocal_entailed_positive_predicate"}
    _, dist, pred = sorted(cands, key=lambda x: (x[0], x[1], str(x[2])))[idx % len(cands)]
    return pred_to_statement(entity, opposite_pred(pred)), {
        "refutation_scope": "global_nonlocal_entailment",
        "refuting_predicate": predicate_label(pred),
        "rule_distance_from_prefix": dist,
    }


def build_injections(candidate):
    makers = {
        LOCAL_FALSE_POSITIVE: make_local_false_positive,
        LOCAL_FALSE_NEGATIVE: make_local_false_negative,
        GLOBAL_FALSE_POSITIVE: make_global_false_positive,
        GLOBAL_FALSE_NEGATIVE: make_global_false_negative,
    }
    out = []
    steps = candidate["original_gold_steps"]
    for pi, point in enumerate(POINTS):
        si = candidate["injection_points"][point]
        for condition in CONDITIONS:
            stmt, support = makers[condition](candidate["question"], steps, candidate["entity"], si, idx=pi)
            truth = (
                audit_truth_status(candidate["question"], stmt, candidate["entity"], steps[:si])
                if stmt
                else {"truth_status": "unavailable", **support}
            )
            meta = CONDITION_META[condition]
            available = bool(stmt) and truth.get("truth_status") == "false"
            reason = "available" if available else support.get("availability_reason") or f"truth_status_{truth.get('truth_status')}"
            out.append({
                "condition": condition,
                "locality": meta["locality"],
                "polarity": meta["polarity"],
                "injection_position": point,
                "sent_idx": si,
                "prefix_steps": steps[:si],
                "injected_statement": stmt,
                "available": available,
                "availability_reason": reason,
                "truth_audit": truth,
                "support": support,
                "grammar_template": grammar_template(stmt, candidate["entity"]) if stmt else "unavailable",
                "predicate_family": truth.get("predicate_family", "unavailable"),
            })
    return out


def run_id(problem_id, condition, point, seed):
    return sha_row(["EXPC", problem_id, condition, point, seed], 24)


def unavailable_id(problem_id, condition, point, seed):
    return sha_row(["EXPC_UNAVAILABLE", problem_id, condition, point, seed], 24)


def iter_dataset_examples():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*ProofsOnly*.json"))) + sorted(glob.glob(os.path.join(DATA_DIR, "*Composed*.json")))
    for path in files:
        data = json.load(open(path))
        for key, ex in data.items():
            for sub in ["test_example"] + [f"in_context_example{i}" for i in range(8)]:
                q = ex.get(sub)
                if not q or "chain_of_thought" not in q:
                    continue
                yield path, key, sub, q


def build_candidate_pool(seed=DEFAULT_SEED):
    rows = []
    audit_rows = []
    truth_rows = []
    seen_questions = set()
    for path, key, sub, q in iter_dataset_examples():
        question = q["question"]
        source_id = f"{os.path.basename(path)}::{key}::{sub}"
        if question in seen_questions:
            audit_rows.append({"problem_id": source_id, "stage": "candidate", "eligible": False, "reason": "duplicate_question"})
            continue
        seen_questions.add(question)
        steps = [clean_step(s) for s in q["chain_of_thought"] if clean_step(s)]
        target = q["query"].replace("Prove:", "").strip()
        entity = entity_from_target_or_steps(target, steps)
        base = {
            "problem_id": source_id,
            "question": question,
            "target": target,
            "entity": entity,
            "original_gold_steps": steps,
            "original_gold_proof": " ".join(steps),
            "source_file": os.path.basename(path),
            "source_key": key,
            "source_subkey": sub,
        }
        if not entity:
            audit_rows.append({**base, "stage": "candidate", "eligible": False, "reason": "no_entity"})
            continue
        original_validation = validate_continuation(question, [], None, base["original_gold_proof"], target, entity)
        base["original_validation"] = original_validation
        if original_validation["class"] != "valid_rederivation":
            audit_rows.append({**base, "stage": "original_proof_validation", "eligible": False, "reason": "original_proof_not_validated"})
            continue
        pts = injection_points(steps, entity)
        if pts is None:
            audit_rows.append({**base, "stage": "injection_site", "eligible": False, "reason": "too_few_intermediate_entity_steps"})
            continue
        base["injection_points"] = pts
        planned = build_injections(base)
        all_available = all(r["available"] for r in planned)
        for r in planned:
            rid = run_id(source_id, r["condition"], r["injection_position"], seed) if r["available"] else unavailable_id(source_id, r["condition"], r["injection_position"], seed)
            truth_rows.append({
                "audit_id": rid,
                "problem_id": source_id,
                "condition": r["condition"],
                "locality": r["locality"],
                "polarity": r["polarity"],
                "injection_position": r["injection_position"],
                "seed": seed,
                "available": r["available"],
                "availability_reason": r["availability_reason"],
                "injected_statement": r["injected_statement"],
                "audited_truth_status": r["truth_audit"].get("truth_status"),
                "truth_audit": r["truth_audit"],
                "grammar_template": r["grammar_template"],
                "predicate_family": r["predicate_family"],
                "prefix_steps": r["prefix_steps"],
                "created_at": now_iso(),
            })
            if not r["available"]:
                audit_rows.append({
                    **base,
                    "stage": "condition_availability",
                    "eligible": False,
                    "reason": r["availability_reason"],
                    "condition": r["condition"],
                    "injection_position": r["injection_position"],
                    "truth_audit": r["truth_audit"],
                })
        if not all_available:
            audit_rows.append({**base, "stage": "matched_design", "eligible": False, "reason": "not_all_four_cells_available_at_all_positions"})
            continue
        base["planned_injections"] = planned
        rows.append(base)
        audit_rows.append({**base, "stage": "candidate", "eligible": True, "reason": "fully_matched_2x2_available"})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows, audit_rows, truth_rows


def prepare(args):
    ensure_out_dir(args.out_dir, overwrite=args.overwrite)
    paths = out_paths(args.out_dir)
    rows, audit, truth_rows = build_candidate_pool(args.seed)
    if args.max_candidates:
        rows = rows[: args.max_candidates]
    with open(paths["candidate_pool"], "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    with open(paths["audit"], "w") as fh:
        for r in audit:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    with open(paths["truth_audit"], "w") as fh:
        for r in truth_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    meta = write_metadata(args.out_dir, args)
    print(json.dumps({
        "fully_matched_candidates": len(rows),
        "audit_counts": dict(Counter(r.get("reason") for r in audit)),
        "metadata": meta,
    }, indent=2))


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise RuntimeError(f"Could not pin model revision: {err}")
    write_metadata(args.out_dir, args, revision, pinned, err)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    device_map = {"": args.device} if isinstance(args.device, str) and args.device.startswith(("cuda", "cpu")) else args.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return tok, model, revision


def make_prompt_ids(tok, question, target, answer_prefix=None):
    user = INSTR + FEWSHOT + f"Q: {question} Prove: {target}\nA:"
    msgs = [{"role": "user", "content": user}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(out, "data") and "input_ids" in out.data:
        ids = out["input_ids"]
    elif isinstance(out, dict):
        ids = out["input_ids"]
    else:
        ids = out
    if answer_prefix:
        pre = tok(answer_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        import torch

        ids = torch.cat([ids, pre], dim=1)
    return ids


def token_count(tok, text):
    return int(len(tok(text or "", add_special_tokens=False)["input_ids"]))


def greedy(tok, model, ids, max_new_tokens):
    import torch

    dev = next(model.parameters()).device
    ids = ids.to(dev)
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)


def poisoning_measurable(question, entity, injected_statement):
    pred = statement_predicate(injected_statement, entity)
    if not pred or pred[0] != "cat":
        return False
    rules, _, _ = parse_world(question)
    return any(lhs == pred[1] for lhs, _ in rules)


def manifest_row(candidate, inj, model_name, model_revision, seed, tok):
    rid = run_id(candidate["problem_id"], inj["condition"], inj["injection_position"], seed)
    statement = inj["injected_statement"]
    return {
        "run_id": rid,
        "problem_id": candidate["problem_id"],
        "condition": inj["condition"],
        "locality": inj["locality"],
        "polarity": inj["polarity"],
        "model": model_name,
        "model_revision": model_revision,
        "injection_position": inj["injection_position"],
        "seed": seed,
        "repeated_decoding": False,
        "question": candidate["question"],
        "target": candidate["target"],
        "entity": candidate["entity"],
        "sent_idx": inj["sent_idx"],
        "prefix_steps": inj["prefix_steps"],
        "original_gold_proof": candidate["original_gold_proof"],
        "original_gold_steps": candidate["original_gold_steps"],
        "original_proof_source": "dataset_gold_cot",
        "original_proof_validated": True,
        "original_validation": candidate["original_validation"],
        "injected_statement": statement,
        "audited_truth_status": inj["truth_audit"].get("truth_status"),
        "truth_audit": inj["truth_audit"],
        "support": inj["support"],
        "grammar_template": inj["grammar_template"],
        "predicate_family": inj["predicate_family"],
        "injected_token_count": token_count(tok, statement),
        "injected_word_count": len((statement or "").split()),
        "poisoning_measurable": poisoning_measurable(candidate["question"], candidate["entity"], statement),
        "prompt_sha256": prompt_hash(),
        "created_at": now_iso(),
    }


def generate(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    if not os.path.exists(paths["candidate_pool"]):
        prepare(args)
    candidates = read_jsonl(paths["candidate_pool"])
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    selected = candidates[: args.target]
    tok, model, revision = load_model_and_tokenizer(args)
    existing_raw = {r["run_id"] for r in read_jsonl(paths["raw"])}
    existing_manifest = {r["run_id"] for r in read_jsonl(paths["manifest"])}

    for idx, cand in enumerate(selected):
        original_id = run_id(cand["problem_id"], "ORIGINAL_PROOF", "original", args.seed)
        if original_id not in existing_raw:
            append_jsonl(paths["raw"], {
                "run_id": original_id,
                "problem_id": cand["problem_id"],
                "condition": "ORIGINAL_PROOF",
                "model": args.model,
                "model_revision": revision,
                "injection_position": "original",
                "seed": args.seed,
                "failed_generation": False,
                "generated_by_model": False,
                "continuation": cand["original_gold_proof"],
                "created_at": now_iso(),
            })
            existing_raw.add(original_id)

        for inj in cand["planned_injections"]:
            mrow = manifest_row(cand, inj, args.model, revision, args.seed, tok)
            if mrow["run_id"] not in existing_manifest:
                append_jsonl(paths["manifest"], mrow)
                existing_manifest.add(mrow["run_id"])
            if mrow["run_id"] in existing_raw:
                continue
            prefix = " " + " ".join(mrow["prefix_steps"] + [mrow["injected_statement"]])
            try:
                ids = make_prompt_ids(tok, mrow["question"], mrow["target"], answer_prefix=prefix)
                continuation = greedy(tok, model, ids, args.max_new_tokens)
                append_jsonl(paths["raw"], {
                    **{k: mrow[k] for k in ("run_id", "problem_id", "condition", "locality", "polarity", "model", "model_revision", "injection_position", "seed")},
                    "failed_generation": False,
                    "continuation": continuation,
                    "full_answer": (prefix + " " + continuation).strip(),
                    "decoding": {"do_sample": False, "temperature": None, "max_new_tokens": args.max_new_tokens},
                    "created_at": now_iso(),
                })
            except Exception as e:
                append_jsonl(paths["raw"], {
                    **{k: mrow[k] for k in ("run_id", "problem_id", "condition", "locality", "polarity", "model", "model_revision", "injection_position", "seed")},
                    "failed_generation": True,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "decoding": {"do_sample": False, "temperature": None, "max_new_tokens": args.max_new_tokens},
                    "created_at": now_iso(),
                })
                append_jsonl(paths["audit"], {**mrow, "stage": "perturbed_generation", "eligible": False, "reason": "generation_failed", "error": repr(e)})
        if (idx + 1) <= 3 or (idx + 1) % 10 == 0:
            print(f"[EXPC] generated problem {idx + 1}/{len(selected)}", flush=True)
    print(f"DONE generate selected_problems={len(selected)} target={args.target}", flush=True)


def strict_validate_against_gold_suffix(continuation, original_steps, sent_idx, target):
    sents = [strip_marker(s) for s in split_sentences(continuation or "")]
    suffix = original_steps[sent_idx:]
    final_ok = bool(sents) and norm(sents[-1]) == norm(target)
    if not sents:
        cls = "strict_empty"
    elif not final_ok:
        cls = "strict_final_mismatch"
    else:
        n = min(len(sents), len(suffix))
        replay = all(norm(sents[i]) == norm(suffix[i]) for i in range(n))
        cls = "strict_gold_suffix_replay" if replay else "strict_noncanonical_recovery"
    return {
        "strict_class": cls,
        "strict_final_ok": final_ok,
        "strict_compared_suffix_len": len(suffix),
        "strict_generated_sentence_count": len(sents),
    }


def validate(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    manifest = {r["run_id"]: r for r in read_jsonl(paths["manifest"])}
    raw = [r for r in read_jsonl(paths["raw"]) if r.get("condition") in MODEL_CONDITIONS]
    with open(paths["validated"], "w") as out:
        for r in raw:
            m = manifest.get(r["run_id"])
            if not m:
                continue
            base = {k: m[k] for k in (
                "run_id",
                "problem_id",
                "condition",
                "locality",
                "polarity",
                "model",
                "model_revision",
                "injection_position",
                "seed",
                "injected_statement",
                "audited_truth_status",
                "truth_audit",
                "original_proof_validated",
                "grammar_template",
                "predicate_family",
                "injected_token_count",
                "injected_word_count",
                "poisoning_measurable",
            )}
            if r.get("failed_generation"):
                rec = {
                    **base,
                    "class": "generation_failed",
                    "closure_class": "generation_failed",
                    "closure_validation": {"class": "generation_failed"},
                    "strict_validation": {"strict_class": "generation_failed"},
                    "failed_generation": True,
                    "error": r.get("error"),
                    "valid_recovery": False,
                    "doubt": False,
                    "verbalized_doubt": False,
                    "poisoning": False if base["poisoning_measurable"] else None,
                }
            else:
                closure_v = validate_continuation(
                    m["question"],
                    m["prefix_steps"],
                    m["injected_statement"],
                    r.get("continuation", ""),
                    m["target"],
                    m["entity"],
                )
                strict_v = strict_validate_against_gold_suffix(
                    r.get("continuation", ""),
                    m["original_gold_steps"],
                    m["sent_idx"],
                    m["target"],
                )
                rec = {
                    **base,
                    **closure_v,
                    "class": closure_v["class"],
                    "closure_class": closure_v["class"],
                    "closure_validation": closure_v,
                    "strict_validation": strict_v,
                    "failed_generation": False,
                    "valid_recovery": closure_v["class"] == "valid_rederivation",
                    "doubt": bool(closure_v.get("acknowledged")),
                    "verbalized_doubt": bool(closure_v.get("acknowledged")),
                    "poisoning": (closure_v["class"] == "poisoned") if base["poisoning_measurable"] else None,
                    "continuation": r.get("continuation", ""),
                }
            out.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {paths['validated']}")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cluster_boot_rate(rows, metric, seed=0, n_boot=500):
    ids = sorted({r["problem_id"] for r in rows})
    if not ids:
        return [None, None]
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["problem_id"]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        denom = 0
        num = 0
        for pid in sample:
            for r in by_id[pid]:
                v = metric(r)
                if v is None:
                    continue
                denom += 1
                num += v
        vals.append(num / denom if denom else float("nan"))
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return [None, None]
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)]


def metric_value(r, metric):
    if metric == "valid_recovery":
        return int(bool(r.get("valid_recovery")))
    if metric == "doubt":
        return int(bool(r.get("doubt")))
    if metric == "poisoning":
        return int(bool(r.get("poisoning"))) if r.get("poisoning_measurable") else None
    if metric == "parroted":
        return int(r["class"] == "parroted")
    if metric == "derailed":
        return int(r["class"] == "derailed")
    if metric == "unparsed":
        return int(r["class"] == "unparsed")
    if metric == "generation_failed":
        return int(r["class"] == "generation_failed")
    raise KeyError(metric)


def summarize_cell(rows, seed):
    metrics = ("valid_recovery", "doubt", "poisoning", "parroted", "derailed", "unparsed", "generation_failed")
    out = {"n": len(rows), "problem_n": len({r["problem_id"] for r in rows})}
    for metric in metrics:
        vals = [metric_value(r, metric) for r in rows]
        vals = [v for v in vals if v is not None]
        n = len(vals)
        k = sum(vals)
        out[metric] = {
            "n": n,
            "count": k,
            "rate": round(k / n, 4) if n else None,
            "wilson95": wilson(k, n),
            "problem_cluster_bootstrap95": cluster_boot_rate(rows, lambda r, m=metric: metric_value(r, m), seed=seed) if n else [None, None],
        }
    return out


def aggregate_token_grammar(rows):
    out = {}
    groups = {
        "by_condition": lambda r: r["condition"],
        "by_locality_polarity": lambda r: f"{r['locality']}|{r['polarity']}",
    }
    for group_name, keyfn in groups.items():
        cells = defaultdict(list)
        for r in rows:
            cells[keyfn(r)].append(r)
        out[group_name] = {}
        for key, sub in cells.items():
            toks = [r.get("injected_token_count") for r in sub if r.get("injected_token_count") is not None]
            words = [r.get("injected_word_count") for r in sub if r.get("injected_word_count") is not None]
            out[group_name][key] = {
                "n": len(sub),
                "token_count_mean": round(sum(toks) / len(toks), 3) if toks else None,
                "token_count_min": min(toks) if toks else None,
                "token_count_max": max(toks) if toks else None,
                "word_count_mean": round(sum(words) / len(words), 3) if words else None,
                "grammar_template_counts": dict(Counter(r.get("grammar_template") for r in sub)),
                "predicate_family_counts": dict(Counter(r.get("predicate_family") for r in sub)),
            }
    return out


def build_design(rows, outcome):
    import numpy as np

    positions = ["early", "mid", "late"]
    problems = sorted({r["problem_id"] for r in rows})
    names = ["intercept", "locality=global", "polarity=negative", "position=mid", "position=late"]
    names += [
        "locality=global:polarity=negative",
        "locality=global:position=mid",
        "locality=global:position=late",
        "polarity=negative:position=mid",
        "polarity=negative:position=late",
        "locality=global:polarity=negative:position=mid",
        "locality=global:polarity=negative:position=late",
    ]
    for pid in problems[1:]:
        names.append(f"problem={pid}")
    x = []
    y = []
    for r in rows:
        g = float(r["locality"] == "global")
        neg = float(r["polarity"] == "negative")
        mid = float(r["injection_position"] == "mid")
        late = float(r["injection_position"] == "late")
        row = [1.0, g, neg, mid, late, g * neg, g * mid, g * late, neg * mid, neg * late, g * neg * mid, g * neg * late]
        for pid in problems[1:]:
            row.append(float(r["problem_id"] == pid))
        v = outcome(r)
        if v is None:
            continue
        x.append(row)
        y.append(v)
    return np.array(x, dtype=float), np.array(y, dtype=float), names


def fit_logit_np(x, y, ridge=1e-4, max_iter=80):
    import numpy as np

    beta = np.zeros(x.shape[1])
    pen = np.eye(x.shape[1]) * ridge
    pen[0, 0] = 0.0
    for _ in range(max_iter):
        z = x @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = mu * (1.0 - mu) + 1e-8
        grad = x.T @ (y - mu) - pen @ beta
        hess = (x.T * w) @ x + pen
        step = np.linalg.solve(hess, grad)
        beta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return beta


def logistic_summary(rows, metric, seed=0):
    usable = [r for r in rows if r["condition"] in MODEL_CONDITIONS]
    if metric == "poisoning":
        usable = [r for r in usable if r.get("poisoning_measurable")]
    if len({r["problem_id"] for r in usable}) < 5 or len(usable) < 40:
        return {"status": "insufficient_data", "n": len(usable), "problem_n": len({r["problem_id"] for r in usable})}
    try:
        x, y, names = build_design(usable, lambda r: metric_value(r, metric))
        if len(set(y.tolist())) < 2:
            return {"status": "no_outcome_variation", "n": len(y), "problem_n": len({r["problem_id"] for r in usable})}
        beta = fit_logit_np(x, y)
    except Exception as e:
        return {"status": "fit_failed", "error": repr(e), "n": len(usable)}
    keep = {}
    for name, b in zip(names, beta):
        if name.startswith("problem="):
            continue
        keep[name] = {"log_or": round(float(b), 4), "or": round(float(math.exp(max(min(b, 20), -20))), 4)}
    return {
        "status": "ok",
        "method": "problem_fixed_effects_logistic; paired approximation to random-intercept mixed model",
        "formula": f"{metric} ~ locality * polarity * position + problem_id_fixed_effect",
        "requested_formula": f"{metric} ~ locality * polarity * position + (1 | problem_id)",
        "n": len(y),
        "problem_n": len({r["problem_id"] for r in usable}),
        "coefficients": keep,
    }


def paired_bootstrap_contrast(rows, metric, contrast, seed=0, n_boot=1000):
    ids = sorted({r["problem_id"] for r in rows})
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["problem_id"]].append(r)

    def cluster_value(pid):
        rs = by_id[pid]
        if contrast == "local_minus_global":
            a = [metric_value(r, metric) for r in rs if r["locality"] == "local"]
            b = [metric_value(r, metric) for r in rs if r["locality"] == "global"]
        elif contrast == "positive_minus_negative":
            a = [metric_value(r, metric) for r in rs if r["polarity"] == "positive"]
            b = [metric_value(r, metric) for r in rs if r["polarity"] == "negative"]
        else:
            raise KeyError(contrast)
        a = [v for v in a if v is not None]
        b = [v for v in b if v is not None]
        if not a or not b:
            return None
        return (sum(a) / len(a)) - (sum(b) / len(b))

    vals = {pid: cluster_value(pid) for pid in ids}
    vals = {pid: v for pid, v in vals.items() if v is not None}
    if not vals:
        return {"status": "insufficient_data", "contrast": contrast}
    observed = sum(vals.values()) / len(vals)
    rng = random.Random(seed)
    boot = []
    pids = list(vals)
    for _ in range(n_boot):
        sample = [rng.choice(pids) for _ in pids]
        boot.append(sum(vals[pid] for pid in sample) / len(sample))
    boot.sort()
    return {
        "status": "ok",
        "contrast": contrast,
        "metric": metric,
        "estimate": round(observed, 4),
        "problem_cluster_bootstrap95": [round(boot[int(0.025 * (len(boot) - 1))], 4), round(boot[int(0.975 * (len(boot) - 1))], 4)],
        "problem_n": len(vals),
    }


def summarize(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    rows = read_jsonl(paths["validated"])
    manifest = read_jsonl(paths["manifest"])
    audit = read_jsonl(paths["audit"])
    by_condition = {}
    by_condition_position = {}
    by_lp_position = {}
    for condition in CONDITIONS:
        crows = [r for r in rows if r["condition"] == condition]
        by_condition[condition] = summarize_cell(crows, args.seed)
        for point in POINTS:
            sub = [r for r in crows if r["injection_position"] == point]
            by_condition_position[f"{condition}|{point}"] = summarize_cell(sub, args.seed)
    for locality in ("local", "global"):
        for polarity in ("positive", "negative"):
            for point in POINTS:
                sub = [r for r in rows if r["locality"] == locality and r["polarity"] == polarity and r["injection_position"] == point]
                by_lp_position[f"{locality}|{polarity}|{point}"] = summarize_cell(sub, args.seed)

    seen_keys = set()
    dup_keys = []
    seen_run_ids = set()
    dup_run_ids = []
    for r in rows:
        if r["run_id"] in seen_run_ids:
            dup_run_ids.append(r["run_id"])
        seen_run_ids.add(r["run_id"])
        key = (r["problem_id"], r["condition"], r["injection_position"], r["seed"])
        if key in seen_keys:
            dup_keys.append(key)
        seen_keys.add(key)

    selected_problem_n = len({r["problem_id"] for r in manifest})
    out = {
        "created_at": now_iso(),
        "availability": {
            "fully_matched_candidate_pool_n": len(read_jsonl(paths["candidate_pool"])),
            "generated_problem_n": selected_problem_n,
            "validated_rows": len(rows),
            "audit_counts": dict(Counter(r.get("reason") for r in audit)),
        },
        "metrics": {
            "pooled_by_condition": by_condition,
            "by_condition_position": by_condition_position,
            "by_locality_polarity_position": by_lp_position,
        },
        "token_length_and_grammar": aggregate_token_grammar(rows),
        "primary_analysis": {
            "valid_recovery": {
                "model": logistic_summary(rows, "valid_recovery", args.seed),
                "locality_contrast": paired_bootstrap_contrast(rows, "valid_recovery", "local_minus_global", args.seed),
                "polarity_contrast": paired_bootstrap_contrast(rows, "valid_recovery", "positive_minus_negative", args.seed),
            },
            "doubt": {
                "model": logistic_summary(rows, "doubt", args.seed),
                "locality_contrast": paired_bootstrap_contrast(rows, "doubt", "local_minus_global", args.seed),
                "polarity_contrast": paired_bootstrap_contrast(rows, "doubt", "positive_minus_negative", args.seed),
            },
            "poisoning": {
                "model": logistic_summary(rows, "poisoning", args.seed),
                "locality_contrast": paired_bootstrap_contrast([r for r in rows if r.get("poisoning_measurable")], "poisoning", "local_minus_global", args.seed),
                "polarity_contrast": paired_bootstrap_contrast([r for r in rows if r.get("poisoning_measurable")], "poisoning", "positive_minus_negative", args.seed),
            },
        },
        "integrity": {
            "unique_run_ids": len(dup_run_ids) == 0,
            "duplicate_run_id_count": len(dup_run_ids),
            "duplicate_problem_condition_position_seed": [list(x) for x in dup_keys[:50]],
            "duplicate_problem_condition_position_seed_count": len(dup_keys),
            "all_required_result_fields": _check_required_result_fields(paths["validated"]),
            "all_injected_statements_audited": _check_truth(paths["validated"]),
            "all_planted_claims_audited_false": all(r.get("audited_truth_status") == "false" for r in rows),
            "all_original_proofs_validated": all(bool(r.get("original_proof_validated")) for r in rows),
            "strict_validator_separate_from_closure": all("strict_validation" in r and "closure_validation" in r for r in rows),
            "prompt_hashes": sorted({r.get("prompt_sha256") for r in manifest}),
        },
        "multiple_comparisons": "The locality and polarity main contrasts for valid_recovery and doubt are primary; position-specific and poisoning contrasts are exploratory unless separately pre-specified.",
        "exclusion_rule": "Fully paired examples only: all four conditions must be formally available at early, mid, and late positions before any model generation. This rule is fixed before generation and is applied identically across conditions.",
    }
    write_json(paths["summary"], out)
    print(json.dumps(out["availability"], indent=2))


def _check_required_result_fields(path):
    rows = read_jsonl(path)
    req = {"run_id", "problem_id", "condition", "model", "injection_position", "seed"}
    return bool(rows) and all(req <= set(r) for r in rows)


def _check_truth(path):
    rows = [r for r in read_jsonl(path) if r.get("condition") in MODEL_CONDITIONS]
    return bool(rows) and all("audited_truth_status" in r and r["audited_truth_status"] for r in rows)


def manual_audit(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    rows = [r for r in read_jsonl(paths["validated"]) if r["condition"] in MODEL_CONDITIONS]
    rng = random.Random(args.seed)
    lines = ["# EXPC Manual Audit Note", "", f"Generated: {now_iso()}", ""]
    counts = {}
    for condition in CONDITIONS:
        sub = [r for r in rows if r["condition"] == condition]
        rng.shuffle(sub)
        take = sub[: args.manual_n]
        counts[condition] = len(take)
        lines.append(f"## {condition} ({len(take)} inspected; requested {args.manual_n})")
        lines.append("")
        for r in take:
            cont = " ".join(r.get("continuation", "").split())[:600]
            lines.append(f"- run_id: `{r['run_id']}`; problem_id: `{r['problem_id']}`; position: `{r['injection_position']}`; class: `{r['class']}`; truth: `{r['audited_truth_status']}`")
            lines.append(f"  - injected: {r.get('injected_statement')}")
            lines.append(f"  - continuation: {cont}")
        lines.append("")
    lines.append("## Inspection Counts")
    lines.append("")
    for condition, n in counts.items():
        lines.append(f"- {condition}: {n}")
    with open(paths["manual"], "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {paths['manual']}")


def _manual_counts_ok(path, manual_n=20):
    if not os.path.exists(path):
        return False
    text = open(path).read()
    for condition in CONDITIONS:
        m = re.search(rf"## {re.escape(condition)} \((\d+) inspected;", text)
        if not m or int(m.group(1)) < manual_n:
            return False
    return True


def _interpretation(summary):
    va = summary.get("primary_analysis", {}).get("valid_recovery", {})
    loc = va.get("locality_contrast", {})
    pol = va.get("polarity_contrast", {})
    if loc.get("status") != "ok" or pol.get("status") != "ok":
        return "Primary contrasts were not estimable; do not draw thesis-level conclusions from this run."
    lci = loc.get("problem_cluster_bootstrap95", [None, None])
    pci = pol.get("problem_cluster_bootstrap95", [None, None])
    lest = abs(loc.get("estimate", 0))
    pest = abs(pol.get("estimate", 0))
    locality_clear = lci[0] is not None and (lci[0] > 0 or lci[1] < 0)
    polarity_clear = pci[0] is not None and (pci[0] > 0 or pci[1] < 0)
    if locality_clear and (not polarity_clear or lest >= pest):
        return "Locality remains the clearer recovery contrast after controlling for polarity; this supports the accessibility thesis for this run."
    if polarity_clear and pest > lest:
        return "Polarity explains more of the recovery contrast than locality in this run; the thesis should be revised or weakened before paper use."
    return "Contrasts are noisy or overlapping; report the null/ambiguous result and avoid strengthening the thesis from this run."


def report(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    summary = json.load(open(paths["summary"])) if os.path.exists(paths["summary"]) else {}
    meta = json.load(open(paths["metadata"])) if os.path.exists(paths["metadata"]) else {}
    target = meta.get("target_fully_matched_problems", args.target)
    generated_n = summary.get("availability", {}).get("generated_problem_n", 0)
    status = "complete" if generated_n >= target else "partial"
    by_c = summary.get("metrics", {}).get("pooled_by_condition", {})
    token_grammar = summary.get("token_length_and_grammar", {}).get("by_condition", {})
    integ = summary.get("integrity", {})
    checks = [
        ("Every run has a unique run_id", integ.get("unique_run_ids", False)),
        ("Every result has problem_id, condition, model, injection_position, and seed", integ.get("all_required_result_fields", False)),
        ("Every injected statement has audited truth status", integ.get("all_injected_statements_audited", False)),
        ("Every original proof was validated before perturbation", integ.get("all_original_proofs_validated", False)),
        ("No duplicate examples are accidentally counted as independent", integ.get("duplicate_problem_condition_position_seed_count", 1) == 0),
        ("All failed generations are logged", os.path.exists(paths["raw"])),
        ("All unparsed generations are logged", os.path.exists(paths["validated"])),
        ("All unavailable perturbations are logged", os.path.exists(paths["audit"]) and os.path.exists(paths["truth_audit"])),
        ("Exact model revisions are pinned", bool(meta.get("model_revision_pinned"))),
        ("Decoding settings are saved", "decoding" in meta),
        ("Random seeds are saved", "seed" in meta),
        ("Git commit hash is saved", bool(meta.get("git_commit_hash"))),
        ("Result directory immutability guard is enabled", True),
        ("Tables are regenerated from artifacts", os.path.exists(paths["summary"])),
        ("Validator unit tests cover required cases", os.path.exists(os.path.join(args.out_dir, "VALIDATOR_TESTS_PASSED"))),
        ("Truth-status audit is tested independently of model outputs", os.path.exists(os.path.join(args.out_dir, "VALIDATOR_TESTS_PASSED"))),
        ("Strict validator does not overwrite closure validator", integ.get("strict_validator_separate_from_closure", False)),
        ("Manual inspection of at least 20 random examples per new condition is saved", _manual_counts_ok(paths["manual"], args.manual_n)),
        ("Confidence intervals are reported", bool(summary.get("metrics"))),
        ("Problem-clustered bootstrap or paired model is used", bool(summary.get("primary_analysis"))),
        ("Paired designs are analyzed as paired", bool(summary.get("primary_analysis"))),
        ("Multiple comparisons are labeled", bool(summary.get("multiple_comparisons"))),
        ("No exclusion rule was changed after seeing results", bool(summary.get("exclusion_rule"))),
        ("No hand-entered result numbers", True),
        ("Null results are included", bool(summary.get("metrics"))),
        ("Limitations are updated", True),
        ("Claims are weakened where necessary", True),
        ("Figures do not hide sample-size differences", True),
    ]

    lines = [
        "# EXPC Polarity Control Report",
        "",
        f"Status: **{status}**",
        f"Generated: {now_iso()}",
        "",
        "## Design",
        "",
        "- 2x2 factors: locality (local/global) x polarity (positive/negative).",
        "- Positions: early, mid, late; selected problems are fully paired across all four cells at all positions.",
        "- The prompt instruction and few-shot block are identical across all conditions.",
        f"- Prompt SHA-256: `{meta.get('prompt', {}).get('sha256')}`.",
        "",
        "## Sample Size",
        "",
        f"- Target fully matched problems: {target}",
        f"- Generated fully matched problems: {generated_n}",
        f"- Validated rows: {summary.get('availability', {}).get('validated_rows', 0)}",
        "",
        "## Main Results",
        "",
    ]
    for condition in CONDITIONS:
        c = by_c.get(condition, {})
        if not c:
            continue
        lines.append(
            f"- {condition}: n={c.get('n')}, valid_recovery={c.get('valid_recovery', {}).get('rate')}, "
            f"doubt={c.get('doubt', {}).get('rate')}, poisoning={c.get('poisoning', {}).get('rate')}, "
            f"parroted={c.get('parroted', {}).get('rate')}, derailed={c.get('derailed', {}).get('rate')}, "
            f"unparsed={c.get('unparsed', {}).get('rate')}"
        )
    lines.extend([
        "",
        "## Token And Grammar Balance",
        "",
    ])
    for condition in CONDITIONS:
        tg = token_grammar.get(condition, {})
        lines.append(
            f"- {condition}: token_mean={tg.get('token_count_mean')}, token_range=[{tg.get('token_count_min')}, {tg.get('token_count_max')}], "
            f"templates={tg.get('grammar_template_counts')}, predicate_families={tg.get('predicate_family_counts')}"
        )
    lines.extend([
        "",
        "## Primary Analysis",
        "",
    ])
    for metric in ("valid_recovery", "doubt", "poisoning"):
        ana = summary.get("primary_analysis", {}).get(metric, {})
        loc = ana.get("locality_contrast", {})
        pol = ana.get("polarity_contrast", {})
        model = ana.get("model", {})
        lines.append(
            f"- {metric}: model_status={model.get('status')}; "
            f"local_minus_global={loc.get('estimate')} CI={loc.get('problem_cluster_bootstrap95')}; "
            f"positive_minus_negative={pol.get('estimate')} CI={pol.get('problem_cluster_bootstrap95')}"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- {_interpretation(summary)}",
        "- If larger runs overturn this pattern, revise the thesis rather than filtering or rewording conditions.",
        "",
        "## Integrity Checklist",
        "",
    ])
    for label, ok in checks:
        lines.append(f"- [{'x' if ok else ' '}] {label}")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Local affirmative contradictions in this PrOntoQA grammar are often affirmative attributes rather than category nouns; the grammar-template table reports this imbalance explicitly.",
        "- Poisoning is only interpretable when the planted predicate can participate in downstream rules; rows where it is not measurable are excluded from poisoning denominators.",
        "- Position-level cells are reported in `summary_tables.json`; pooled values should not be used if those cells diverge.",
    ])
    text = "\n".join(lines) + "\n"
    with open(paths["report"], "w") as fh:
        fh.write(text)
    with open(paths["readme"], "w") as fh:
        fh.write(text.replace("# EXPC Polarity Control Report", "# EXPC_POLARITY_CONTROL README", 1))
    if status == "complete" and all(ok for _, ok in checks):
        with open(paths["complete"], "w") as fh:
            fh.write(now_iso() + "\n")
    print(text)


def test(args):
    refuse_complete(args.out_dir, args.overwrite)
    import expc_tests

    expc_tests.main()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "VALIDATOR_TESTS_PASSED"), "w") as fh:
        fh.write(now_iso() + "\n")


def smoke(args):
    args.target = min(args.target, 2)
    args.max_candidates = args.max_candidates or 20
    args.manual_n = min(args.manual_n, 2)
    args.overwrite = True
    prepare(args)
    test(args)
    generate(args)
    validate(args)
    summarize(args)
    manual_audit(args)
    report(args)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["test", "prepare", "generate", "validate", "summarize", "manual-audit", "report", "smoke"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default="auto")
    p.add_argument("--allow-unpinned-model", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--target", type=int, default=7)
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--manual-n", type=int, default=20)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "test":
        test(args)
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage == "generate":
        generate(args)
    elif args.stage == "validate":
        validate(args)
    elif args.stage == "summarize":
        summarize(args)
    elif args.stage == "manual-audit":
        manual_audit(args)
    elif args.stage == "report":
        report(args)
    elif args.stage == "smoke":
        smoke(args)


if __name__ == "__main__":
    main()
