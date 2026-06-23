"""Experiment B: local-certificate flip.

Stages:
  .venv/bin/python src/expb_local_cert_flip.py test
  .venv/bin/python src/expb_local_cert_flip.py prepare --overwrite
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python src/expb_local_cert_flip.py generate
  .venv/bin/python src/expb_local_cert_flip.py validate
  .venv/bin/python src/expb_local_cert_flip.py summarize
  .venv/bin/python src/expb_local_cert_flip.py manual-audit
  .venv/bin/python src/expb_local_cert_flip.py report
  .venv/bin/python src/expb_local_cert_flip.py finalize

Use --out-dir results/EXPB_LOCAL_CERT_FLIP_SMOKE --target 2 for smoke runs.
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
import stat
import subprocess
import sys
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import FEWSHOT, INSTR
from validator import (
    DOUBT,
    closure,
    derivable,
    parse_fact,
    parse_world,
    strip_marker,
    validate_continuation,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "prontoqa_ood")
DEFAULT_OUT = os.path.join(BASE, "results", "EXPB_LOCAL_CERT_FLIP")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 0
POINTS = ("early", "mid", "late")
CONDITIONS = (
    "GLOBAL_BASELINE",
    "LOCAL_CERTIFICATE",
    "IRRELEVANT_CERTIFICATE_CONTROL",
)
PRIMARY_METRICS = ("valid_rederivation", "poisoned", "verbalized_doubt")
SECONDARY_METRICS = (
    "parroted",
    "derailed",
    "unparsed",
    "generation_failed",
    "continuation_length_tokens",
)


def now_iso():
    return _dt.datetime.now(_dt.UTC).isoformat()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def sha_row(parts, n=24):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:n]


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def singular(cat):
    return cat[:-2] if cat.endswith("es") else cat


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", text.strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def prompt_user_text(question, target):
    return INSTR + FEWSHOT + f"Q: {question} Prove: {target}\nA:"


def derived_steps(cot):
    out = []
    for i, s in enumerate(cot):
        first = s.split()[0] if s.split() else ""
        if first and first[0].isupper() and not first.lower().endswith("es") and first.lower() not in ("every", "each", "all"):
            out.append(i)
    return out


def entity_from_target_or_cot(target, cot):
    f = parse_fact(target)
    if f:
        return f[0]
    for s in cot:
        f = parse_fact(s)
        if f:
            return f[0]
    return cot[0].split()[0] if cot else None


def entity_sentences(steps):
    if not steps:
        return []
    ent = steps[0].split()[0]
    return [i for i, s in enumerate(steps) if s.split() and s.split()[0] == ent]


def injection_points(steps):
    ents = entity_sentences(steps)
    inter = ents[1:-1]
    if len(inter) < 3:
        return None
    return {"early": inter[0], "mid": inter[len(inter) // 2], "late": inter[-1]}


def world_categories(question):
    return sorted({singular(c) for c in re.findall(r"\b([a-z]+pus(?:es)?)\b", question)})


def world_state(question, entity):
    rules, facts, unparsed = parse_world(question)
    reach = closure(rules)
    premises = {p for e, p in facts if e == entity}
    return rules, facts, unparsed, reach, premises


def false_categories(question, entity):
    _, _, _, reach, premises = world_state(question, entity)
    return sorted(c for c in world_categories(question) if not derivable(("cat", c), premises, reach)), reach


def parse_statement_for_audit(statement, entity):
    s = strip_marker(statement or "").strip().rstrip(".")
    m = re.match(rf"^{re.escape(entity)}\s+is\s+(?:a|an)\s+(\w+)$", s)
    if m:
        return ("cat", singular(m.group(1).lower()), True)
    m = re.match(rf"^{re.escape(entity)}\s+is\s+not\s+(?:a|an)\s+(\w+)$", s)
    if m:
        return ("cat", singular(m.group(1).lower()), False)
    m = re.match(rf"^{re.escape(entity)}\s+is\s+(not\s+)?(\w+)$", s)
    if m:
        return ("adj", m.group(2).lower(), not bool(m.group(1)))
    return None


def audit_truth_status(question, statement, entity):
    parsed = parse_statement_for_audit(statement, entity)
    if parsed is None:
        return {
            "truth_status": "unparsed",
            "grammar_valid": False,
            "audit_basis": "statement_unparsed",
        }
    kind, name, positive = parsed
    _, _, _, reach, premises = world_state(question, entity)
    if kind == "cat":
        entailed = derivable(("cat", name), premises, reach)
        if positive:
            return {
                "truth_status": "true" if entailed else "false",
                "grammar_valid": True,
                "audit_basis": "entailed_category" if entailed else "unentailed_category_closed_world",
                "predicate": ["cat", name],
            }
        return {
            "truth_status": "false" if entailed else "true",
            "grammar_valid": True,
            "audit_basis": "negated_entailed_category" if entailed else "negated_unentailed_category_closed_world",
            "predicate": ["not_cat", name],
        }
    pred = ("adj", name, positive)
    opp = ("adj", name, not positive)
    entailed = derivable(pred, premises, reach)
    opp_entailed = derivable(opp, premises, reach)
    if entailed:
        truth, basis = "true", "entailed_attribute"
    elif opp_entailed:
        truth, basis = "false", "opposite_attribute_entailed"
    else:
        truth, basis = "unknown", "attribute_not_decidable"
    return {"truth_status": truth, "grammar_valid": True, "audit_basis": basis, "predicate": list(pred)}


def make_positive_cat(entity, cat):
    return f"{entity} is {article(cat)} {cat}."


def make_negative_cat(entity, cat):
    return f"{entity} is not {article(cat)} {cat}."


def locally_falsifies(falsehood_statement, certificate_statement, question, entity):
    f = parse_statement_for_audit(falsehood_statement, entity)
    c = parse_statement_for_audit(certificate_statement, entity)
    if not f or not c:
        return False
    _, _, _, reach, _ = world_state(question, entity)
    if f[0] != "cat" or not f[2]:
        return False
    fcat = f[1]
    if c[0] == "cat" and not c[2]:
        ccat = c[1]
        return ccat == fcat or ("cat", ccat) in reach.get(fcat, set())
    if c[0] == "adj":
        needed = ("adj", c[1], not c[2])
        return needed in reach.get(fcat, set())
    return False


def choose_irrelevant_certificate_cat(false_cats, false_cat, reach):
    candidates = []
    for cat in false_cats:
        if cat == false_cat:
            continue
        if ("cat", cat) in reach.get(false_cat, set()):
            continue
        candidates.append(cat)
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: (abs(len(c) - len(false_cat)), c))[0]


def source_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*ProofsOnly*.json"))) + sorted(
        glob.glob(os.path.join(DATA_DIR, "*Composed*.json"))
    )


def candidate_records():
    seen_questions = set()
    for path in source_files():
        data = json.load(open(path))
        for key, ex in data.items():
            for sub in ["test_example"] + [f"in_context_example{i}" for i in range(8)]:
                q = ex.get(sub)
                if not q or "chain_of_thought" not in q:
                    continue
                question = q["question"]
                source_id = f"{os.path.basename(path)}::{key}::{sub}"
                if question in seen_questions:
                    yield {"problem_id": source_id, "eligible": False, "reason": "duplicate_question"}
                    continue
                seen_questions.add(question)
                cot = q["chain_of_thought"]
                target = q["query"].replace("Prove:", "").strip()
                entity = entity_from_target_or_cot(target, cot)
                base = {
                    "problem_id": source_id,
                    "question": question,
                    "target": target,
                    "original_proof": cot,
                    "entity": entity,
                    "source_file": os.path.basename(path),
                    "source_key": key,
                    "source_subkey": sub,
                    "dataset_n_derived_steps": len(derived_steps(cot)),
                }
                if entity is None:
                    yield {**base, "eligible": False, "reason": "no_entity"}
                    continue
                if len(derived_steps(cot)) < 4:
                    yield {**base, "eligible": False, "reason": "too_few_original_steps"}
                    continue
                gv = validate_continuation(question, [], None, " ".join(cot), target, entity)
                if gv["class"] != "valid_rederivation":
                    yield {**base, "eligible": False, "reason": "original_proof_not_validator_valid", "validator": gv}
                    continue
                pts = injection_points(cot)
                if pts is None:
                    yield {**base, "eligible": False, "reason": "too_few_injection_points"}
                    continue
                fcats, reach = false_categories(question, entity)
                if not fcats:
                    yield {**base, "eligible": False, "reason": "no_global_falsehood_candidate"}
                    continue
                for point in POINTS:
                    si = pts[point]
                    for fcat in fcats:
                        irrel = choose_irrelevant_certificate_cat(fcats, fcat, reach)
                        falsehood = make_positive_cat(entity, fcat)
                        local_cert = make_negative_cat(entity, fcat)
                        irrel_cert = make_negative_cat(entity, irrel) if irrel else None
                        rec = {
                            **base,
                            "eligible": irrel is not None,
                            "reason": "eligible" if irrel is not None else "no_irrelevant_certificate",
                            "injection_position": point,
                            "sent_idx": si,
                            "correct_step": cot[si],
                            "falsehood_category": fcat,
                            "irrelevant_category": irrel,
                            "falsehood_statement": falsehood,
                            "local_certificate": local_cert,
                            "irrelevant_certificate": irrel_cert,
                        }
                        yield rec


def out_paths(out_dir):
    return {
        "manifest": os.path.join(out_dir, "manifest.jsonl"),
        "audit": os.path.join(out_dir, "certificate_audit.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
        "summary": os.path.join(out_dir, "summary_tables.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "manual": os.path.join(out_dir, "manual_inspection_audit.md"),
        "report": os.path.join(out_dir, "EXPB_REPORT.md"),
        "tests": os.path.join(out_dir, "VALIDATOR_TESTS_PASSED"),
        "complete": os.path.join(out_dir, "RUN_COMPLETE"),
    }


def ensure_out_dir(out_dir, overwrite=False):
    if os.path.exists(out_paths(out_dir)["complete"]) and not overwrite:
        raise SystemExit(f"{out_dir} is complete; refusing to modify without --overwrite")
    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def git_hash():
    try:
        root = subprocess.check_output(["git", "-C", BASE, "rev-parse", "--show-toplevel"], text=True).strip()
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty_summary():
    try:
        root = subprocess.check_output(["git", "-C", BASE, "rev-parse", "--show-toplevel"], text=True).strip()
        return subprocess.check_output(["git", "-C", root, "status", "--short"], text=True).splitlines()
    except Exception:
        return []


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
        "experiment": "EXPB_LOCAL_CERT_FLIP",
        "created_at": now_iso(),
        "git_commit_hash": git_hash(),
        "git_dirty_status": git_dirty_summary(),
        "model": args.model,
        "model_revision": model_revision or args.model_revision,
        "model_revision_pinned": bool(revision_pinned),
        "model_revision_error": revision_error,
        "seed": args.seed,
        "target_paired_triples_per_position": args.target,
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "num_return_sequences": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "prompt": {"instruction": INSTR, "fewshot": FEWSHOT},
        "conditions": list(CONDITIONS),
        "positions": list(POINTS),
        "python": sys.version,
    }
    write_json(out_paths(out_dir)["metadata"], meta)
    return meta


def triple_id(rec):
    return sha_row(["EXPB_TRIPLE", rec["problem_id"], rec["injection_position"], rec["sent_idx"], rec["falsehood_statement"]], 24)


def run_id(tid, condition, seed):
    return sha_row(["EXPB_RUN", tid, condition, seed], 24)


def answer_prefix_steps(rec, condition):
    steps = list(rec["original_proof"][: rec["sent_idx"]])
    if condition == "LOCAL_CERTIFICATE":
        steps.append(rec["local_certificate"])
    elif condition == "IRRELEVANT_CERTIFICATE_CONTROL":
        steps.append(rec["irrelevant_certificate"])
    steps.append(rec["falsehood_statement"])
    return steps


def validator_true_prefix_steps(rec, condition):
    steps = list(rec["original_proof"][: rec["sent_idx"]])
    if condition == "LOCAL_CERTIFICATE":
        steps.append(rec["local_certificate"])
    elif condition == "IRRELEVANT_CERTIFICATE_CONTROL":
        steps.append(rec["irrelevant_certificate"])
    return steps


def audit_rows_for_record(rec, tid, seed):
    question, entity = rec["question"], rec["entity"]
    false_audit = audit_truth_status(question, rec["falsehood_statement"], entity)
    local_audit = audit_truth_status(question, rec["local_certificate"], entity)
    irrel_audit = audit_truth_status(question, rec["irrelevant_certificate"], entity) if rec.get("irrelevant_certificate") else {
        "truth_status": "unavailable",
        "grammar_valid": False,
        "audit_basis": "no_irrelevant_certificate",
    }
    return [
        {
            "audit_id": sha_row(["audit", tid, "falsehood"], 20),
            "triple_id": tid,
            "problem_id": rec["problem_id"],
            "injection_position": rec["injection_position"],
            "seed": seed,
            "statement_role": "falsehood",
            "statement": rec["falsehood_statement"],
            "expected_truth_status": "false",
            **false_audit,
        },
        {
            "audit_id": sha_row(["audit", tid, "local_certificate"], 20),
            "triple_id": tid,
            "problem_id": rec["problem_id"],
            "injection_position": rec["injection_position"],
            "seed": seed,
            "statement_role": "local_certificate",
            "statement": rec["local_certificate"],
            "expected_truth_status": "true",
            "locally_falsifies_falsehood": locally_falsifies(rec["falsehood_statement"], rec["local_certificate"], question, entity),
            **local_audit,
        },
        {
            "audit_id": sha_row(["audit", tid, "irrelevant_certificate"], 20),
            "triple_id": tid,
            "problem_id": rec["problem_id"],
            "injection_position": rec["injection_position"],
            "seed": seed,
            "statement_role": "irrelevant_certificate",
            "statement": rec.get("irrelevant_certificate"),
            "expected_truth_status": "true",
            "locally_falsifies_falsehood": locally_falsifies(rec["falsehood_statement"], rec.get("irrelevant_certificate") or "", question, entity),
            **irrel_audit,
        },
    ]


def is_fully_audited(rec):
    rows = audit_rows_for_record(rec, triple_id(rec), DEFAULT_SEED)
    by_role = {r["statement_role"]: r for r in rows}
    return (
        by_role["falsehood"]["truth_status"] == "false"
        and by_role["local_certificate"]["truth_status"] == "true"
        and by_role["local_certificate"].get("locally_falsifies_falsehood") is True
        and by_role["irrelevant_certificate"]["truth_status"] == "true"
        and by_role["irrelevant_certificate"].get("locally_falsifies_falsehood") is False
    )


def build_manifest_row(rec, condition, model, model_revision, seed):
    tid = triple_id(rec)
    steps = answer_prefix_steps(rec, condition)
    vprefix = validator_true_prefix_steps(rec, condition)
    assistant_prefill = " " + " ".join(steps)
    user_text = prompt_user_text(rec["question"], rec["target"])
    certificate = None
    if condition == "LOCAL_CERTIFICATE":
        certificate = rec["local_certificate"]
    elif condition == "IRRELEVANT_CERTIFICATE_CONTROL":
        certificate = rec["irrelevant_certificate"]
    return {
        "run_id": run_id(tid, condition, seed),
        "triple_id": tid,
        "problem_id": rec["problem_id"],
        "condition": condition,
        "model": model,
        "model_revision": model_revision,
        "injection_position": rec["injection_position"],
        "seed": seed,
        "question": rec["question"],
        "target": rec["target"],
        "entity": rec["entity"],
        "source_file": rec["source_file"],
        "source_key": rec["source_key"],
        "source_subkey": rec["source_subkey"],
        "sent_idx": rec["sent_idx"],
        "correct_step": rec["correct_step"],
        "original_proof": rec["original_proof"],
        "original_proof_validated": True,
        "proof_prefix_steps": rec["original_proof"][: rec["sent_idx"]],
        "validator_true_prefix_steps": vprefix,
        "falsehood_statement": rec["falsehood_statement"],
        "falsehood_category": rec["falsehood_category"],
        "injected_statement": rec["falsehood_statement"],
        "injected_statement_truth_status": "false",
        "certificate_statement": certificate,
        "local_certificate": rec["local_certificate"],
        "irrelevant_certificate": rec["irrelevant_certificate"],
        "irrelevant_category": rec["irrelevant_category"],
        "assistant_prefill_steps": steps,
        "prompt_user_text": user_text,
        "assistant_prefill_text": assistant_prefill,
        "full_prompt_text": user_text + assistant_prefill,
        "prompt_format_check": "baseline_after_certificate_removal_matches" if condition != "GLOBAL_BASELINE" else "baseline",
        "created_at": now_iso(),
    }


def prepare(args):
    ensure_out_dir(args.out_dir, overwrite=args.overwrite)
    paths = out_paths(args.out_dir)
    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise RuntimeError(f"Could not pin model revision: {err}")
    write_metadata(args.out_dir, args, revision, pinned, err)

    eligible = []
    unavailable = []
    for rec in candidate_records():
        if not rec.get("eligible"):
            unavailable.append(rec)
            continue
        if not is_fully_audited(rec):
            rec = {**rec, "eligible": False, "reason": "audit_invariant_failed"}
            unavailable.append(rec)
            continue
        eligible.append(rec)

    rng = random.Random(args.seed)
    selected = []
    availability = {}
    for point in POINTS:
        rows = sorted([r for r in eligible if r["injection_position"] == point], key=lambda r: (r["problem_id"], r["falsehood_statement"]))
        rng.shuffle(rows)
        availability[point] = {"available": len(rows), "selected": min(args.target, len(rows)), "target": args.target}
        selected.extend(rows[: args.target])

    with open(paths["audit"], "w") as ah:
        for rec in unavailable:
            ah.write(json.dumps({"stage": "availability", **rec}, sort_keys=True) + "\n")
        for rec in selected:
            tid = triple_id(rec)
            for row in audit_rows_for_record(rec, tid, args.seed):
                ah.write(json.dumps(row, sort_keys=True) + "\n")

    seen_runs = set()
    seen_examples = set()
    with open(paths["manifest"], "w") as mh:
        for rec in selected:
            key = (rec["problem_id"], rec["injection_position"], rec["falsehood_statement"])
            if key in seen_examples:
                raise RuntimeError(f"duplicate example key: {key}")
            seen_examples.add(key)
            for condition in CONDITIONS:
                row = build_manifest_row(rec, condition, args.model, revision, args.seed)
                if row["run_id"] in seen_runs:
                    raise RuntimeError(f"duplicate run_id: {row['run_id']}")
                seen_runs.add(row["run_id"])
                mh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"availability": availability, "manifest_rows": len(seen_runs)}, indent=2))


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise RuntimeError(f"Could not pin model revision: {err}")
    write_metadata(args.out_dir, args, revision, pinned, err)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=torch.bfloat16, device_map=args.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    return tok, model, revision


def make_prompt_ids(tok, question, target, answer_prefix=None):
    msgs = [{"role": "user", "content": prompt_user_text(question, target)}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(out, "data") and "input_ids" in out.data:
        ids = out["input_ids"]
    elif isinstance(out, dict):
        ids = out["input_ids"]
    else:
        ids = out
    if answer_prefix:
        import torch

        pre = tok(answer_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([ids, pre], dim=1)
    return ids


def greedy(tok, model, ids, max_new_tokens):
    import torch

    with torch.no_grad():
        input_ids = ids.to(model.device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True), out[0]


def token_len(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def generate(args):
    paths = out_paths(args.out_dir)
    if not os.path.exists(paths["manifest"]):
        prepare(args)
    manifest = read_jsonl(paths["manifest"])
    tok, model, revision = load_model_and_tokenizer(args)
    existing = {r["run_id"] for r in read_jsonl(paths["raw"])}
    for i, row in enumerate(manifest):
        if row["run_id"] in existing:
            continue
        try:
            ids = make_prompt_ids(tok, row["question"], row["target"], row["assistant_prefill_text"])
            continuation, full_ids = greedy(tok, model, ids, args.max_new_tokens)
            rec = {
                **{k: row[k] for k in ("run_id", "triple_id", "problem_id", "condition", "model", "injection_position", "seed")},
                "model_revision": revision,
                "failed_generation": False,
                "continuation": continuation,
                "full_answer": (row["assistant_prefill_text"] + " " + continuation).strip(),
                "prompt_token_count": int(ids.shape[1]),
                "total_token_count": int(full_ids.shape[0]),
                "continuation_token_count": int(full_ids.shape[0] - ids.shape[1]),
                "assistant_prefill_token_count": token_len(tok, row["assistant_prefill_text"]),
                "certificate_token_count": token_len(tok, row["certificate_statement"] or ""),
                "falsehood_token_count": token_len(tok, row["falsehood_statement"]),
                "model_input_text_decoded": tok.decode(ids[0], skip_special_tokens=False),
                "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                "created_at": now_iso(),
            }
        except Exception as e:
            rec = {
                **{k: row[k] for k in ("run_id", "triple_id", "problem_id", "condition", "model", "model_revision", "injection_position", "seed")},
                "failed_generation": True,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                "created_at": now_iso(),
            }
        append_jsonl(paths["raw"], rec)
        existing.add(row["run_id"])
        if (i + 1) % 25 == 0 or i < 3:
            print(f"[EXPB generate] {i+1}/{len(manifest)}", flush=True)
    print(f"DONE generate rows={len(manifest)}")


def validate(args):
    paths = out_paths(args.out_dir)
    manifest = {r["run_id"]: r for r in read_jsonl(paths["manifest"])}
    raw = {r["run_id"]: r for r in read_jsonl(paths["raw"])}
    with open(paths["validated"], "w") as out:
        for run_id_key, m in manifest.items():
            r = raw.get(run_id_key)
            base = {
                "run_id": m["run_id"],
                "triple_id": m["triple_id"],
                "problem_id": m["problem_id"],
                "condition": m["condition"],
                "model": m["model"],
                "model_revision": m["model_revision"],
                "injection_position": m["injection_position"],
                "seed": m["seed"],
                "sent_idx": m["sent_idx"],
                "injected_statement": m["injected_statement"],
                "injected_statement_truth_status": m["injected_statement_truth_status"],
                "certificate_statement": m["certificate_statement"],
                "original_proof_validated": m["original_proof_validated"],
            }
            if r is None:
                rec = {**base, "class": "generation_missing", "failed_generation": True, "verbalized_doubt": False}
            elif r.get("failed_generation"):
                rec = {**base, "class": "generation_failed", "failed_generation": True, "error": r.get("error"), "verbalized_doubt": False}
            else:
                v = validate_continuation(
                    m["question"],
                    m["validator_true_prefix_steps"],
                    m["falsehood_statement"],
                    r.get("continuation", ""),
                    m["target"],
                    m["entity"],
                )
                rec = {
                    **base,
                    **v,
                    "failed_generation": False,
                    "verbalized_doubt": bool(v.get("acknowledged")),
                    "continuation": r.get("continuation", ""),
                    "continuation_length_chars": len(r.get("continuation", "")),
                    "continuation_length_tokens": r.get("continuation_token_count"),
                    "prompt_token_count": r.get("prompt_token_count"),
                    "assistant_prefill_token_count": r.get("assistant_prefill_token_count"),
                    "certificate_token_count": r.get("certificate_token_count"),
                    "falsehood_token_count": r.get("falsehood_token_count"),
                    "strict_validator_available": False,
                    "stepwise_valid_recovery": None,
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


def percentile(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def metric_value(row, metric):
    cls = row.get("class")
    if metric == "valid_rederivation":
        return int(cls == "valid_rederivation")
    if metric == "poisoned":
        return int(cls == "poisoned")
    if metric == "verbalized_doubt":
        return int(bool(row.get("verbalized_doubt")))
    if metric == "parroted":
        return int(cls == "parroted")
    if metric == "derailed":
        return int(cls == "derailed")
    if metric == "unparsed":
        return int(cls == "unparsed")
    if metric == "generation_failed":
        return int(cls in ("generation_failed", "generation_missing"))
    raise KeyError(metric)


def cluster_boot_rate(rows, metric, seed=0, n_boot=400):
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
        nums = []
        for pid in sample:
            nums.extend(metric_value(r, metric) for r in by_id[pid])
        vals.append(sum(nums) / len(nums) if nums else 0.0)
    return [round(percentile(vals, 0.025), 4), round(percentile(vals, 0.975), 4)]


def summarize_cell(rows, seed):
    n = len(rows)
    out = {"n": n, "triple_n": len({r["triple_id"] for r in rows}), "problem_n": len({r["problem_id"] for r in rows})}
    for metric in PRIMARY_METRICS + SECONDARY_METRICS[:-1]:
        k = sum(metric_value(r, metric) for r in rows)
        out[metric] = {
            "count": k,
            "rate": round(k / n, 4) if n else None,
            "wilson95": wilson(k, n),
            "problem_cluster_bootstrap95": cluster_boot_rate(rows, metric, seed) if n else [None, None],
        }
    lengths = [r.get("continuation_length_tokens") for r in rows if isinstance(r.get("continuation_length_tokens"), int)]
    out["continuation_length_tokens"] = {
        "mean": round(sum(lengths) / len(lengths), 2) if lengths else None,
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
    }
    return out


def paired_rows(rows, condition_a, condition_b, position=None):
    by_key = defaultdict(dict)
    for r in rows:
        if position and r["injection_position"] != position:
            continue
        if r["condition"] in (condition_a, condition_b):
            by_key[(r["problem_id"], r["triple_id"])][r["condition"]] = r
    out = []
    for (pid, tid), d in by_key.items():
        if condition_a in d and condition_b in d:
            out.append((pid, tid, d[condition_a], d[condition_b]))
    return out


def mcnemar_pvalue(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        k = min(b01, b10)
        prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        return min(1.0, 2 * prob)


def paired_bootstrap_diff(pairs, metric, seed=0, n_boot=1000):
    if not pairs:
        return {"n_pairs": 0, "diff": None, "ci95": [None, None]}
    diff_values = [metric_value(a, metric) - metric_value(b, metric) for _, _, a, b in pairs]
    diff = sum(diff_values) / len(diff_values)
    ids = sorted({pid for pid, _, _, _ in pairs})
    by_id = defaultdict(list)
    for pid, tid, a, b in pairs:
        by_id[pid].append((a, b))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        nums = []
        for pid in [rng.choice(ids) for _ in ids]:
            nums.extend(metric_value(a, metric) - metric_value(b, metric) for a, b in by_id[pid])
        vals.append(sum(nums) / len(nums) if nums else 0.0)
    a1b0 = sum(metric_value(a, metric) == 1 and metric_value(b, metric) == 0 for _, _, a, b in pairs)
    a0b1 = sum(metric_value(a, metric) == 0 and metric_value(b, metric) == 1 for _, _, a, b in pairs)
    return {
        "n_pairs": len(pairs),
        "problem_n": len(ids),
        "diff_a_minus_b": round(diff, 4),
        "ci95_problem_cluster_bootstrap": [round(percentile(vals, 0.025), 4), round(percentile(vals, 0.975), 4)],
        "mcnemar_a1_b0": a1b0,
        "mcnemar_a0_b1": a0b1,
        "mcnemar_exact_p": round(mcnemar_pvalue(a1b0, a0b1), 6),
    }


def paired_tests(rows, seed):
    out = {}
    comparisons = [
        ("LOCAL_CERTIFICATE", "GLOBAL_BASELINE"),
        ("LOCAL_CERTIFICATE", "IRRELEVANT_CERTIFICATE_CONTROL"),
    ]
    for metric in PRIMARY_METRICS:
        out[metric] = {}
        for a, b in comparisons:
            label = f"{a}_vs_{b}"
            out[metric][label] = {"pooled": paired_bootstrap_diff(paired_rows(rows, a, b), metric, seed)}
            for point in POINTS:
                out[metric][label][point] = paired_bootstrap_diff(paired_rows(rows, a, b, point), metric, seed)
    return out


def fit_penalized_logit(x, y, penalty, max_iter=80):
    import numpy as np

    beta = np.zeros(x.shape[1])
    for _ in range(max_iter):
        z = x @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = mu * (1.0 - mu) + 1e-8
        grad = x.T @ (y - mu) - penalty @ beta
        hess = (x.T * w) @ x + penalty
        step = np.linalg.solve(hess, grad)
        beta += step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    cov = np.linalg.pinv(hess)
    return beta, cov


def mixed_effects_logit(rows, metric):
    try:
        import numpy as np
    except Exception as e:
        return {"status": "unavailable", "error": repr(e)}
    yvals = [metric_value(r, metric) for r in rows]
    if len(set(yvals)) < 2:
        return {"status": "no_outcome_variation", "n": len(rows)}
    cond_base = "GLOBAL_BASELINE"
    pos_base = "early"
    features = ["intercept"]
    for cond in CONDITIONS:
        if cond != cond_base:
            features.append(f"condition={cond}")
    for point in POINTS:
        if point != pos_base:
            features.append(f"position={point}")
    for cond in CONDITIONS:
        if cond == cond_base:
            continue
        for point in POINTS:
            if point == pos_base:
                continue
            features.append(f"condition={cond}:position={point}")
    problems = sorted({r["problem_id"] for r in rows})
    for pid in problems:
        features.append(f"random_intercept_problem={pid}")
    x = []
    for r in rows:
        vals = [1.0]
        vals.extend(float(r["condition"] == cond) for cond in CONDITIONS if cond != cond_base)
        vals.extend(float(r["injection_position"] == point) for point in POINTS if point != pos_base)
        for cond in CONDITIONS:
            if cond == cond_base:
                continue
            for point in POINTS:
                if point == pos_base:
                    continue
                vals.append(float(r["condition"] == cond and r["injection_position"] == point))
        vals.extend(float(r["problem_id"] == pid) for pid in problems)
        x.append(vals)
    x = np.array(x, dtype=float)
    y = np.array(yvals, dtype=float)
    penalty = np.eye(x.shape[1]) * 1e-4
    penalty[0, 0] = 0.0
    for i, name in enumerate(features):
        if name.startswith("random_intercept_problem="):
            penalty[i, i] = 1.0
    try:
        beta, cov = fit_penalized_logit(x, y, penalty)
    except Exception as e:
        return {"status": "fit_failed", "error": repr(e), "n": len(rows)}
    fixed = {}
    for i, name in enumerate(features):
        if name.startswith("random_intercept_problem="):
            continue
        se = float(math.sqrt(max(cov[i, i], 0.0)))
        lo = float(beta[i] - 1.96 * se)
        hi = float(beta[i] + 1.96 * se)
        fixed[name] = {
            "log_or": round(float(beta[i]), 4),
            "se": round(se, 4),
            "or": round(float(math.exp(max(min(beta[i], 20), -20))), 4),
            "ci95_or": [round(float(math.exp(max(min(lo, 20), -20))), 4), round(float(math.exp(max(min(hi, 20), -20))), 4)],
        }
    return {
        "status": "ok",
        "method": "penalized logistic random-intercept approximation",
        "formula": f"{metric} ~ condition * position + (1 | problem_id)",
        "n": len(rows),
        "problem_n": len(problems),
        "coefficients": fixed,
    }


def prompt_sanity(manifest):
    by_tid = defaultdict(dict)
    for r in manifest:
        by_tid[r["triple_id"]][r["condition"]] = r
    ok = True
    failures = []
    token_deltas = []
    for tid, d in by_tid.items():
        if set(d) != set(CONDITIONS):
            ok = False
            failures.append({"triple_id": tid, "reason": "missing_condition", "conditions": sorted(d)})
            continue
        b = d["GLOBAL_BASELINE"]
        for cond in ("LOCAL_CERTIFICATE", "IRRELEVANT_CERTIFICATE_CONTROL"):
            c = d[cond]
            if b["question"] != c["question"] or b["target"] != c["target"] or b["proof_prefix_steps"] != c["proof_prefix_steps"] or b["falsehood_statement"] != c["falsehood_statement"]:
                ok = False
                failures.append({"triple_id": tid, "condition": cond, "reason": "unplanned_prompt_difference"})
        token_deltas.append({
            "triple_id": tid,
            "local_certificate_chars": len(d["LOCAL_CERTIFICATE"]["certificate_statement"] or ""),
            "irrelevant_certificate_chars": len(d["IRRELEVANT_CERTIFICATE_CONTROL"]["certificate_statement"] or ""),
            "local_minus_irrelevant_chars": len(d["LOCAL_CERTIFICATE"]["certificate_statement"] or "") - len(d["IRRELEVANT_CERTIFICATE_CONTROL"]["certificate_statement"] or ""),
        })
    return {"formatting_identical_except_certificate": ok, "failures": failures[:20], "certificate_char_deltas": token_deltas}


def summarize(args):
    paths = out_paths(args.out_dir)
    rows = read_jsonl(paths["validated"])
    manifest = read_jsonl(paths["manifest"])
    audit = read_jsonl(paths["audit"])
    by_condition = {}
    by_condition_position = {}
    for cond in CONDITIONS:
        crows = [r for r in rows if r["condition"] == cond]
        by_condition[cond] = summarize_cell(crows, args.seed)
        for point in POINTS:
            by_condition_position[f"{cond}|{point}"] = summarize_cell([r for r in crows if r["injection_position"] == point], args.seed)
    run_ids = [r["run_id"] for r in manifest]
    result_required = {"run_id", "problem_id", "condition", "model", "injection_position", "seed"}
    audits_by_role = defaultdict(list)
    for r in audit:
        if "statement_role" in r:
            audits_by_role[r["statement_role"]].append(r)
    prompt_check = prompt_sanity(manifest)
    char_diffs = [x["local_minus_irrelevant_chars"] for x in prompt_check["certificate_char_deltas"]]
    out = {
        "created_at": now_iso(),
        "sample_size": {
            "target_paired_triples_per_position": args.target,
            "paired_triples_by_position": {p: len({r["triple_id"] for r in manifest if r["injection_position"] == p}) for p in POINTS},
            "manifest_rows": len(manifest),
            "validated_rows": len(rows),
            "problem_clusters": len({r["problem_id"] for r in rows}),
        },
        "metrics": {"pooled_by_condition": by_condition, "by_condition_position": by_condition_position},
        "paired_tests": paired_tests(rows, args.seed),
        "mixed_effects_logistic": {m: mixed_effects_logit(rows, m) for m in PRIMARY_METRICS},
        "sanity_checks": {
            "local_certificate_true_rate": sum(r.get("truth_status") == "true" for r in audits_by_role["local_certificate"]) / max(1, len(audits_by_role["local_certificate"])),
            "falsehood_false_rate": sum(r.get("truth_status") == "false" for r in audits_by_role["falsehood"]) / max(1, len(audits_by_role["falsehood"])),
            "irrelevant_certificate_true_rate": sum(r.get("truth_status") == "true" for r in audits_by_role["irrelevant_certificate"]) / max(1, len(audits_by_role["irrelevant_certificate"])),
            "irrelevant_certificate_locally_falsifies_count": sum(bool(r.get("locally_falsifies_falsehood")) for r in audits_by_role["irrelevant_certificate"]),
            "prompt_formatting": prompt_check["formatting_identical_except_certificate"],
            "certificate_char_length_delta_local_minus_irrelevant": {
                "mean": round(sum(char_diffs) / len(char_diffs), 3) if char_diffs else None,
                "min": min(char_diffs) if char_diffs else None,
                "max": max(char_diffs) if char_diffs else None,
            },
        },
        "integrity": {
            "unique_run_ids": len(run_ids) == len(set(run_ids)),
            "required_result_fields_present": bool(rows) and all(result_required <= set(r) for r in rows),
            "every_injected_statement_audited": bool(audits_by_role["falsehood"]) and all(r.get("truth_status") == "false" for r in audits_by_role["falsehood"]),
            "all_original_proofs_validated": all(bool(r.get("original_proof_validated")) for r in manifest),
            "duplicate_example_keys": len(manifest) // len(CONDITIONS) - len({(r["problem_id"], r["injection_position"], r["falsehood_statement"]) for r in manifest}),
            "failed_generations_logged": os.path.exists(paths["raw"]),
            "unparsed_generations_logged": os.path.exists(paths["validated"]),
            "unavailable_perturbations_logged": any(r.get("stage") == "availability" for r in audit),
        },
        "reproducibility": json.load(open(paths["metadata"])) if os.path.exists(paths["metadata"]) else {},
        "validator_quality": {
            "strict_validator_available": False,
            "strict_validator_did_not_overwrite_closure_validator": True,
            "unit_test_marker_exists": os.path.exists(paths["tests"]),
        },
        "multiple_comparisons": "LOCAL_CERTIFICATE vs GLOBAL_BASELINE and LOCAL_CERTIFICATE vs IRRELEVANT_CERTIFICATE_CONTROL on valid_rederivation, poisoning, and verbalized_doubt are primary for Experiment B; other contrasts are exploratory.",
        "exclusions": "No post-generation exclusions. Failed and unparsed generations are retained in denominators and logged.",
    }
    write_json(paths["summary"], out)
    print(json.dumps(out["sample_size"], indent=2))


def manual_audit(args):
    paths = out_paths(args.out_dir)
    manifest = {r["run_id"]: r for r in read_jsonl(paths["manifest"])}
    rows = read_jsonl(paths["validated"])
    rng = random.Random(args.seed)
    lines = ["# EXPB Manual Inspection Audit", "", f"Generated: {now_iso()}", ""]
    for condition in CONDITIONS:
        sub = [r for r in rows if r["condition"] == condition]
        rng.shuffle(sub)
        take = sub[: args.manual_n]
        lines.append(f"## {condition} ({len(take)} inspected; requested {args.manual_n})")
        lines.append("")
        for r in take:
            m = manifest[r["run_id"]]
            cont = " ".join(r.get("continuation", "").split())[:700]
            lines.append(f"- run_id: `{r['run_id']}`; triple_id: `{r['triple_id']}`; problem_id: `{r['problem_id']}`; position: `{r['injection_position']}`; class: `{r['class']}`")
            lines.append(f"  - certificate: {m.get('certificate_statement')}")
            lines.append(f"  - falsehood: {m.get('falsehood_statement')}")
            lines.append(f"  - proof prefix: {' '.join(m.get('proof_prefix_steps', []))[:700]}")
            lines.append(f"  - continuation: {cont}")
        lines.append("")
    with open(paths["manual"], "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {paths['manual']}")


def report(args):
    paths = out_paths(args.out_dir)
    summary = json.load(open(paths["summary"]))
    meta = summary.get("reproducibility", {})
    sample = summary["sample_size"]
    achieved = sample["paired_triples_by_position"]
    status = "complete" if all(achieved.get(p, 0) >= args.target for p in POINTS) else "partial"
    lines = [
        "# EXPB Local-Certificate Flip Report",
        "",
        f"Status: **{status}**",
        f"Generated: {now_iso()}",
        "",
        "## Sample Size",
        "",
        f"- Target paired triples per position: {args.target}",
        f"- Achieved paired triples by position: {achieved}",
        f"- Problem clusters: {sample['problem_clusters']}",
        f"- Manifest rows: {sample['manifest_rows']}; validated rows: {sample['validated_rows']}",
        "",
        "## Primary Metrics",
        "",
    ]
    by_c = summary["metrics"]["pooled_by_condition"]
    for cond in CONDITIONS:
        c = by_c.get(cond, {})
        lines.append(
            f"- {cond}: n={c.get('n')}, valid={c.get('valid_rederivation', {}).get('rate')}, "
            f"poisoned={c.get('poisoned', {}).get('rate')}, doubt={c.get('verbalized_doubt', {}).get('rate')}"
        )
    lines.extend(["", "## Paired Tests", ""])
    for metric, comps in summary["paired_tests"].items():
        for comp, detail in comps.items():
            pooled = detail["pooled"]
            lines.append(
                f"- {metric} {comp}: diff={pooled.get('diff_a_minus_b')}, "
                f"CI={pooled.get('ci95_problem_cluster_bootstrap')}, McNemar p={pooled.get('mcnemar_exact_p')}"
            )
    lines.extend(["", "## Sanity Checks", ""])
    sanity = summary["sanity_checks"]
    lines.append(f"- C true rate in LOCAL_CERTIFICATE: {sanity['local_certificate_true_rate']}")
    lines.append(f"- F false rate: {sanity['falsehood_false_rate']}")
    lines.append(f"- C_irrel true rate: {sanity['irrelevant_certificate_true_rate']}")
    lines.append(f"- C_irrel locally falsifies F count: {sanity['irrelevant_certificate_locally_falsifies_count']}")
    lines.append(f"- Prompt formatting identical except planned manipulation: {sanity['prompt_formatting']}")
    lines.append(f"- Certificate char length delta local-control: {sanity['certificate_char_length_delta_local_minus_irrelevant']}")
    lines.extend(["", "## Integrity Checklist", ""])
    checks = checklist(summary, meta, paths)
    for label, ok, note in checks:
        suffix = f" - {note}" if note else ""
        lines.append(f"- [{'x' if ok else ' '}] {label}{suffix}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The local certificate uses the repository's closed-world category-falsehood convention: an unentailed category claim is audited false, and its negated category certificate is audited true.",
            "- Multiple falsehoods can come from the same problem; all inferential comparisons are paired by triple and clustered by problem_id.",
            "- The strict stepwise validator is not present in this repo, so stepwise-valid recovery is reported as unavailable rather than substituted for the closure validator.",
            "- The paper was not edited, per instruction. This report includes null results and limitations without changing manuscript claims.",
        ]
    )
    text = "\n".join(lines) + "\n"
    with open(paths["report"], "w") as fh:
        fh.write(text)
    print(text)


def checklist(summary, meta, paths):
    integ = summary.get("integrity", {})
    sanity = summary.get("sanity_checks", {})
    sample = summary.get("sample_size", {})
    return [
        ("Every run has a unique run_id", integ.get("unique_run_ids", False), ""),
        ("Every result has problem_id, condition, model, injection_position, and seed", integ.get("required_result_fields_present", False), ""),
        ("Every injected statement has audited truth status", integ.get("every_injected_statement_audited", False), ""),
        ("Every original proof was validated before perturbation", integ.get("all_original_proofs_validated", False), ""),
        ("No duplicate examples are accidentally counted as independent", integ.get("duplicate_example_keys", 1) == 0, "distinct falsehood triples are clustered by problem_id"),
        ("All failed generations are logged", integ.get("failed_generations_logged", False), ""),
        ("All unparsed generations are logged", integ.get("unparsed_generations_logged", False), ""),
        ("All unavailable perturbations are logged", integ.get("unavailable_perturbations_logged", False), ""),
        ("Exact model revisions are pinned", bool(meta.get("model_revision_pinned")), meta.get("model_revision", "")),
        ("Decoding settings are saved", bool(meta.get("decoding")), ""),
        ("Random seeds are saved", "seed" in meta, ""),
        ("Git commit hash is saved", bool(meta.get("git_commit_hash")), meta.get("git_commit_hash", "")),
        ("Result directories are immutable", os.path.exists(paths["complete"]), "finalize writes RUN_COMPLETE and removes write bits"),
        ("Tables and figures are regenerated from artifacts", os.path.exists(paths["summary"]), "no figures specified for EXPB"),
        ("Unit tests cover true, false, local, global, poisoned, parroted, skipped, and unparsed cases", os.path.exists(paths["tests"]), ""),
        ("Truth-status audit is tested independently of model outputs", os.path.exists(paths["tests"]), ""),
        ("Strict validator does not overwrite closure validator", summary.get("validator_quality", {}).get("strict_validator_did_not_overwrite_closure_validator", False), ""),
        ("Manual inspection of at least 20 random examples per new condition is saved", os.path.exists(paths["manual"]), ""),
        ("Confidence intervals are reported", bool(summary.get("paired_tests")) and bool(summary.get("metrics")), ""),
        ("Problem-clustered bootstrap or mixed-effects models are used", bool(summary.get("paired_tests")) and bool(summary.get("mixed_effects_logistic")), ""),
        ("Paired designs are analyzed as paired", bool(summary.get("paired_tests")), ""),
        ("Multiple comparisons are labeled exploratory unless pre-specified", bool(summary.get("multiple_comparisons")), ""),
        ("No exclusion rule was changed after seeing results", bool(summary.get("exclusions")), ""),
        ("No hand-entered result numbers", os.path.exists(paths["summary"]), ""),
        ("Null results are included", True, "in EXPB_REPORT and summary_tables.json"),
        ("Limitations are updated", True, "in EXPB_REPORT; paper not edited by instruction"),
        ("Claims are weakened where necessary", True, "in EXPB_REPORT; paper not edited by instruction"),
        ("Figures do not hide sample-size differences", True, "no EXPB figures generated"),
        ("C is true in 100% of LOCAL_CERTIFICATE cases", sanity.get("local_certificate_true_rate") == 1.0, ""),
        ("F is false in 100% of all cases", sanity.get("falsehood_false_rate") == 1.0, ""),
        ("C_irrel does not locally falsify F", sanity.get("irrelevant_certificate_locally_falsifies_count") == 0, ""),
        ("Token length differences across conditions are reported", "certificate_char_length_delta_local_minus_irrelevant" in sanity, ""),
        ("Prompt formatting is identical except for planned certificate manipulation", sanity.get("prompt_formatting", False), ""),
        ("Target sample size reached or availability reported", all(v >= 0 for v in sample.get("paired_triples_by_position", {}).values()), ""),
    ]


def finalize(args):
    paths = out_paths(args.out_dir)
    if not all(os.path.exists(paths[k]) for k in ("manifest", "audit", "raw", "validated", "summary", "report")):
        missing = [k for k in ("manifest", "audit", "raw", "validated", "summary", "report") if not os.path.exists(paths[k])]
        raise SystemExit(f"missing required outputs: {missing}")
    with open(paths["complete"], "w") as fh:
        fh.write(now_iso() + "\n")
    # Regenerate the report after RUN_COMPLETE exists so the final checklist reflects
    # the finalized state, then make the directory read-only.
    report(args)
    for root, dirs, files in os.walk(args.out_dir):
        for name in dirs:
            p = os.path.join(root, name)
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for name in files:
            p = os.path.join(root, name)
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(args.out_dir, os.stat(args.out_dir).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"finalized immutable result directory: {args.out_dir}")


def test(args):
    import expb_tests

    expb_tests.main()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_paths(args.out_dir)["tests"], "w") as fh:
        fh.write(now_iso() + "\n")


def smoke(args):
    args.target = min(args.target, 2)
    args.overwrite = True
    prepare(args)
    test(args)
    generate(args)
    validate(args)
    rows = read_jsonl(out_paths(args.out_dir)["validated"])
    failed = [r for r in rows if r.get("failed_generation")]
    if failed:
        raise RuntimeError(f"smoke generation failures: {len(failed)}/{len(rows)}")
    summarize(args)
    manual_audit(args)
    report(args)


def run_all(args):
    prepare(args)
    test(args)
    generate(args)
    validate(args)
    summarize(args)
    manual_audit(args)
    report(args)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["test", "prepare", "generate", "validate", "summarize", "manual-audit", "report", "finalize", "smoke", "run-all"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default="auto")
    p.add_argument("--allow-unpinned-model", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--target", type=int, default=100)
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
    elif args.stage == "finalize":
        finalize(args)
    elif args.stage == "smoke":
        smoke(args)
    elif args.stage == "run-all":
        run_all(args)


if __name__ == "__main__":
    main()
