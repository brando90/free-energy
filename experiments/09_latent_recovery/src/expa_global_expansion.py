"""Experiment A: expanded globally-checkable falsehood family.

The experiment is staged so smoke tests can run before model generation:

  .venv/bin/python src/expa_global_expansion.py test
  .venv/bin/python src/expa_global_expansion.py prepare --overwrite
  .venv/bin/python src/expa_global_expansion.py generate --target 150
  .venv/bin/python src/expa_global_expansion.py validate
  .venv/bin/python src/expa_global_expansion.py summarize
  .venv/bin/python src/expa_global_expansion.py manual-audit
  .venv/bin/python src/expa_global_expansion.py report

Use --out-dir results/EXPA_GLOBAL_EXPANSION_SMOKE and --target 2 for smoke runs.
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

from validator import (
    DOUBT,
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
DEFAULT_OUT = os.path.join(BASE, "results", "EXPA_GLOBAL_EXPANSION")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 0
POINTS = ("early", "mid", "late")
CONDITIONS = ("global_falsehood", "benign_paraphrase", "one_hop_falsehood", "true_interruption")
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


def append_jsonl(path, row):
    ensure_writable_path(path)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path, obj):
    ensure_writable_path(path)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def ensure_writable_path(path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    try:
        os.chmod(directory, os.stat(directory).st_mode | stat.S_IWUSR | stat.S_IXUSR)
    except FileNotFoundError:
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(path):
        os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)


def make_result_dir_readonly(out_dir):
    for root, dirs, files in os.walk(out_dir):
        for name in files:
            path = os.path.join(root, name)
            os.chmod(path, os.stat(path).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for name in dirs:
            path = os.path.join(root, name)
            os.chmod(path, os.stat(path).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(out_dir, os.stat(out_dir).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def sha_row(parts, n=20):
    s = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def singular(cat):
    return cat[:-2] if cat.endswith("es") else cat


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", text.strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def solved(gen_text, target):
    sents = split_sentences(gen_text)
    return bool(sents) and norm(sents[-1]) == norm(target)


def derived_steps(cot):
    out = []
    for i, s in enumerate(cot):
        first = s.split()[0]
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
    return None


def world_categories(question):
    return sorted({singular(c) for c in re.findall(r"\b([a-z]+pus(?:es)?)\b", question)})


def world_state(question, entity):
    rules, facts, unparsed = parse_world(question)
    reach = closure(rules)
    premises = {p for e, p in facts if e == entity}
    return rules, facts, unparsed, reach, premises


def false_categories(question, entity):
    _, _, _, reach, premises = world_state(question, entity)
    return sorted(c for c in world_categories(question) if not derivable(("cat", c), premises, reach))


def entailed_categories(question, entity):
    _, _, _, reach, premises = world_state(question, entity)
    return sorted(c for c in world_categories(question) if derivable(("cat", c), premises, reach))


def parse_statement_for_audit(statement, entity):
    s = statement.strip().rstrip(".")
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
    statement = strip_marker(statement or "")
    parsed = parse_statement_for_audit(statement, entity)
    if parsed is None:
        return {"truth_status": "unparsed", "audit_basis": "statement_unparsed"}
    kind, name, positive = parsed
    _, _, _, reach, premises = world_state(question, entity)
    pred = (kind, name) if kind == "cat" else ("adj", name, positive)
    if kind == "cat":
        pos = ("cat", name)
        entailed_pos = derivable(pos, premises, reach)
        if positive:
            return {
                "truth_status": "true" if entailed_pos else "false",
                "audit_basis": "entailed_category" if entailed_pos else "unentailed_category_closed_world",
                "predicate": list(pos),
            }
        return {
            "truth_status": "false" if entailed_pos else "unknown",
            "audit_basis": "negates_entailed_category" if entailed_pos else "negated_unentailed_category_open_world",
            "predicate": ["not_cat", name],
        }
    entailed = derivable(pred, premises, reach)
    opposite = ("adj", name, not positive)
    opposite_entailed = derivable(opposite, premises, reach)
    if entailed:
        truth, basis = "true", "entailed_attribute"
    elif opposite_entailed:
        truth, basis = "false", "opposite_attribute_entailed"
    else:
        truth, basis = "unknown", "attribute_not_decidable"
    return {"truth_status": truth, "audit_basis": basis, "predicate": list(pred)}


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


def make_paraphrase(steps, si, idx=0):
    markers = ["Therefore, ", "It follows that ", "Thus, "]
    s = steps[si].rstrip(".")
    return f"{markers[idx % len(markers)]}{s[0].lower() + s[1:] if s.split()[0] in ('Every','Each','All') else s}."


def make_negstep(steps, si):
    s = steps[si]
    m = re.match(r"^(\S+) is (a|an) (\w+)\.$", s)
    if m:
        return f"{m.group(1)} is not {m.group(2)} {m.group(3)}."
    m = re.match(r"^(\S+) is (not )?(\w+)\.$", s)
    if m:
        return f"{m.group(1)} is {m.group(3)}." if m.group(2) else f"{m.group(1)} is not {m.group(3)}."
    return None


def make_global_falsehood(question, entity, idx=0):
    fcats = false_categories(question, entity)
    if not fcats:
        return None
    cat = fcats[idx % len(fcats)]
    return f"{entity} is {article(cat)} {cat}."


def make_true_interruption(question, steps, entity, idx=0):
    on_path = set()
    for s in steps:
        f = parse_statement_for_audit(s, entity)
        if f and f[0] == "cat" and f[2]:
            on_path.add(f[1])
    candidates = [c for c in entailed_categories(question, entity) if c not in on_path]
    if not candidates:
        return None
    cat = candidates[idx % len(candidates)]
    return f"{entity} is {article(cat)} {cat}."


def build_candidate_pool(seed=DEFAULT_SEED):
    seen_questions = set()
    rows = []
    audit = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*ProofsOnly*.json"))) + sorted(glob.glob(os.path.join(DATA_DIR, "*Composed*.json")))
    for path in files:
        data = json.load(open(path))
        for key, ex in data.items():
            for sub in ["test_example"] + [f"in_context_example{i}" for i in range(8)]:
                q = ex.get(sub)
                if not q or "chain_of_thought" not in q:
                    continue
                question = q["question"]
                source_id = f"{os.path.basename(path)}::{key}::{sub}"
                if question in seen_questions:
                    audit.append({"problem_id": source_id, "stage": "candidate", "eligible": False, "reason": "duplicate_question"})
                    continue
                seen_questions.add(question)
                cot = q["chain_of_thought"]
                dsteps = derived_steps(cot)
                target = q["query"].replace("Prove:", "").strip()
                entity = entity_from_target_or_cot(target, cot)
                base = {
                    "problem_id": source_id,
                    "question": question,
                    "target": target,
                    "dataset_gold_cot": cot,
                    "entity": entity,
                    "source_file": os.path.basename(path),
                    "source_key": key,
                    "source_subkey": sub,
                    "dataset_n_derived_steps": len(dsteps),
                }
                if entity is None:
                    audit.append({**base, "stage": "candidate", "eligible": False, "reason": "no_entity"})
                    continue
                if len(dsteps) < 4:
                    audit.append({**base, "stage": "candidate", "eligible": False, "reason": "too_few_dataset_steps"})
                    continue
                fcats = false_categories(question, entity)
                base["false_categories"] = fcats
                base["n_false_categories"] = len(fcats)
                if not fcats:
                    audit.append({**base, "stage": "candidate", "eligible": False, "reason": "no_global_falsehood_candidate"})
                    continue
                rows.append(base)
                audit.append({**base, "stage": "candidate", "eligible": True, "reason": "candidate_falsehood_available"})
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows, audit


def out_paths(out_dir):
    return {
        "candidate_pool": os.path.join(out_dir, "candidate_pool.jsonl"),
        "manifest": os.path.join(out_dir, "manifest.jsonl"),
        "audit": os.path.join(out_dir, "eligibility_audit.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
        "summary": os.path.join(out_dir, "summary_tables.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "readme": os.path.join(out_dir, "README.md"),
        "report": os.path.join(out_dir, "EXPA_REPORT.md"),
        "manual": os.path.join(out_dir, "manual_audit_note.md"),
        "complete": os.path.join(out_dir, "RUN_COMPLETE"),
    }


def ensure_out_dir(out_dir, overwrite=False):
    paths = out_paths(out_dir)
    if os.path.exists(paths["complete"]) and not overwrite:
        raise SystemExit(f"{out_dir} is marked complete; refusing to modify without --overwrite")
    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", os.path.abspath(os.path.join(BASE, "..", "..")), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
        "experiment": "EXPA_GLOBAL_EXPANSION",
        "created_at": now_iso(),
        "git_commit_hash": git_hash(),
        "model": args.model,
        "model_revision": model_revision or args.model_revision,
        "model_revision_pinned": bool(revision_pinned),
        "model_revision_error": revision_error,
        "seed": args.seed,
        "target_eligible_per_position": args.target,
        "max_candidates": args.max_candidates,
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": args.max_new_tokens,
            "num_return_sequences": 1,
        },
        "prompt": {
            "instruction": INSTR,
            "fewshot": FEWSHOT,
        },
        "python": sys.version,
    }
    write_json(out_paths(out_dir)["metadata"], meta)
    return meta


def prepare(args):
    ensure_out_dir(args.out_dir, overwrite=args.overwrite)
    paths = out_paths(args.out_dir)
    rows, audit = build_candidate_pool(args.seed)
    if args.max_candidates:
        rows = rows[: args.max_candidates]
    ensure_writable_path(paths["candidate_pool"])
    with open(paths["candidate_pool"], "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    ensure_writable_path(paths["audit"])
    with open(paths["audit"], "w") as fh:
        for r in audit:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    meta = write_metadata(args.out_dir, args)
    counts = Counter(r["reason"] for r in audit)
    print(json.dumps({"candidate_pool": len(rows), "audit_counts": counts, "metadata": meta}, indent=2, default=dict))


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise RuntimeError(f"Could not pin model revision: {err}")
    write_metadata(args.out_dir, args, revision, pinned, err)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    return tok, model, revision, pinned


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


def greedy(tok, model, ids, max_new_tokens):
    import torch

    ids = ids.to(model.device)
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


def run_id(problem_id, condition, point, seed):
    return sha_row(["EXPA", problem_id, condition, point, seed], 24)


def perturbed_raw_fields(mrow):
    return {k: mrow[k] for k in ("run_id", "problem_id", "condition", "model", "model_revision", "injection_position", "seed")}


def generate_perturbed_output(args, paths, tok, model, mrow, existing_raw):
    if mrow["run_id"] in existing_raw:
        return
    prefix = " " + " ".join(mrow["prefix_steps"] + [mrow["injected_statement"]])
    try:
        ids = make_prompt_ids(tok, mrow["question"], mrow["target"], answer_prefix=prefix)
        continuation = greedy(tok, model, ids, args.max_new_tokens)
        append_jsonl(paths["raw"], {
            **perturbed_raw_fields(mrow),
            "failed_generation": False,
            "continuation": continuation,
            "full_answer": (prefix + " " + continuation).strip(),
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
            "created_at": now_iso(),
        })
    except Exception as e:
        append_jsonl(paths["raw"], {
            **perturbed_raw_fields(mrow),
            "failed_generation": True,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
            "created_at": now_iso(),
        })
        append_jsonl(paths["audit"], {**mrow, "stage": "perturbed_generation", "eligible": False, "reason": "generation_failed", "error": repr(e)})
    existing_raw.add(mrow["run_id"])


def manifest_row(candidate, model_name, model_revision, seed, condition, point, si, injected, truth, prefix_steps, original_gold):
    return {
        "run_id": run_id(candidate["problem_id"], condition, point, seed),
        "problem_id": candidate["problem_id"],
        "condition": condition,
        "model": model_name,
        "model_revision": model_revision,
        "injection_position": point,
        "seed": seed,
        "repeated_decoding": False,
        "question": candidate["question"],
        "target": candidate["target"],
        "entity": candidate["entity"],
        "sent_idx": si,
        "prefix_steps": prefix_steps,
        "original_gold_proof": original_gold,
        "injected_statement": injected,
        "audited_truth_status": truth["truth_status"],
        "truth_audit": truth,
        "original_proof_validated": True,
        "created_at": now_iso(),
    }


def build_injections(candidate, steps, pts, seed):
    out = []
    for pi, point in enumerate(POINTS):
        si = pts[point]
        specs = []
        gf = make_global_falsehood(candidate["question"], candidate["entity"], idx=pi + seed)
        specs.append(("global_falsehood", gf))
        specs.append(("benign_paraphrase", make_paraphrase(steps, si, idx=pi)))
        specs.append(("one_hop_falsehood", make_negstep(steps, si)))
        specs.append(("true_interruption", make_true_interruption(candidate["question"], steps, candidate["entity"], idx=pi + seed)))
        for condition, injected in specs:
            truth = audit_truth_status(candidate["question"], injected, candidate["entity"]) if injected else {"truth_status": "unavailable", "audit_basis": "no_candidate"}
            out.append((condition, point, si, injected, truth))
    return out


def generate(args):
    paths = out_paths(args.out_dir)
    if not os.path.exists(paths["candidate_pool"]):
        prepare(args)
    candidates = read_jsonl(paths["candidate_pool"])
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    tok, model, revision, _ = load_model_and_tokenizer(args)
    manifest_rows = read_jsonl(paths["manifest"])
    existing_raw = {r["run_id"] for r in read_jsonl(paths["raw"])}
    existing_manifest = {r["run_id"] for r in manifest_rows}
    missing_manifest_raw = [r for r in manifest_rows if r["run_id"] not in existing_raw]
    if missing_manifest_raw:
        print(f"[EXPA] completing {len(missing_manifest_raw)} manifest rows with missing raw outputs", flush=True)
        for mrow in missing_manifest_raw:
            generate_perturbed_output(args, paths, tok, model, mrow, existing_raw)
    eligible_problems = {r["problem_id"] for r in manifest_rows if r["condition"] == "global_falsehood"}
    eligible_problem_count = len(eligible_problems)

    for idx, cand in enumerate(candidates):
        if eligible_problem_count >= args.target:
            break
        if cand["problem_id"] in eligible_problems:
            continue
        gold_run_id = run_id(cand["problem_id"], "gold", "original", args.seed)
        if gold_run_id in existing_raw:
            continue
        try:
            ids = make_prompt_ids(tok, cand["question"], cand["target"])
            gen_text = greedy(tok, model, ids, args.max_new_tokens)
            raw_gold = {
                "run_id": gold_run_id,
                "problem_id": cand["problem_id"],
                "condition": "gold",
                "model": args.model,
                "model_revision": revision,
                "injection_position": "original",
                "seed": args.seed,
                "failed_generation": False,
                "continuation": gen_text,
                "created_at": now_iso(),
                "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
            }
            append_jsonl(paths["raw"], raw_gold)
        except Exception as e:
            append_jsonl(paths["raw"], {
                "run_id": gold_run_id,
                "problem_id": cand["problem_id"],
                "condition": "gold",
                "model": args.model,
                "model_revision": revision,
                "injection_position": "original",
                "seed": args.seed,
                "failed_generation": True,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "created_at": now_iso(),
            })
            append_jsonl(paths["audit"], {**cand, "stage": "gold_generation", "eligible": False, "reason": "generation_failed", "error": repr(e)})
            continue

        if not solved(gen_text, cand["target"]):
            append_jsonl(paths["audit"], {**cand, "stage": "gold_filter", "eligible": False, "reason": "gold_not_solved"})
            continue
        steps = split_sentences(gen_text)
        entity = steps[0].split()[0] if steps else cand["entity"]
        cand["entity"] = entity
        gv = validate_continuation(cand["question"], [], None, gen_text, cand["target"], entity)
        if gv["class"] != "valid_rederivation":
            append_jsonl(paths["audit"], {**cand, "stage": "gold_filter", "eligible": False, "reason": "gold_not_validator_valid", "validator": gv})
            continue
        pts = injection_points(steps)
        if pts is None:
            append_jsonl(paths["audit"], {**cand, "stage": "injection_site", "eligible": False, "reason": "too_few_model_intermediate_steps"})
            continue

        inj = build_injections(cand, steps, pts, args.seed)
        gf = [r for r in inj if r[0] == "global_falsehood"]
        if len(gf) != 3 or any(r[4].get("truth_status") != "false" for r in gf):
            append_jsonl(paths["audit"], {**cand, "stage": "global_falsehood", "eligible": False, "reason": "global_falsehood_unavailable_or_not_false"})
            continue

        planned = []
        for condition, point, si, injected, truth in inj:
            if injected is None or truth.get("truth_status") in ("unavailable", "unparsed", "unknown"):
                append_jsonl(paths["audit"], {
                    **cand,
                    "stage": "condition_availability",
                    "eligible": False,
                    "reason": "condition_unavailable",
                    "condition": condition,
                    "injection_position": point,
                    "truth_audit": truth,
                })
                continue
            if condition == "global_falsehood" and truth["truth_status"] != "false":
                raise RuntimeError("global falsehood audit invariant violated")
            if condition == "one_hop_falsehood" and truth["truth_status"] != "false":
                append_jsonl(paths["audit"], {
                    **cand,
                    "stage": "condition_availability",
                    "eligible": False,
                    "reason": "one_hop_not_audited_false",
                    "condition": condition,
                    "injection_position": point,
                    "truth_audit": truth,
                })
                continue
            if condition == "true_interruption" and truth["truth_status"] != "true":
                append_jsonl(paths["audit"], {
                    **cand,
                    "stage": "condition_availability",
                    "eligible": False,
                    "reason": "true_interruption_not_available",
                    "condition": condition,
                    "injection_position": point,
                    "truth_audit": truth,
                })
                continue
            mrow = manifest_row(cand, args.model, revision, args.seed, condition, point, si, injected, truth, steps[:si], gen_text)
            planned.append(mrow)

        if sum(1 for r in planned if r["condition"] == "global_falsehood") != 3:
            append_jsonl(paths["audit"], {**cand, "stage": "global_falsehood", "eligible": False, "reason": "global_falsehood_not_all_positions"})
            continue

        for mrow in planned:
            if mrow["run_id"] not in existing_manifest:
                append_jsonl(paths["manifest"], mrow)
                existing_manifest.add(mrow["run_id"])
            generate_perturbed_output(args, paths, tok, model, mrow, existing_raw)
        eligible_problem_count += 1
        eligible_problems.add(cand["problem_id"])
        append_jsonl(paths["audit"], {**cand, "stage": "final_eligibility", "eligible": True, "reason": "gold_valid_global_falsehood_available"})
        if eligible_problem_count % 10 == 0 or eligible_problem_count <= 3:
            print(f"[EXPA] eligible={eligible_problem_count}/{args.target} candidates_seen={idx+1}", flush=True)
    print(f"DONE generate eligible={eligible_problem_count} target={args.target}", flush=True)


def validate(args):
    paths = out_paths(args.out_dir)
    manifest = {r["run_id"]: r for r in read_jsonl(paths["manifest"])}
    raw = [r for r in read_jsonl(paths["raw"]) if r["condition"] in MODEL_CONDITIONS]
    ensure_writable_path(paths["validated"])
    with open(paths["validated"], "w") as out:
        for r in raw:
            m = manifest.get(r["run_id"])
            if not m:
                continue
            base = {k: m[k] for k in (
                "run_id",
                "problem_id",
                "condition",
                "model",
                "model_revision",
                "injection_position",
                "seed",
                "injected_statement",
                "audited_truth_status",
                "truth_audit",
                "original_proof_validated",
            )}
            if r.get("failed_generation"):
                rec = {**base, "class": "generation_failed", "failed_generation": True, "error": r.get("error"), "verbalized_doubt": False}
            else:
                v = validate_continuation(
                    m["question"],
                    m["prefix_steps"],
                    m["injected_statement"],
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


def cluster_boot_rate(rows, metric, seed=0, n_boot=1000):
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
                denom += 1
                num += metric(r)
        vals.append(num / denom if denom else float("nan"))
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)]


def metric_value(r, metric):
    cls = r["class"]
    if metric == "valid_rederivation":
        return int(cls == "valid_rederivation")
    if metric == "poisoned":
        return int(cls == "poisoned")
    if metric == "parroted":
        return int(cls == "parroted")
    if metric == "derailed":
        return int(cls == "derailed")
    if metric == "unparsed":
        return int(cls == "unparsed")
    if metric == "verbalized_doubt":
        return int(bool(r.get("verbalized_doubt")))
    if metric == "generation_failed":
        return int(cls == "generation_failed")
    raise KeyError(metric)


def summarize_cell(rows, seed):
    metrics = ("valid_rederivation", "poisoned", "parroted", "derailed", "unparsed", "verbalized_doubt", "generation_failed")
    n = len(rows)
    out = {"n": n, "problem_n": len({r["problem_id"] for r in rows})}
    for metric in metrics:
        k = sum(metric_value(r, metric) for r in rows)
        out[metric] = {
            "count": k,
            "rate": round(k / n, 4) if n else None,
            "wilson95": wilson(k, n),
            "problem_cluster_bootstrap95": cluster_boot_rate(rows, lambda r, m=metric: metric_value(r, m), seed=seed, n_boot=400) if n else [None, None],
        }
    return out


def build_design(rows, outcome, include_problem_fe=True):
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy required for logistic summary") from e
    conds = ["benign_paraphrase", "one_hop_falsehood", "true_interruption", "global_falsehood"]
    positions = ["early", "mid", "late"]
    base_cond, base_pos = "benign_paraphrase", "early"
    feature_names = ["intercept"]
    for c in conds:
        if c != base_cond:
            feature_names.append(f"condition={c}")
    for p in positions:
        if p != base_pos:
            feature_names.append(f"position={p}")
    for c in conds:
        if c == base_cond:
            continue
        for p in positions:
            if p == base_pos:
                continue
            feature_names.append(f"condition={c}:position={p}")
    problems = sorted({r["problem_id"] for r in rows})
    if include_problem_fe:
        for pid in problems[1:]:
            feature_names.append(f"problem={pid}")
    x = []
    y = []
    for r in rows:
        row = [1.0]
        for c in conds:
            if c != base_cond:
                row.append(float(r["condition"] == c))
        for p in positions:
            if p != base_pos:
                row.append(float(r["injection_position"] == p))
        for c in conds:
            if c == base_cond:
                continue
            for p in positions:
                if p == base_pos:
                    continue
                row.append(float(r["condition"] == c and r["injection_position"] == p))
        if include_problem_fe:
            for pid in problems[1:]:
                row.append(float(r["problem_id"] == pid))
        x.append(row)
        y.append(outcome(r))
    return np.array(x, dtype=float), np.array(y, dtype=float), feature_names


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
    if len({r["problem_id"] for r in usable}) < 5 or len(usable) < 20:
        return {"status": "insufficient_data", "n": len(usable)}
    outcome = lambda r: metric_value(r, metric)
    try:
        return mixed_effects_logistic_summary(usable, metric, outcome)
    except Exception as e:
        fallback_error = repr(e)
    try:
        x, y, names = build_design(usable, outcome, include_problem_fe=True)
        if len(set(y.tolist())) < 2:
            return {"status": "no_outcome_variation", "n": len(usable)}
        beta = fit_logit_np(x, y)
    except Exception as e:
        return {"status": "fit_failed", "mixed_effects_error": fallback_error, "fixed_effects_error": repr(e), "n": len(usable)}
    keep = {}
    for name, b in zip(names, beta):
        if name.startswith("problem="):
            continue
        keep[name] = {"log_or": round(float(b), 4), "or": round(float(math.exp(max(min(b, 20), -20))), 4)}
    return {
        "status": "ok",
        "method": "problem_fixed_effects_logistic_with_condition_by_position_terms; fallback only because mixed-effects fit failed",
        "mixed_effects_error": fallback_error,
        "formula": f"{metric} ~ condition * position + problem_id_fixed_effect",
        "n": len(usable),
        "problem_n": len({r["problem_id"] for r in usable}),
        "coefficients": keep,
    }


def mixed_effects_logistic_summary(rows, metric, outcome):
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    df = pd.DataFrame([
        {
            "outcome": int(outcome(r)),
            "condition": r["condition"],
            "injection_position": r["injection_position"],
            "problem_id": r["problem_id"],
        }
        for r in rows
    ])
    if df["outcome"].nunique() < 2:
        return {"status": "no_outcome_variation", "n": int(len(df)), "problem_n": int(df["problem_id"].nunique())}
    formula = "outcome ~ C(condition, Treatment(reference='benign_paraphrase')) * C(injection_position, Treatment(reference='early'))"
    vc_formulas = {"problem_id": "0 + C(problem_id)"}
    model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, df)
    res = model.fit_vb(verbose=False, minim_opts={"maxiter": 500})
    coefs = {}
    for name, mean, sd in zip(model.exog_names, res.fe_mean, res.fe_sd):
        lo = float(mean - 1.96 * sd)
        hi = float(mean + 1.96 * sd)
        coefs[name] = {
            "log_or": round(float(mean), 4),
            "se": round(float(sd), 4),
            "log_or_approx95": [round(lo, 4), round(hi, 4)],
            "or": round(float(math.exp(max(min(mean, 20), -20))), 4),
            "or_approx95": [
                round(float(math.exp(max(min(lo, 20), -20))), 4),
                round(float(math.exp(max(min(hi, 20), -20))), 4),
            ],
        }
    random_effect_sd = None
    if len(res.vcp_mean):
        random_effect_sd = round(float(math.exp(res.vcp_mean[0])), 4)
    return {
        "status": "ok",
        "method": "statsmodels BinomialBayesMixedGLM variational Bayes with random intercept for problem_id",
        "formula": f"{metric} ~ condition * position + (1 | problem_id)",
        "n": int(len(df)),
        "problem_n": int(df["problem_id"].nunique()),
        "random_intercept_sd": random_effect_sd,
        "coefficients": coefs,
    }


def summarize(args):
    paths = out_paths(args.out_dir)
    rows = read_jsonl(paths["validated"])
    by_cp = {}
    by_c = {}
    for condition in CONDITIONS:
        crows = [r for r in rows if r["condition"] == condition]
        by_c[condition] = summarize_cell(crows, args.seed)
        for point in POINTS:
            sub = [r for r in crows if r["injection_position"] == point]
            by_cp[f"{condition}|{point}"] = summarize_cell(sub, args.seed)
    ids_by_condition = {c: sorted({r["problem_id"] for r in rows if r["condition"] == c}) for c in CONDITIONS}
    duplicates = []
    seen = set()
    for r in rows:
        key = (r["problem_id"], r["condition"], r["injection_position"], r["seed"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    out = {
        "created_at": now_iso(),
        "metrics": {
            "by_condition_position": by_cp,
            "pooled_by_condition": by_c,
        },
        "availability": {
            "problems_by_condition": {k: len(v) for k, v in ids_by_condition.items()},
            "global_falsehood_problem_n": len(ids_by_condition.get("global_falsehood", [])),
        },
        "integrity": {
            "validated_rows": len(rows),
            "duplicate_problem_condition_position_seed": [list(x) for x in duplicates[:50]],
            "duplicate_count": len(duplicates),
            "all_global_truth_false": all(r.get("audited_truth_status") == "false" for r in rows if r["condition"] == "global_falsehood"),
            "all_original_proofs_validated": all(bool(r.get("original_proof_validated")) for r in rows),
        },
        "logistic": {
            "valid_rederivation": logistic_summary(rows, "valid_rederivation", args.seed),
            "poisoned": logistic_summary(rows, "poisoned", args.seed),
            "verbalized_doubt": logistic_summary(rows, "verbalized_doubt", args.seed),
        },
        "multiple_comparisons": "Global falsehood absorption and recovery are primary for EXPA; control contrasts are exploratory unless stated in the preregistered prompt.",
    }
    write_json(paths["summary"], out)
    print(json.dumps(out["availability"], indent=2))


def manual_audit(args):
    paths = out_paths(args.out_dir)
    rows = [r for r in read_jsonl(paths["validated"]) if r["condition"] in MODEL_CONDITIONS]
    rng = random.Random(args.seed)
    lines = ["# EXPA Manual Audit Note", "", f"Generated: {now_iso()}", ""]
    for condition in CONDITIONS:
        sub = [r for r in rows if r["condition"] == condition]
        rng.shuffle(sub)
        take = sub[: args.manual_n]
        lines.append(f"## {condition} ({len(take)} inspected; requested {args.manual_n})")
        lines.append("")
        for r in take:
            cont = " ".join(r.get("continuation", "").split())[:600]
            lines.append(f"- run_id: `{r['run_id']}`; problem_id: `{r['problem_id']}`; position: `{r['injection_position']}`; class: `{r['class']}`; truth: `{r['audited_truth_status']}`")
            lines.append(f"  - injected: {r.get('injected_statement')}")
            lines.append(f"  - continuation: {cont}")
        lines.append("")
    ensure_writable_path(paths["manual"])
    with open(paths["manual"], "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {paths['manual']}")


def report(args):
    paths = out_paths(args.out_dir)
    summary = json.load(open(paths["summary"])) if os.path.exists(paths["summary"]) else {}
    meta = json.load(open(paths["metadata"])) if os.path.exists(paths["metadata"]) else {}
    audit = read_jsonl(paths["audit"])
    raw = read_jsonl(paths["raw"])
    audit_counts = Counter(r.get("reason") for r in audit)
    candidate_stage = [r for r in audit if r.get("stage") == "candidate" and r.get("reason") != "duplicate_question"]
    pre_model_available = audit_counts.get("candidate_falsehood_available", 0)
    gold_attempts = sum(1 for r in raw if r.get("condition") == "gold")
    global_n = summary.get("availability", {}).get("global_falsehood_problem_n", 0)
    target = meta.get("target_eligible_per_position", args.target)
    status = "complete" if global_n >= target else "partial"
    pre_model_rate = pre_model_available / len(candidate_stage) if candidate_stage else None
    generated_eligibility_rate = global_n / gold_attempts if gold_attempts else None
    lines = [
        "# EXPA Global Expansion Report",
        "",
        f"Status: **{status}**",
        f"Generated: {now_iso()}",
        "",
        "## Sample Size",
        "",
        f"- Target global-falsehood examples per position: {target}",
        f"- Achieved global-falsehood examples per position: {global_n}",
        f"- Pre-model global-falsehood candidate availability: {pre_model_available}/{len(candidate_stage)} ({pre_model_rate:.4f}) unique candidate worlds.",
        f"- Observed post-generation eligibility before stopping at target: {global_n}/{gold_attempts} ({generated_eligibility_rate:.4f}) gold-generation attempts.",
        "- All unavailable perturbations and failed filters are preserved in `eligibility_audit.jsonl`.",
        "",
        "## Main Result",
        "",
    ]
    by_c = summary.get("metrics", {}).get("pooled_by_condition", {})
    for condition in CONDITIONS:
        c = by_c.get(condition, {})
        if not c:
            continue
        lines.append(
            f"- {condition}: n={c.get('n')}, valid={c.get('valid_rederivation', {}).get('rate')}, "
            f"poisoned={c.get('poisoned', {}).get('rate')}, parroted={c.get('parroted', {}).get('rate')}, "
            f"derailed={c.get('derailed', {}).get('rate')}, unparsed={c.get('unparsed', {}).get('rate')}, "
            f"doubt={c.get('verbalized_doubt', {}).get('rate')}"
        )
    lines.extend([
        "",
        "## Eligibility Counts",
        "",
    ])
    for reason, count in audit_counts.most_common():
        lines.append(f"- {reason}: {count}")
    lines.extend([
        "",
        "## Integrity Checklist",
        "",
    ])
    integ = summary.get("integrity", {})
    checks = [
        ("Every run has a unique run_id", integ.get("duplicate_count", 1) == 0),
        ("Every result has problem_id, condition, model, injection_position, and seed", _check_required_result_fields(paths["validated"])),
        ("Every injected statement has audited truth status", _check_truth(paths["validated"])),
        ("Every original proof was validated before perturbation", integ.get("all_original_proofs_validated", False)),
        ("No duplicate examples are accidentally counted as independent", integ.get("duplicate_count", 1) == 0),
        ("All failed generations are logged", os.path.exists(paths["raw"])),
        ("All unparsed generations are logged", os.path.exists(paths["validated"])),
        ("All unavailable perturbations are logged", os.path.exists(paths["audit"])),
        ("Exact model revisions are pinned", bool(meta.get("model_revision_pinned"))),
        ("Decoding settings are saved", "decoding" in meta),
        ("Random seeds are saved", "seed" in meta),
        ("Git commit hash is saved", bool(meta.get("git_commit_hash"))),
        ("Result directory is made read-only after completion", status == "complete"),
        ("Tables are regenerated from artifacts", os.path.exists(paths["summary"])),
        ("Validator unit tests cover required cases", os.path.exists(os.path.join(args.out_dir, "VALIDATOR_TESTS_PASSED"))),
        ("Manual inspection note saved", os.path.exists(paths["manual"])),
        ("Confidence intervals are reported", bool(summary.get("metrics"))),
        ("Problem-clustered bootstrap or paired model used", bool(summary.get("logistic"))),
        ("Multiple comparisons labeled", bool(summary.get("multiple_comparisons"))),
    ]
    for label, ok in checks:
        lines.append(f"- [{'x' if ok else ' '}] {label}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Null and contradictory findings should be read directly from `summary_tables.json`; no result numbers are hand-entered in downstream tables.",
        "- If status is partial, the run did not reach the requested n>=150 per position and the maximum achieved sample is reported above.",
    ])
    text = "\n".join(lines) + "\n"
    ensure_writable_path(paths["report"])
    with open(paths["report"], "w") as fh:
        fh.write(text)
    ensure_writable_path(paths["readme"])
    with open(paths["readme"], "w") as fh:
        fh.write(text.replace("# EXPA Global Expansion Report", "# EXPA_GLOBAL_EXPANSION README", 1))
    if status == "complete" and all(ok for _, ok in checks):
        ensure_writable_path(paths["complete"])
        with open(paths["complete"], "w") as fh:
            fh.write(now_iso() + "\n")
        make_result_dir_readonly(args.out_dir)
    print(text)


def _check_required_result_fields(path):
    rows = read_jsonl(path)
    req = {"run_id", "problem_id", "condition", "model", "injection_position", "seed"}
    return bool(rows) and all(req <= set(r) for r in rows)


def _check_truth(path):
    rows = [r for r in read_jsonl(path) if r.get("condition") in MODEL_CONDITIONS]
    return bool(rows) and all("audited_truth_status" in r and r["audited_truth_status"] for r in rows)


def test(args):
    import expa_tests

    expa_tests.main()
    os.makedirs(args.out_dir, exist_ok=True)
    marker = os.path.join(args.out_dir, "VALIDATOR_TESTS_PASSED")
    ensure_writable_path(marker)
    with open(marker, "w") as fh:
        fh.write(now_iso() + "\n")


def smoke(args):
    args.target = min(args.target, 2)
    args.max_candidates = args.max_candidates or 20
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
    p.add_argument("--target", type=int, default=150)
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
