"""EXPD matched refutation-distance gradient - staged runner (200-world PILOT).

Staged usage (EXPC pattern):

  python expd_matched_gradient.py test      --out-dir <dir>
  python expd_matched_gradient.py prepare   --out-dir <dir> [--overwrite]
  python expd_matched_gradient.py generate  --out-dir <dir> --backend vllm
  python expd_matched_gradient.py validate  --out-dir <dir>
  python expd_matched_gradient.py summarize --out-dir <dir>
  python expd_matched_gradient.py report    --out-dir <dir>

PILOT scope (pre-registration-exempt per PLAN2 section 8: the pilot gates
go/no-go and tunes knobs - MOD-12 anchor margin, H4 MDE - it scores no
hypothesis). The FULL confirmatory grid is refused without
--confirm-timestamped-plan2 (PLAN2 section 9.10: external timestamp must
precede the first confirmatory EXPD gold generation).

Fail-closed audits at injection time (MC-4): measured shortest_rule_distance
from the actual gold prefix state == designed d; audited truth status ==
designed; planted statement parses; anchors audited true; rejected worlds are
counted, never replaced.

Cohort bookkeeping for MOD-13/MC-7: mirrored-pair joint-solve and gold-prefix
identity up to the injection site are computed and reported.

Backend note (MEASURED, tooling/replay_full): doubt absolute rates are
backend-sensitive (vLLM 8-15pp below locked HF levels; structural metrics
robust). The anchor TOST gate vs EXPA locked levels is therefore evaluated on
valid_recovery/unparsed only; the doubt comparison is reported without a
pass/fail verdict pending a vLLM-basis EXPA anchor replay.
"""
import argparse
import datetime as _dt
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_worlds_expd as genw  # noqa: E402  (also inserts src/ into sys.path)

SRC = genw.SRC
from validator import (  # noqa: E402
    parse_fact,
    strip_marker,
    validate_continuation,
)
from expc_polarity_control import (  # noqa: E402
    audit_truth_status,
    grammar_template,
    injection_points,
    shortest_rule_distance,
    statement_predicate,
    world_state,
)

REPO = os.path.dirname(SRC)
DEFAULT_OUT = os.path.join(REPO, "results", "EXPD_MATCHED_GRADIENT_PILOT")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 0
POINTS = ("early", "mid", "late")

# Prompt block: byte-identical to src/common.py / src/expc_polarity_control.py
# (required for EXPA anchor comparability; SHA-pinned in run_metadata).
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

PILOT_CELLS = (
    "aff_false_attr_d1",
    "aff_false_attr_d3",
    "aff_false_attr_d5",
    "neg_false_attr_d1",
    "cat_false_usable_d1",
    "cat_false_inert_d1",
    "benign_paraphrase",
    "true_interruption",
)
ANCHOR_CELLS = ("benign_paraphrase", "true_interruption")

# Locked EXPA anchor levels (HF backend), results/EXPA_GLOBAL_EXPANSION/summary_tables.json
EXPA_LOCKED_ANCHORS = {
    "benign_paraphrase": {"n": 450, "valid_rederivation": 0.8489, "verbalized_doubt": 0.08, "unparsed": 0.0244},
    "true_interruption": {"n": 450, "valid_rederivation": 0.8556, "verbalized_doubt": 0.02, "unparsed": 0.0133},
}
EXPA_PROVENANCE = "results/EXPA_GLOBAL_EXPANSION/summary_tables.json (locked, HF backend)"
VLLM_DOUBT_CAVEAT = (
    "Backend confound (measured, improvement_plan/tooling/replay_full/REPLAY_FULL_REPORT.md): "
    "doubt absolute rates drop 8-15pp under vLLM vs locked HF numbers while structural metrics "
    "(valid_recovery, poisoning, unparsed) are cell-level robust (<=5pp deltas). The doubt-anchor "
    "comparison vs EXPA locked HF levels is reported WITHOUT a pass/fail verdict; a vLLM-basis "
    "EXPA anchor replay is the pending fix (owned upstream). The anchor TOST gate is evaluated on "
    "valid_recovery and unparsed only. Do not fail the pilot go/no-go on a doubt-anchor gap alone."
)

LOCKED_DIR_PATTERNS = ("EXPA_GLOBAL_EXPANSION", "EXPC_POLARITY_CONTROL", "/src", "/data", "paper_latex")


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


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
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:n]


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", text.strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def prompt_hash():
    return hashlib.sha256((INSTR + FEWSHOT).encode()).hexdigest()


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_paraphrase(steps, si, idx=0):
    """common.make_paraphrase verbatim: same content, different surface form."""
    markers = ["Therefore, ", "It follows that ", "Thus, "]
    s = steps[si].rstrip(".")
    return "%s%s." % (markers[idx % len(markers)],
                      s[0].lower() + s[1:] if s.split()[0] in ("Every", "Each", "All") else s)


def out_paths(out_dir):
    return {
        "worlds_dir": os.path.join(out_dir, "worlds"),
        "world_manifest": os.path.join(out_dir, "worlds", "world_manifest.jsonl"),
        "world_audit": os.path.join(out_dir, "worlds", "world_audit.json"),
        "candidate_pool": os.path.join(out_dir, "candidate_pool.jsonl"),
        "audit": os.path.join(out_dir, "eligibility_audit.jsonl"),
        "gold_cohort": os.path.join(out_dir, "gold_cohort.jsonl"),
        "manifest": os.path.join(out_dir, "manifest.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
        "summary": os.path.join(out_dir, "summary_tables.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "report": os.path.join(out_dir, "EXPD_PILOT_REPORT.md"),
        "readme": os.path.join(out_dir, "README.md"),
        "tests_passed": os.path.join(out_dir, "EXPD_TESTS_PASSED"),
        "complete": os.path.join(out_dir, "RUN_COMPLETE"),
    }


def guard_out_dir(out_dir):
    ab = os.path.abspath(out_dir)
    for pat in LOCKED_DIR_PATTERNS:
        if pat in ab and "EXPD" not in ab:
            raise SystemExit("refusing to write into protected path: %s" % ab)
    if os.path.basename(ab).startswith("EXPD_MATCHED_GRADIENT") is False:
        print("[EXPD] warning: out dir %s is not results/EXPD_MATCHED_GRADIENT*" % ab)


def ensure_out_dir(out_dir, overwrite=False):
    guard_out_dir(out_dir)
    paths = out_paths(out_dir)
    if os.path.exists(paths["complete"]) and not overwrite:
        raise SystemExit("%s is marked complete; refusing to modify without --overwrite" % out_dir)
    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def refuse_complete(out_dir, overwrite=False):
    guard_out_dir(out_dir)
    if os.path.exists(out_paths(out_dir)["complete"]) and not overwrite:
        raise SystemExit("%s is marked complete; refusing to modify without --overwrite" % out_dir)


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
    full = getattr(args, "world_config", "pilot") == "full"
    prior = {}
    mp = out_paths(out_dir)["metadata"]
    if os.path.exists(mp):
        try:
            prior = json.load(open(mp))
        except Exception:
            prior = {}
    invocations = prior.get("invocations", [])
    invocations.append({"at": now_iso(), "cells": list(args.cells),
                        "positions": list(args.positions), "cap": args.cap,
                        "world_config": getattr(args, "world_config", "pilot")})
    meta = {
        "experiment": "EXPD_MATCHED_GRADIENT" if full else "EXPD_MATCHED_GRADIENT_PILOT",
        "status_note": (
            "FULL confirmatory grid (PLAN2 v1.1 section C.5), generated under the externally "
            "timestamped PLAN2 (section 9.10); timestamp-gate evidence recorded in "
            "pre_registration_evidence." if full else
            "200-world PILOT, pre-registration-exempt (PLAN2 section 8: gates go/no-go and "
            "tunes MOD-12 anchor margin + H4 MDE; scores no confirmatory hypothesis). The full "
            "confirmatory run requires the externally timestamped PLAN2 (section 9.10) and is "
            "refused by this runner without --confirm-timestamped-plan2."),
        "pre_registration_evidence": {
            "confirm_timestamped_plan2": bool(getattr(args, "confirm_timestamped_plan2", False)),
            "plan2_git_commit": getattr(args, "plan2_commit", None),
            "plan2_commit_timestamp": getattr(args, "plan2_commit_timestamp", None),
            "plan2_file_sha256": getattr(args, "plan2_sha256", None),
        },
        "invocations": invocations,
        "created_at": now_iso(),
        "git_commit_hash": git_hash(),
        "model": args.model,
        "model_revision": model_revision or args.model_revision,
        "model_revision_pinned": bool(revision_pinned),
        "model_revision_error": revision_error,
        "backend": args.backend,
        "backend_note": (
            "New suites run wholly under one backend; never mix backends within an experiment "
            "(vLLM port fidelity report). " + VLLM_DOUBT_CAVEAT),
        "seed": args.seed,
        "world_seed": args.seed,
        "world_config": args.world_config,
        "cells": list(args.cells),
        "positions": list(args.positions),
        "per_cell_cap": args.cap,
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "decoding": {"do_sample": False, "temperature": 0.0, "max_new_tokens": args.max_new_tokens,
                     "num_return_sequences": 1},
        "prompt": {"instruction": INSTR, "fewshot": FEWSHOT, "sha256": prompt_hash(),
                   "shared_across_conditions": True},
        "design_references": {
            "pre_registration": "PLAN2.md (sections 3, 4, 6, 8)",
            "design": "reports/causal2.md section 1.1",
            "amendments": "reports/verdict.md MAJOR-1, MAJOR-2, MOD-12, MOD-13",
            "distance_function": "src/expc_polarity_control.py shortest_rule_distance (canonical)",
        },
        "expa_locked_anchor_levels": {"values": EXPA_LOCKED_ANCHORS, "provenance": EXPA_PROVENANCE},
        "python": sys.version,
    }
    write_json(out_paths(out_dir)["metadata"], meta)
    return meta


# ------------------------------------------------------------------ backends

def find_tooling():
    cands = [
        os.environ.get("EXPD_TOOLING"),
        os.path.abspath(os.path.join(HERE, "..", "tooling")),
        os.path.abspath(os.path.join(REPO, "improvement_plan", "tooling")),
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "vllm_gen.py")):
            return c
    raise SystemExit("cannot locate tooling/vllm_gen.py; set EXPD_TOOLING")


class VllmBackend:
    name = "vllm"

    def __init__(self, args, revision):
        tooling = find_tooling()
        if tooling not in sys.path:
            sys.path.insert(0, tooling)
        import vllm_gen
        self.vg = vllm_gen
        self.model = args.model
        self.revision = revision
        self.tok = vllm_gen.get_tokenizer(args.model, revision)

    def render(self, question, target, prefix=None):
        ids = self.vg.render_prompt_ids(self.tok, question, target, prefix, INSTR, FEWSHOT)
        return {"prompt_token_ids": ids}

    def generate(self, rows, max_new_tokens):
        return self.vg.generate_continuations(self.model, rows, max_new_tokens=max_new_tokens,
                                              revision=self.revision)

    def token_count(self, text):
        return int(len(self.tok(text or "", add_special_tokens=False)["input_ids"]))


class HfBackend:
    name = "hf"

    def __init__(self, args, revision):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(args.seed)
        self.tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
        device_map = {"": args.device}
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, dtype=torch.bfloat16, device_map=device_map)
        self.model.eval()

    def render(self, question, target, prefix=None):
        user = INSTR + FEWSHOT + "Q: %s Prove: %s\nA:" % (question, target)
        out = self.tok.apply_chat_template([{"role": "user", "content": user}],
                                           add_generation_prompt=True, return_tensors="pt")
        ids = out["input_ids"] if (hasattr(out, "data") or isinstance(out, dict)) else out
        if prefix:
            pre = self.tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
            ids = self.torch.cat([ids, pre], dim=1)
        return {"ids": ids}

    def generate(self, rows, max_new_tokens):
        outs = []
        for r in rows:
            ids = r["ids"].to(next(self.model.parameters()).device)
            with self.torch.no_grad():
                out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids),
                                          max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            outs.append({"text": self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)})
        return outs

    def token_count(self, text):
        return int(len(self.tok(text or "", add_special_tokens=False)["input_ids"]))


def make_backend(args, revision):
    if args.backend == "vllm":
        return VllmBackend(args, revision)
    return HfBackend(args, revision)


# ------------------------------------------------------------------- prepare

def prepare(args):
    ensure_out_dir(args.out_dir, overwrite=args.overwrite)
    paths = out_paths(args.out_dir)
    if args.world_config == "full" and not args.confirm_timestamped_plan2:
        raise SystemExit(
            "REFUSED: full confirmatory grid requires the externally timestamped PLAN2 "
            "(PLAN2 section 9.10). Re-run with --confirm-timestamped-plan2 only after the "
            "public commit hash / OSF registration exists.")
    if args.world_config == "full" and not (
            args.plan2_commit and args.plan2_sha256 and args.plan2_commit_timestamp):
        raise SystemExit(
            "REFUSED: --confirm-timestamped-plan2 requires the evidence flags "
            "--plan2-commit, --plan2-commit-timestamp and --plan2-sha256 so the gate "
            "evidence is recorded in run_metadata.json.")
    if not os.path.exists(paths["world_manifest"]):
        genw.generate_and_write(paths["worlds_dir"], args.seed, args.world_config)
    worlds = read_jsonl(paths["world_manifest"])
    with open(paths["candidate_pool"], "w") as fh:
        for w in worlds:
            fh.write(json.dumps(w, sort_keys=True) + "\n")
    meta = write_metadata(args.out_dir, args)
    audit_doc = json.load(open(paths["world_audit"]))
    print(json.dumps({
        "worlds": len(worlds),
        "world_audit_failures": audit_doc["audit_failures"],
        "worlds_json_sha256": audit_doc["worlds_json_sha256"],
        "metadata_written": True,
        "experiment": meta["experiment"],
    }, indent=2))


# ------------------------------------------------------------------ generate

def gold_run_id(world_id, seed):
    return sha_row(["EXPD", world_id, "gold", "original", seed])


def run_id(world_id, condition, point, seed):
    return sha_row(["EXPD", world_id, condition, point, seed])


def batched_generate(backend, jobs, paths, args, existing_raw):
    """jobs: list of (raw_base_row, question, target, prefix). Resume-safe,
    chunked; per-row fallback on batch failure."""
    pending = [j for j in jobs if j[0]["run_id"] not in existing_raw]
    bs = max(1, args.batch_size)
    done = 0
    for i in range(0, len(pending), bs):
        chunk = pending[i:i + bs]
        rows = [backend.render(q, t, pre) for _, q, t, pre in chunk]
        try:
            outs = backend.generate(rows, args.max_new_tokens)
        except Exception:
            outs = []
            for row in rows:
                try:
                    outs.extend(backend.generate([row], args.max_new_tokens))
                except Exception as e:
                    outs.append({"text": None, "error": repr(e), "traceback": traceback.format_exc()})
        for (base, q, t, pre), out in zip(chunk, outs):
            if out.get("text") is None:
                rec = dict(base)
                rec.update({"failed_generation": True, "error": out.get("error"),
                            "traceback": out.get("traceback"), "created_at": now_iso()})
            else:
                rec = dict(base)
                rec.update({"failed_generation": False, "continuation": out["text"],
                            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                            "created_at": now_iso()})
            append_jsonl(paths["raw"], rec)
            existing_raw.add(base["run_id"])
        done += len(chunk)
        print("[EXPD] generated %d/%d" % (done, len(pending)), flush=True)
    return len(pending)


def build_gold_cohort(pool, raw_rows, seed):
    """Eligibility ladder per world version: gold solved AND validator-valid AND
    injection points available (EXPA double filter)."""
    gold = {}
    for r in raw_rows:
        if r.get("condition") == "gold" and not r.get("failed_generation"):
            gold[r["world_id"]] = r
    cohort = []
    for w in pool:
        rec = {
            "world_id": w["world_id"], "family_id": w["family_id"], "family_idx": w["family_idx"],
            "kind": w["kind"], "d": w["d"], "version": w["version"],
            "entity": w["entity"], "target": w["target"],
            "partner_world_id": w.get("partner_world_id"),
        }
        g = gold.get(w["world_id"])
        if g is None:
            rec.update({"gold_generated": False, "eligible": False, "reason": "gold_missing_or_failed"})
            cohort.append(rec)
            continue
        text = g.get("continuation", "")
        steps = split_sentences(text)
        solved = bool(steps) and norm(steps[-1]) == norm(w["target"])
        v = validate_continuation(w["question"], [], None, text, w["target"], w["entity"])
        pts = injection_points(steps, w["entity"])
        eligible = solved and v["class"] == "valid_rederivation" and pts is not None
        reason = ("eligible" if eligible else
                  "gold_not_solved" if not solved else
                  "gold_not_validator_valid" if v["class"] != "valid_rederivation" else
                  "too_few_intermediate_entity_steps")
        rec.update({
            "gold_generated": True, "solved": solved, "gold_class": v["class"],
            "gold_steps": steps, "injection_points": pts, "eligible": eligible, "reason": reason,
        })
        cohort.append(rec)
    return cohort


def _designed_cell_rows(pool, cells):
    """cell_name -> [(world_row, cell_spec)] ordered by family_idx then world_id."""
    out = defaultdict(list)
    for w in sorted(pool, key=lambda x: (x["family_idx"], x["world_id"])):
        for c in w["cells"]:
            if c["cell"] in cells:
                out[c["cell"]].append((w, c))
    return out


def plan_injections(pool, cohort, args, paths, backend, revision):
    """Build fail-closed manifest rows for the requested cells at the requested
    positions, capped per cell at args.cap distinct worlds."""
    coh = {r["world_id"]: r for r in cohort}
    planned = []
    reject = lambda w, cell, point, reason, extra=None: append_jsonl(paths["audit"], {
        "stage": "injection_audit", "eligible": False, "world_id": w["world_id"],
        "family_id": w["family_id"], "condition": cell, "injection_position": point,
        "reason": reason, "extra": extra, "created_at": now_iso()})

    def manifest_row(w, rec, cell_name, cell_meta, point, si, injected, truth, measured_d):
        return {
            "run_id": run_id(w["world_id"], cell_name, point, args.seed),
            "problem_id": w["world_id"],
            "world_id": w["world_id"],
            "family_id": w["family_id"],
            "family_idx": w["family_idx"],
            "partner_world_id": w.get("partner_world_id"),
            "condition": cell_name,
            "cell_group": cell_meta.get("group"),
            "designed_d": cell_meta.get("designed_d", "na"),
            "world_d": w["d"],
            "world_version": w["version"],
            "world_kind": w["kind"],
            "polarity": cell_meta.get("polarity", "na"),
            "claim_family": cell_meta.get("claim_family", "na"),
            "model": args.model,
            "model_revision": revision,
            "injection_position": point,
            "seed": args.seed,
            "question": w["question"],
            "target": w["target"],
            "entity": w["entity"],
            "sent_idx": si,
            "prefix_steps": rec["gold_steps"][:si],
            "gold_steps": rec["gold_steps"],
            "injected_statement": injected,
            "audited_truth_status": truth.get("truth_status"),
            "truth_audit": truth,
            "measured_d_from_prefix": measured_d,
            "grammar_template": grammar_template(injected, w["entity"]),
            "predicate_family": truth.get("predicate_family"),
            "injected_token_count": backend.token_count(injected),
            "injected_word_count": len(injected.split()),
            "poisoning_measurable": bool(cell_meta.get("claim_family") == "category"
                                         and w.get("version") == "usable"),
            "original_proof_validated": True,
            "prompt_sha256": prompt_hash(),
            "created_at": now_iso(),
        }

    # ---- designed false/true cells (from the world manifest)
    designed = _designed_cell_rows(pool, [c for c in args.cells if c not in ANCHOR_CELLS])
    for cell_name in [c for c in args.cells if c not in ANCHOR_CELLS]:
        n_taken = 0
        for w, cmeta in designed.get(cell_name, []):
            if n_taken >= args.cap:
                break
            rec = coh.get(w["world_id"])
            if not rec or not rec.get("eligible"):
                reject(w, cell_name, "any", "gold_ineligible:%s" % (rec or {}).get("reason"))
                continue
            rows_for_world = []
            ok = True
            for point in args.positions:
                si = rec["injection_points"][point]
                prefix = rec["gold_steps"][:si]
                stmt = cmeta["injected_statement"]
                truth = audit_truth_status(w["question"], stmt, w["entity"], prefix)
                if truth["truth_status"] != cmeta["designed_truth"]:
                    reject(w, cell_name, point, "truth_status_mismatch", truth)
                    ok = False
                    break
                dd = cmeta["designed_d"]
                if cmeta["designed_truth"] == "false":
                    md = truth["local_rule_distance"]
                    want = None if dd == "inf" else dd
                    if md != want:
                        reject(w, cell_name, point, "measured_d_mismatch",
                               {"measured": md, "designed": dd})
                        ok = False
                        break
                else:
                    _, direct, _, state, _ = world_state(w["question"], w["entity"], prefix)
                    pred = statement_predicate(stmt, w["entity"])
                    md = 0 if pred in state else shortest_rule_distance(pred, state, direct)
                meta2 = dict(cmeta)
                meta2["group"] = "cat_false" if cmeta["claim_family"] == "category" else "attr_" + cmeta["designed_truth"]
                rows_for_world.append(manifest_row(w, rec, cell_name, meta2, point, si, stmt, truth, md))
            if ok and rows_for_world:
                planned.extend(rows_for_world)
                n_taken += 1

    # ---- anchors: paired on eligible attribute version-A worlds, round-robin over d
    anchors_wanted = [c for c in args.cells if c in ANCHOR_CELLS]
    if anchors_wanted:
        anchor_pool = [w for w in sorted(pool, key=lambda x: (x["family_idx"], str(x["d"])))
                       if w.get("anchors_supported")]
        anchor_pool.sort(key=lambda x: (x["family_idx"], str(x["d"])))
        n_taken = 0
        for w in anchor_pool:
            if n_taken >= args.cap:
                break
            rec = coh.get(w["world_id"])
            if not rec or not rec.get("eligible"):
                continue
            rows_for_world = []
            ok = True
            for point in args.positions:
                si = rec["injection_points"][point]
                prefix = rec["gold_steps"][:si]
                specs = []
                if "benign_paraphrase" in anchors_wanted:
                    specs.append(("benign_paraphrase",
                                  make_paraphrase(rec["gold_steps"], si, idx=w["family_idx"])))
                if "true_interruption" in anchors_wanted:
                    specs.append(("true_interruption",
                                  "%s is a %s." % (w["entity"], w["t0"])))
                for cell_name, stmt in specs:
                    truth = audit_truth_status(w["question"], stmt, w["entity"], prefix)
                    if truth["truth_status"] != "true":
                        reject(w, cell_name, point, "anchor_not_audited_true", truth)
                        ok = False
                        continue
                    if cell_name == "true_interruption" and w["t0"] in " ".join(rec["gold_steps"]):
                        reject(w, cell_name, point, "t0_on_gold_path")
                        ok = False
                        continue
                    meta2 = {"group": "anchor", "designed_d": "na", "polarity": "na",
                             "claim_family": "anchor", "designed_truth": "true"}
                    rows_for_world.append(manifest_row(w, rec, cell_name, meta2, point, si, stmt, truth, None))
            if ok and rows_for_world:
                planned.extend(rows_for_world)
                n_taken += 1
    return planned


def generate(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    if not os.path.exists(paths["candidate_pool"]):
        raise SystemExit("run prepare first")
    pool = read_jsonl(paths["candidate_pool"])
    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise SystemExit("could not pin model revision: %s" % err)
    write_metadata(args.out_dir, args, revision, pinned, err)
    backend = make_backend(args, revision)
    existing_raw = {r["run_id"] for r in read_jsonl(paths["raw"])}

    # phase 1: gold rollouts for every world version
    gold_jobs = []
    for w in pool:
        base = {"run_id": gold_run_id(w["world_id"], args.seed), "world_id": w["world_id"],
                "problem_id": w["world_id"], "family_id": w["family_id"], "condition": "gold",
                "model": args.model, "model_revision": revision,
                "injection_position": "original", "seed": args.seed}
        gold_jobs.append((base, w["question"], w["target"], None))
    n_new_gold = batched_generate(backend, gold_jobs, paths, args, existing_raw)
    print("[EXPD] gold phase: %d new rollouts" % n_new_gold, flush=True)

    # phase 2: eligibility ladder (recomputed fresh each run; overwrites cohort)
    raw_rows = read_jsonl(paths["raw"])
    cohort = build_gold_cohort(pool, raw_rows, args.seed)
    with open(paths["gold_cohort"], "w") as fh:
        for rec in cohort:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    n_eligible = sum(1 for r in cohort if r.get("eligible"))
    print("[EXPD] eligibility: %d/%d world versions eligible" % (n_eligible, len(cohort)), flush=True)

    # phase 3: fail-closed injection manifest (existing manifest rows kept)
    existing_manifest = {r["run_id"] for r in read_jsonl(paths["manifest"])}
    planned = plan_injections(pool, cohort, args, paths, backend, revision)
    for row in planned:
        if row["run_id"] not in existing_manifest:
            append_jsonl(paths["manifest"], row)
            existing_manifest.add(row["run_id"])
    print("[EXPD] manifest: %d rows planned" % len(planned), flush=True)

    # phase 4: perturbed continuations
    manifest_rows = read_jsonl(paths["manifest"])
    pert_jobs = []
    for m in manifest_rows:
        base = {k: m[k] for k in ("run_id", "problem_id", "world_id", "family_id", "condition",
                                  "model", "model_revision", "injection_position", "seed")}
        prefix = " " + " ".join(m["prefix_steps"] + [m["injected_statement"]])
        pert_jobs.append((base, m["question"], m["target"], prefix))
    n_new = batched_generate(backend, pert_jobs, paths, args, existing_raw)
    print("DONE generate gold_new=%d perturbed_new=%d manifest=%d"
          % (n_new_gold, n_new, len(manifest_rows)), flush=True)


# ------------------------------------------------------------------ validate

def strict_validate_against_gold_suffix(continuation, gold_steps, sent_idx, target):
    sents = [strip_marker(s) for s in split_sentences(continuation or "")]
    suffix = gold_steps[sent_idx:]
    final_ok = bool(sents) and norm(sents[-1]) == norm(target)
    if not sents:
        cls = "strict_empty"
    elif not final_ok:
        cls = "strict_final_mismatch"
    else:
        n = min(len(sents), len(suffix))
        replay = all(norm(sents[i]) == norm(suffix[i]) for i in range(n))
        cls = "strict_gold_suffix_replay" if replay else "strict_noncanonical_recovery"
    return {"strict_class": cls, "strict_final_ok": final_ok}


def echo_flags(continuation, injected, entity):
    """M4-style stated-reliance: does any continuation sentence re-assert the
    planted predicate?"""
    pred = statement_predicate(injected, entity)
    if pred is None:
        return {"inj_echo": False}
    for s in split_sentences(continuation or ""):
        f = parse_fact(strip_marker(s), entity)
        if f and f[1] == pred:
            return {"inj_echo": True}
    return {"inj_echo": False}


def validate(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    manifest = {r["run_id"]: r for r in read_jsonl(paths["manifest"])}
    raw = [r for r in read_jsonl(paths["raw"]) if r.get("condition") not in ("gold",)]
    keep = ("run_id", "problem_id", "world_id", "family_id", "family_idx", "partner_world_id",
            "condition", "cell_group", "designed_d", "world_d", "world_version", "world_kind",
            "polarity", "claim_family", "model", "model_revision", "injection_position", "seed",
            "sent_idx", "injected_statement", "audited_truth_status", "measured_d_from_prefix",
            "grammar_template", "predicate_family", "injected_token_count", "injected_word_count",
            "poisoning_measurable", "original_proof_validated", "prompt_sha256")
    n = 0
    with open(paths["validated"], "w") as out:
        for r in raw:
            m = manifest.get(r["run_id"])
            if not m:
                continue
            base = {k: m.get(k) for k in keep}
            if r.get("failed_generation"):
                rec = dict(base)
                rec.update({"class": "generation_failed", "failed_generation": True,
                            "error": r.get("error"), "valid_recovery": False, "doubt": False,
                            "inj_derivational": False, "inj_echo": False})
            else:
                cont = r.get("continuation", "")
                v = validate_continuation(m["question"], m["prefix_steps"], m["injected_statement"],
                                          cont, m["target"], m["entity"])
                strict = strict_validate_against_gold_suffix(cont, m["gold_steps"], m["sent_idx"],
                                                             m["target"])
                rec = dict(base)
                rec.update(v)
                rec.update({
                    "class": v["class"],
                    "closure_validation": v,
                    "strict_validation": strict,
                    "failed_generation": False,
                    "valid_recovery": v["class"] == "valid_rederivation",
                    "doubt": bool(v.get("acknowledged")),
                    "verbalized_doubt": bool(v.get("acknowledged")),
                    "inj_derivational": v["class"] == "poisoned",
                    "continuation": cont,
                })
                rec.update(echo_flags(cont, m["injected_statement"], m["entity"]))
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            n += 1
    print("wrote %s (%d rows)" % (paths["validated"], n))


# ----------------------------------------------------------------- summarize

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cluster_boot_rate(rows, metric, seed=0, n_boot=1000, cluster_key="family_id"):
    ids = sorted({r[cluster_key] for r in rows})
    if not ids:
        return [None, None]
    by_id = defaultdict(list)
    for r in rows:
        by_id[r[cluster_key]].append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        num = den = 0
        for cid in sample:
            for r in by_id[cid]:
                v = metric(r)
                if v is None:
                    continue
                den += 1
                num += v
        vals.append(num / den if den else float("nan"))
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return [None, None]
    return [round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4)]


METRICS = ("valid_recovery", "doubt", "poisoned", "inj_derivational", "inj_echo",
           "parroted", "derailed", "unparsed", "generation_failed")


def metric_value(r, metric):
    if metric == "valid_recovery":
        return int(bool(r.get("valid_recovery")))
    if metric == "doubt":
        return int(bool(r.get("doubt")))
    if metric == "poisoned":
        return int(r.get("class") == "poisoned")
    if metric == "inj_derivational":
        return int(bool(r.get("inj_derivational")))
    if metric == "inj_echo":
        return int(bool(r.get("inj_echo")))
    if metric in ("parroted", "derailed", "unparsed", "generation_failed"):
        return int(r.get("class") == metric)
    raise KeyError(metric)


def summarize_cell(rows, seed):
    out = {"n": len(rows), "family_n": len({r["family_id"] for r in rows})}
    for metric in METRICS:
        vals = [metric_value(r, metric) for r in rows]
        n, k = len(vals), sum(vals)
        out[metric] = {
            "n": n, "count": k, "rate": round(k / n, 4) if n else None,
            "wilson95": wilson(k, n),
            "family_cluster_bootstrap95": cluster_boot_rate(
                rows, lambda r, m=metric: metric_value(r, m), seed=seed) if n else [None, None],
        }
    par = out["unparsed"]["rate"]
    gf = out["generation_failed"]["rate"]
    out["parse_rate"] = round(1.0 - (par or 0) - (gf or 0), 4) if out["n"] else None
    return out


def tost_vs_locked(p_new, n_new, p_ref, n_ref, margin=0.10):
    if not n_new or not n_ref:
        return {"status": "no_data"}
    se = math.sqrt(p_new * (1 - p_new) / n_new + p_ref * (1 - p_ref) / n_ref)
    diff = p_new - p_ref
    z = 1.6449
    lo, hi = diff - z * se, diff + z * se
    return {"diff": round(diff, 4), "ci90": [round(lo, 4), round(hi, 4)],
            "margin": margin, "tost_pass": bool(lo > -margin and hi < margin)}


def summarize(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    rows = read_jsonl(paths["validated"])
    cohort = read_jsonl(paths["gold_cohort"])
    audit_rows = read_jsonl(paths["audit"])
    world_audit = json.load(open(paths["world_audit"])) if os.path.exists(paths["world_audit"]) else {}

    cells = sorted({r["condition"] for r in rows})
    by_cell = {}
    for cell in cells:
        by_cell[cell] = summarize_cell([r for r in rows if r["condition"] == cell], args.seed)

    # ---- gates -------------------------------------------------------------
    attempted = [c for c in cohort if c.get("gold_generated")]
    solved = [c for c in attempted if c.get("solved")]
    valid = [c for c in attempted if c.get("gold_class") == "valid_rederivation"]
    eligible = [c for c in attempted if c.get("eligible")]
    solve_rate = round(len(solved) / len(attempted), 4) if attempted else None
    gates = {}
    gates["gold_greedy_solve_rate"] = {
        "value": solve_rate, "n": len(attempted), "target": 0.60,
        "pass": bool(solve_rate is not None and solve_rate >= 0.60),
        "wilson95": wilson(len(solved), len(attempted)),
    }
    gates["gold_double_filter"] = {
        "solved_rate": solve_rate,
        "valid_rederivation_rate": round(len(valid) / len(attempted), 4) if attempted else None,
        "eligible_rate": round(len(eligible) / len(attempted), 4) if attempted else None,
        "by_kind": {},
    }
    for kind in ("attr", "cat"):
        sub = [c for c in attempted if c["kind"] == kind]
        if sub:
            gates["gold_double_filter"]["by_kind"][kind] = {
                "n": len(sub),
                "solved": round(sum(1 for c in sub if c.get("solved")) / len(sub), 4),
                "eligible": round(sum(1 for c in sub if c.get("eligible")) / len(sub), 4),
            }

    gates["parse_rate_per_cell"] = {}
    for cell in cells:
        c = by_cell[cell]
        gates["parse_rate_per_cell"][cell] = {
            "parse_rate": c["parse_rate"], "unparsed_rate": c["unparsed"]["rate"],
            "unparsed_count": c["unparsed"]["count"], "n": c["n"],
            "pass_090": bool(c["parse_rate"] is not None and c["parse_rate"] >= 0.90),
        }

    anchors = {}
    for cell in ANCHOR_CELLS:
        if cell not in by_cell:
            continue
        c = by_cell[cell]
        ref = EXPA_LOCKED_ANCHORS[cell]
        anchors[cell] = {
            "pilot": {"n": c["n"], "valid_recovery": c["valid_recovery"]["rate"],
                      "doubt": c["doubt"]["rate"], "unparsed": c["unparsed"]["rate"]},
            "expa_locked": ref,
            "valid_recovery_tost": tost_vs_locked(
                c["valid_recovery"]["rate"] or 0, c["n"], ref["valid_rederivation"], ref["n"]),
            "unparsed_tost": tost_vs_locked(
                c["unparsed"]["rate"] or 0, c["n"], ref["unparsed"], ref["n"]),
            "doubt_comparison": {
                "diff_vs_locked_hf": (round(c["doubt"]["rate"] - ref["verbalized_doubt"], 4)
                                      if c["doubt"]["rate"] is not None else None),
                "verdict": "backend_confounded_no_verdict",
            },
        }
    gates["anchor_equivalence_mc2"] = {
        "cells": anchors,
        "scope": "MOD-12: gates cross-generator continuity claims only, never H1-H4",
        "doubt_caveat": VLLM_DOUBT_CAVEAT,
        "expa_provenance": EXPA_PROVENANCE,
    }

    # joint-solve of mirrored pairs + MC-7 prefix identity
    coh = {c["world_id"]: c for c in cohort}
    joint = {}
    prefix_stats = {"pairs_checked": 0, "identical_si": 0, "identical_prefix": 0, "details_by_d": {}}
    fams = defaultdict(dict)
    for c in cohort:
        if c["kind"] == "attr":
            fams[(c["family_id"], str(c["d"]))][c["version"]] = c
    for (fid, d), vv in sorted(fams.items()):
        a, b = vv.get("A"), vv.get("B")
        if not a or not b:
            continue
        dk = "d%s" % d
        j = joint.setdefault(dk, {"pairs": 0, "joint_eligible": 0})
        j["pairs"] += 1
        if a.get("eligible") and b.get("eligible"):
            j["joint_eligible"] += 1
            pt = args.positions[0]
            sia, sib = a["injection_points"][pt], b["injection_points"][pt]
            same_si = sia == sib
            same_prefix = same_si and all(
                norm(x) == norm(y) for x, y in zip(a["gold_steps"][:sia], b["gold_steps"][:sib]))
            prefix_stats["pairs_checked"] += 1
            prefix_stats["identical_si"] += int(same_si)
            prefix_stats["identical_prefix"] += int(same_prefix)
            det = prefix_stats["details_by_d"].setdefault(dk, {"n": 0, "identical_prefix": 0})
            det["n"] += 1
            det["identical_prefix"] += int(same_prefix)
    for dk, j in joint.items():
        j["joint_solve_rate"] = round(j["joint_eligible"] / j["pairs"], 4) if j["pairs"] else None
    if prefix_stats["pairs_checked"]:
        prefix_stats["identical_prefix_rate"] = round(
            prefix_stats["identical_prefix"] / prefix_stats["pairs_checked"], 4)
    gates["joint_solve_mirrored_quadruples"] = joint
    gates["gold_prefix_identity_mc7"] = prefix_stats

    grad = {}
    for cell in ("aff_false_attr_d1", "aff_false_attr_d3", "aff_false_attr_d5"):
        if cell in by_cell:
            grad[cell] = {"doubt": by_cell[cell]["doubt"]["rate"],
                          "doubt_ci": by_cell[cell]["doubt"]["family_cluster_bootstrap95"],
                          "valid": by_cell[cell]["valid_recovery"]["rate"], "n": by_cell[cell]["n"]}
    gates["aff_false_doubt_gradient_descriptive"] = grad

    reuse = {}
    for cell in ("cat_false_usable_d1", "cat_false_inert_d1"):
        if cell in by_cell:
            reuse[cell] = {
                "derivational_inj_dep": by_cell[cell]["inj_derivational"]["rate"],
                "derivational_ci": by_cell[cell]["inj_derivational"]["family_cluster_bootstrap95"],
                "echo": by_cell[cell]["inj_echo"]["rate"],
                "doubt": by_cell[cell]["doubt"]["rate"],
                "valid": by_cell[cell]["valid_recovery"]["rate"],
                "n": by_cell[cell]["n"],
            }
    gates["reuse_liveness_major2"] = {
        "cells": reuse,
        "note": "MAJOR-2 pilot check: derivational injection-dependence must be live (nonzero) "
                "in the usable cell and structurally ~0 in the inert cell; H4 MDE input.",
    }

    funnel = {
        "worlds_generated": world_audit.get("world_count"),
        "world_audit_failures": len(world_audit.get("audit_failures", [])),
        "gold_attempted": len(attempted),
        "gold_solved": len(solved),
        "gold_valid_rederivation": len(valid),
        "gold_eligible": len(eligible),
        "injection_audit_rejections": dict(Counter(
            r.get("reason") for r in audit_rows if r.get("stage") == "injection_audit")),
        "manifest_rows": len(read_jsonl(paths["manifest"])),
        "validated_rows": len(rows),
    }

    dup_ids = [rid for rid, cnt in Counter(r["run_id"] for r in rows).items() if cnt > 1]
    integrity = {
        "unique_run_ids": not dup_ids,
        "duplicate_run_id_count": len(dup_ids),
        "all_planted_false_cells_audited_false": all(
            r.get("audited_truth_status") == "false" for r in rows
            if r.get("cell_group") in ("attr_false", "cat_false")),
        "all_anchor_cells_audited_true": all(
            r.get("audited_truth_status") == "true" for r in rows if r.get("cell_group") == "anchor"),
        "all_measured_d_match": all(
            (r.get("measured_d_from_prefix") == (None if r.get("designed_d") == "inf" else r.get("designed_d")))
            for r in rows if r.get("cell_group") in ("attr_false", "cat_false")),
        "prompt_hashes": sorted({r.get("prompt_sha256") for r in rows}),
        "no_attr_inf_cells": not any("attr" in (r.get("condition") or "") and "inf" in (r.get("condition") or "")
                                     for r in rows),
    }

    meta = json.load(open(paths["metadata"])) if os.path.exists(paths["metadata"]) else {}
    out = {
        "created_at": now_iso(),
        "experiment": meta.get("experiment", "EXPD_MATCHED_GRADIENT_PILOT"),
        "metrics_by_cell": by_cell,
        "gates_plan2_section6": gates,
        "funnel_consort": funnel,
        "integrity": integrity,
        "grammar_theorems": world_audit.get("grammar_theorems"),
        "design_notes": world_audit.get("design_notes"),
        "denominator_convention": "unparsed and generation_failed retained in all denominators "
                                  "(house convention); parse_rate reported per cell (MC-1).",
    }
    write_json(paths["summary"], out)
    print(json.dumps({"cells": {c: by_cell[c]["n"] for c in cells}, "gates_keys": sorted(gates)},
                     indent=2))


# -------------------------------------------------------------------- report

def _fmt_rate(m):
    return "%s %s" % (m["rate"], m["family_cluster_bootstrap95"])


def report(args):
    refuse_complete(args.out_dir, args.overwrite)
    paths = out_paths(args.out_dir)
    s = json.load(open(paths["summary"]))
    meta = json.load(open(paths["metadata"])) if os.path.exists(paths["metadata"]) else {}
    rows = read_jsonl(paths["validated"])
    gates = s["gates_plan2_section6"]
    by_cell = s["metrics_by_cell"]

    full = meta.get("experiment") == "EXPD_MATCHED_GRADIENT"
    ev = meta.get("pre_registration_evidence", {})
    if full:
        status_lines = [
            "**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5), generated under the",
            "externally timestamped PLAN2.** Timestamp-gate evidence: git commit `%s`" % ev.get("plan2_git_commit"),
            "(%s), PLAN2.md sha256 `%s`." % (ev.get("plan2_commit_timestamp"), ev.get("plan2_file_sha256")),
            "This file reports the runner's mechanical stages only; the registered adjudications",
            "(H2', H3', H4, C-0) live in EXPD_FULL_REPORT.md + hypothesis_verdicts.json.",
        ]
    else:
        status_lines = [
            "**Status: 200-world pilot, pre-registration-exempt (PLAN2 section 8).** The full",
            "confirmatory run is gated on the externally timestamped PLAN2 (section 9.10) and was NOT run.",
        ]
    lines = [
        "# EXPD Matched-Gradient %s Report" % ("FULL RUN (runner stages)" if full else "PILOT"),
        "",
        "Generated: %s" % now_iso(),
        "Backend: %s | model: %s (%s) | prompt sha256: %s" % (
            meta.get("backend"), meta.get("model"), meta.get("model_revision"),
            meta.get("prompt", {}).get("sha256")),
        "",
    ] + status_lines + [
        "",
        "## Gate table (PLAN2 section 6)",
        "",
    ]
    g = gates["gold_greedy_solve_rate"]
    lines.append("| gate | value | target | pass |")
    lines.append("|---|---|---|---|")
    lines.append("| gold greedy solve rate | %s (n=%s, wilson %s) | >=0.60 | %s |" % (
        g["value"], g["n"], g["wilson95"], g["pass"]))
    df = gates["gold_double_filter"]
    lines.append("| gold double filter (solved AND valid AND >=3 sites) | %s | - | - |" % df["eligible_rate"])
    for cell, p in sorted(gates["parse_rate_per_cell"].items()):
        lines.append("| parse rate: %s | %s (unparsed n=%s/%s) | >=0.90 | %s |" % (
            cell, p["parse_rate"], p["unparsed_count"], p["n"], p["pass_090"]))
    for cell, a in gates["anchor_equivalence_mc2"]["cells"].items():
        vt = a["valid_recovery_tost"]
        lines.append("| anchor valid TOST(+-0.10): %s | diff=%s ci90=%s | in margin | %s |" % (
            cell, vt.get("diff"), vt.get("ci90"), vt.get("tost_pass")))
        lines.append("| anchor doubt: %s | pilot=%s locked-HF=%s diff=%s | n/a | backend-confounded, no verdict |" % (
            cell, a["pilot"]["doubt"], a["expa_locked"]["verbalized_doubt"],
            a["doubt_comparison"]["diff_vs_locked_hf"]))
    for dk, j in sorted(gates["joint_solve_mirrored_quadruples"].items()):
        lines.append("| joint-solve mirrored pairs %s | %s (%s/%s) | - | - |" % (
            dk, j["joint_solve_rate"], j["joint_eligible"], j["pairs"]))
    pf = gates["gold_prefix_identity_mc7"]
    lines.append("| MC-7 identical gold prefix at injection (jointly-solved pairs) | %s (%s/%s) | - | - |" % (
        pf.get("identical_prefix_rate"), pf.get("identical_prefix"), pf.get("pairs_checked")))
    lines += ["", "## Headline per-cell rates (family-cluster bootstrap 95% CIs)", ""]
    lines.append("| cell | n | valid | doubt | derivational inj-dep | echo | parroted | derailed | unparsed |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cell in sorted(by_cell):
        c = by_cell[cell]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            cell, c["n"], _fmt_rate(c["valid_recovery"]), _fmt_rate(c["doubt"]),
            _fmt_rate(c["inj_derivational"]), c["inj_echo"]["rate"],
            c["parroted"]["rate"], c["derailed"]["rate"], c["unparsed"]["rate"]))
    grad = gates["aff_false_doubt_gradient_descriptive"]
    lines += ["", "## Descriptive doubt gradient (aff_false_attr, negation-free planted sentence)", ""]
    for cell, v in sorted(grad.items()):
        lines.append("- %s: doubt=%s %s (n=%s, valid=%s)" % (cell, v["doubt"], v["doubt_ci"], v["n"], v["valid"]))
    ru = gates["reuse_liveness_major2"]["cells"]
    lines += ["", "## MAJOR-2 reuse liveness (H4 MDE input)", ""]
    for cell, v in sorted(ru.items()):
        lines.append("- %s: derivational=%s %s echo=%s doubt=%s valid=%s (n=%s)" % (
            cell, v["derivational_inj_dep"], v["derivational_ci"], v["echo"], v["doubt"], v["valid"], v["n"]))
    lines += ["", "## Funnel (CONSORT, MC-4)", "", "```",
              json.dumps(s["funnel_consort"], indent=2), "```",
              "", "## Integrity", "", "```", json.dumps(s["integrity"], indent=2), "```",
              "", "## Disclosures", "",
              "- " + gates["anchor_equivalence_mc2"]["doubt_caveat"],
              "- Planted-token frequency: ==1 in attribute and inert-categorical cells; the usable "
              "categorical cell is structurally 2 (refuting head + MAJOR-2 usable-branch body), held "
              "constant across d. See worlds/world_audit.json design_notes.",
              "- cat d=inf worlds (not in pilot cells) carry the refuting-shaped head on an "
              "entity-disconnected body: complement underivable from the entity state (canonical "
              "PLAN2 section-3 d=inf), lexically matched across d.",
              "", "## Sample continuations", ""]
    rng = random.Random(args.seed)
    for cell in sorted(by_cell):
        sub = [r for r in rows if r["condition"] == cell and not r.get("failed_generation")]
        rng.shuffle(sub)
        for r in sub[:2]:
            lines.append("- `%s` [%s] inj=`%s` -> %s" % (
                cell, r["class"], r["injected_statement"],
                " ".join((r.get("continuation") or "").split())[:220]))
    text = "\n".join(lines) + "\n"
    report_path = (os.path.join(args.out_dir, "EXPD_RUNNER_STAGES_REPORT.md")
                   if full else paths["report"])
    with open(report_path, "w") as fh:
        fh.write(text)
    with open(paths["readme"], "w") as fh:
        fh.write(text)
    print(text)


def test(args):
    import expd_tests
    expd_tests.main()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_paths(args.out_dir)["tests_passed"], "w") as fh:
        fh.write(now_iso() + "\n")
    print("EXPD tests passed; marker written")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["test", "prepare", "generate", "validate", "summarize", "report"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default="auto")
    p.add_argument("--allow-unpinned-model", action="store_true")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--world-config", choices=["pilot", "full"], default="pilot")
    p.add_argument("--confirm-timestamped-plan2", action="store_true",
                   help="required for the full confirmatory grid (PLAN2 section 9.10)")
    p.add_argument("--plan2-commit", default=None,
                   help="git commit hash of the timestamped PLAN2 (recorded in run_metadata)")
    p.add_argument("--plan2-commit-timestamp", default=None,
                   help="author timestamp of the PLAN2 commit (recorded in run_metadata)")
    p.add_argument("--plan2-sha256", default=None,
                   help="sha256 of the timestamped PLAN2.md (recorded in run_metadata)")
    p.add_argument("--cells", default=",".join(PILOT_CELLS))
    p.add_argument("--positions", default="mid")
    p.add_argument("--cap", type=int, default=35, help="max worlds per cell (pilot n<=35)")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args(argv)
    args.cells = tuple(x for x in args.cells.split(",") if x)
    args.positions = tuple(x for x in args.positions.split(",") if x)
    for pt in args.positions:
        if pt not in POINTS:
            raise SystemExit("bad position %r" % pt)
    return args


def main(argv=None):
    args = parse_args(argv)
    stage = {"test": test, "prepare": prepare, "generate": generate,
             "validate": validate, "summarize": summarize, "report": report}[args.stage]
    stage(args)


if __name__ == "__main__":
    main()
