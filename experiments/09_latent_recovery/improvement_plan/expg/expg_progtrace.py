"""EXPG_PROGTRACE staged runner -- Stage-0 grammar gate + Stage-1 PILOT.

Usage (EXPC/EXPD pattern; one stage per invocation):

  python expg_progtrace.py test      --out-dir <dir>
  python expg_progtrace.py stage0    --out-dir <dir> --stage0-n 100
  python expg_progtrace.py prepare   --out-dir <dir>
  python expg_progtrace.py generate  --out-dir <dir> --backend vllm
  python expg_progtrace.py validate  --out-dir <dir>
  python expg_progtrace.py judge     --out-dir <dir>          (32B, optional)
  python expg_progtrace.py summarize --out-dir <dir>
  python expg_progtrace.py report    --out-dir <dir>

Scope guard: this runner executes the PILOT ONLY (mid position, n<=35/cell,
the 7 pilot cells). The full grid is REFUSED without
--confirm-timestamped-plan-progtrace (PLAN2 section 9.10: PLAN_PROGTRACE.md
must be committed and hash-published before EXPG Stage 2).

Stage gates (regime2 section 4 staged path; PLAN2 section 10):
  Stage-0: greedy solve >= 0.70 and program-level parse >= 0.95, else retune
           grammar knobs and rerun stage0.
  G-A:     pilot gold parse >= 0.90 and solve >= 0.60, else loop Stage 0.
  G-B:     adjacent_contradiction or opfree_kr1 shows doubt >= 0.10 OR
           next-read absorption <= 0.85  ->  full grid viable; else the
           floor-documentation branch (pre-registered, scope-bounding).

Protocol invariants: frozen Qwen2.5-7B-Instruct at the PINNED revision,
greedy, gold max_new=512 / continuation max_new=384, no feedback; injection =
replacement of the stated result value on the model's OWN trace line
(string-prefix mechanics; token ids under vLLM). New suites run wholly under
ONE backend (vLLM port fidelity report: never mix backends within an
experiment; doubt absolute rates are backend-sensitive -> contrasts only).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import traceback
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_programs as gp                      # noqa: E402
import inject as inj_mod                       # noqa: E402
import trace_format as tf                      # noqa: E402
import validate as val_mod                     # noqa: E402
from interp import parse_program               # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "results", "EXPG_PROGTRACE_PILOT")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PINNED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_SEED = 0

PILOT_CELLS = ("benign_paraphrase", "true_interruption", "adjacent_contradiction",
               "opfree_kr1", "opfree_kr8", "onehop_kc1", "deep_kc5")
CELL_SHAPE = {"benign_paraphrase": "kr1", "true_interruption": "kr1",
              "adjacent_contradiction": "kr1", "opfree_kr1": "kr1",
              "opfree_kr8": "kr8", "onehop_kc1": "kc1", "deep_kc5": "kc5"}
SHAPE_SEED_BASE = {"kr1": 100000, "kr8": 200000, "kc1": 300000, "kc5": 400000}
STAGE0_SEED_BASE = {"kr1": 910000, "kr8": 920000, "kc1": 930000, "kc5": 940000}

LOCKED_PATTERNS = ["/src/", "/data/", "paper_latex", "/results/arith",
                   "/results/gsm8k", "/results/aligned_gold", "/prontoqa"]


# ------------------------------------------------------------------ plumbing

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha_row(parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def out_paths(out_dir):
    return {
        "programs": os.path.join(out_dir, "programs.jsonl"),
        "manifest": os.path.join(out_dir, "manifest.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "cohort": os.path.join(out_dir, "cohort.json"),
        "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
        "judge": os.path.join(out_dir, "judge_verdicts.jsonl"),
        "summary": os.path.join(out_dir, "summary_tables.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "audit": os.path.join(out_dir, "generation_audit.jsonl"),
        "report": os.path.join(out_dir, "REPORT.md"),
        "stage0_dir": os.path.join(out_dir, "stage0"),
        "complete": os.path.join(out_dir, "COMPLETE"),
    }


def guard_out_dir(out_dir):
    ab = os.path.abspath(out_dir)
    for pat in LOCKED_PATTERNS:
        if pat in ab:
            raise SystemExit("refusing to write into protected path: %s" % ab)
    if "EXPG_PROGTRACE" not in os.path.basename(ab):
        raise SystemExit("out dir %s must be results/EXPG_PROGTRACE*" % ab)


def guard_scope(args):
    cells = [c for c in args.cells.split(",") if c]
    if args.confirm_timestamped_plan_progtrace:
        return cells
    bad = [c for c in cells if c not in PILOT_CELLS]
    if bad or args.cap > 35:
        raise SystemExit(
            "REFUSED: full grid (cells beyond the pilot set, or n/cell > 35) "
            "requires the externally timestamped PLAN_PROGTRACE.md "
            "(PLAN2 section 9.10). Offending: cells=%s cap=%d" % (bad, args.cap))
    return cells


def knob_overrides(args):
    knobs = dict(gp.KNOBS)
    for kv in (args.knob or []):
        k, v = kv.split("=", 1)
        if k not in knobs:
            raise SystemExit("unknown knob %r" % k)
        knobs[k] = type(knobs[k])(float(v)) if isinstance(knobs[k], (int, float)) \
            else v
    return knobs


def write_metadata(out_dir, args, knobs, extra=None):
    meta = {
        "experiment": "EXPG_PROGTRACE_PILOT",
        "status_note": (
            "Stage-0 + Stage-1 PILOT, pre-registration-exempt (PLAN2 section 8: "
            "the pilot gates go/no-go -- G-A, G-B -- and tunes grammar knobs; it "
            "scores no confirmatory hypothesis; H7 runs only in the full grid, "
            "which this runner refuses without the timestamped PLAN_PROGTRACE.md)."),
        "created_at": now_iso(),
        "model": args.model,
        "model_revision": args.revision,
        "model_revision_pinned": args.revision == PINNED_REVISION,
        "judge_model": JUDGE_MODEL,
        "backend": args.backend,
        "backend_note": (
            "New suites run wholly under one backend; never mix backends within "
            "an experiment. Doubt absolute rates are backend-sensitive (vLLM "
            "8-15pp below locked HF levels; tooling/replay_full) -> doubt is "
            "reported as within-regime contrasts, never absolute one-seed "
            "quantities."),
        "seed": args.seed,
        "cells": list(PILOT_CELLS),
        "cell_shape": CELL_SHAPE,
        "positions": ["mid"],
        "per_cell_cap": args.cap,
        "oversample_per_shape": args.oversample,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens_gold": args.max_new_gold,
                     "max_new_tokens_continuation": args.max_new_cont},
        "prompt": {"instruction_plus_exemplars_sha256": tf.prompt_sha256(),
                   "gold_prefill": tf.GOLD_PREFILL,
                   "shared_across_conditions": True},
        "grammar_knobs": knobs,
        "doubt_lexicon_frozen": val_mod.DOUBT_LEX_RE.pattern,
        "doubt_lexicon_broad_exploratory": val_mod.DOUBT_BROAD_RE.pattern,
        "design_references": {
            "design": "reports/regime2.md sections 2, 4",
            "amendments": "reports/verdict.md MAJOR-5, MOD-9, MOD-10 "
                          "(binding, generation-time; PLAN2 section 4)",
            "pre_registration": "PLAN2.md (H7 slot; sections 3, 4, 6, 8, 10)",
            "stats": "wilson + cluster_boot_rate, program = cluster "
                     "(expa_global_expansion.py:738-767 convention)",
        },
        "interpreter_note": "hand-rolled evaluator (interp.py); no exec() on "
                            "generated programs anywhere; no CRUXEval rows in "
                            "the pilot (optional arm skipped)",
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version,
    }
    meta.update(extra or {})
    write_json(out_paths(out_dir)["metadata"], meta)
    return meta


# ------------------------------------------------------------------ backends

def find_tooling():
    cands = [os.environ.get("EXPG_TOOLING"),
             os.path.abspath(os.path.join(HERE, "..", "tooling"))]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "vllm_gen.py")):
            return c
    raise SystemExit("cannot locate tooling/vllm_gen.py; set EXPG_TOOLING")


class VllmBackend:
    name = "vllm"

    def __init__(self, args):
        tooling = find_tooling()
        if tooling not in sys.path:
            sys.path.insert(0, tooling)
        import vllm_gen
        self.vg = vllm_gen
        self.model = args.model
        self.revision = args.revision
        self.tok = vllm_gen.get_tokenizer(args.model, args.revision)

    def render(self, user_msg, prefill):
        ids = self.vg._chat_ids(self.tok, [{"role": "user", "content": user_msg}])
        if prefill:
            ids = ids + list(self.tok(prefill, add_special_tokens=False)["input_ids"])
        return {"prompt_token_ids": ids}

    def generate(self, rows, max_new_tokens):
        return self.vg.generate_continuations(self.model, rows,
                                              max_new_tokens=max_new_tokens,
                                              revision=self.revision)


class HfBackend:
    name = "hf"

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(args.seed)
        self.tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            device_map={"": args.device})
        self.model.eval()

    def render(self, user_msg, prefill):
        out = self.tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                           add_generation_prompt=True,
                                           return_tensors="pt")
        ids = out["input_ids"] if (hasattr(out, "data") or isinstance(out, dict)) else out
        if prefill:
            pre = self.tok(prefill, return_tensors="pt",
                           add_special_tokens=False)["input_ids"]
            ids = self.torch.cat([ids, pre], dim=1)
        return {"ids": ids}

    def generate(self, rows, max_new_tokens):
        outs = []
        for r in rows:
            ids = r["ids"].to(next(self.model.parameters()).device)
            with self.torch.no_grad():
                o = self.model.generate(ids, attention_mask=self.torch.ones_like(ids),
                                        max_new_tokens=max_new_tokens,
                                        do_sample=False,
                                        pad_token_id=self.tok.eos_token_id)
            outs.append({"text": self.tok.decode(o[0][ids.shape[1]:],
                                                 skip_special_tokens=True)})
        return outs


def make_backend(args):
    return VllmBackend(args) if args.backend == "vllm" else HfBackend(args)


def batched_generate(backend, jobs, raw_path, existing, max_new, batch_size, tag):
    """jobs: list of (base_row, user_msg, prefill). Resume-safe, chunked,
    per-row fallback on batch failure."""
    pending = [j for j in jobs if j[0]["run_id"] not in existing]
    done = 0
    for i in range(0, len(pending), batch_size):
        chunk = pending[i:i + batch_size]
        rows = [backend.render(u, p) for _, u, p in chunk]
        try:
            outs = backend.generate(rows, max_new)
        except Exception:
            outs = []
            for row in rows:
                try:
                    outs.extend(backend.generate([row], max_new))
                except Exception as e:
                    outs.append({"text": None, "error": repr(e),
                                 "traceback": traceback.format_exc()})
        for (base, _u, _p), out in zip(chunk, outs):
            rec = dict(base)
            if out.get("text") is None:
                rec.update({"failed_generation": True, "error": out.get("error"),
                            "created_at": now_iso()})
            else:
                rec.update({"failed_generation": False, "continuation": out["text"],
                            "finish_reason": out.get("finish_reason"),
                            "created_at": now_iso()})
            append_jsonl(raw_path, rec)
            existing.add(base["run_id"])
        done += len(chunk)
        print("[EXPG:%s] generated %d/%d" % (tag, done, len(pending)), flush=True)
    return len(pending)


# ----------------------------------------------------------- program helpers

def program_row(p):
    r = dict(p)
    r["listing_text"] = tf.make_listing_text(p["stmt_texts"])
    return r


def gold_run_id(program_id, seed):
    return sha_row(["EXPG", program_id, "gold", seed])


def inj_run_id(program_id, cell, seed):
    return sha_row(["EXPG", program_id, cell, "mid", seed])


def gold_full_text(raw_row):
    return tf.GOLD_PREFILL + raw_row["continuation"]


def gold_eval_all(programs, raw_rows):
    """Solve/parse evaluation for every program with a gold generation."""
    gold = {r["program_id"]: r for r in raw_rows
            if r.get("condition") == "gold" and not r.get("failed_generation")}
    per_prog, n_claim_lines, n_unparsed_lines = [], 0, 0
    for p in programs:
        g = gold.get(p["program_id"])
        if g is None:
            per_prog.append({"program_id": p["program_id"], "shape": p["shape"],
                             "gold_generated": False, "solved": False,
                             "parse_ok": False, "eligible": False,
                             "reason": "gold_missing_or_failed"})
            continue
        ev = inj_mod.gold_solve_eval(p, gold_full_text(g))
        n_claim_lines += ev["n_claim_lines"]
        n_unparsed_lines += ev["n_unparsed_lines"]
        per_prog.append({"program_id": p["program_id"], "shape": p["shape"],
                         "gold_generated": True, "solved": ev["solved"],
                         "parse_ok": ev["parse_ok"], "complete": ev["complete"],
                         "correct": ev["correct"], "eligible": ev["solved"],
                         "first_error": ev["first_error"],
                         "post_output_extra": ev["post_output_extra"],
                         "reason": "eligible" if ev["solved"] else
                         (ev["first_error"] or {}).get("type", "unknown")})
    n = len(per_prog)
    summ = {
        "n_programs": n,
        "solve_rate": round(sum(r["solved"] for r in per_prog) / float(n), 4) if n else None,
        "parse_rate_program_level": round(sum(r["parse_ok"] for r in per_prog) / float(n), 4) if n else None,
        "parse_rate_line_level": round(1.0 - n_unparsed_lines / float(n_claim_lines), 4) if n_claim_lines else None,
        "failure_taxonomy": dict(Counter(r["reason"] for r in per_prog
                                         if not r["solved"])),
        "by_shape": {},
    }
    for shape in gp.SHAPES:
        rows = [r for r in per_prog if r["shape"] == shape]
        if rows:
            summ["by_shape"][shape] = {
                "n": len(rows),
                "solve_rate": round(sum(r["solved"] for r in rows) / float(len(rows)), 4),
                "parse_rate": round(sum(r["parse_ok"] for r in rows) / float(len(rows)), 4),
            }
    return per_prog, summ


# ------------------------------------------------------------------- stage 0

def stage0(args):
    guard_out_dir(args.out_dir)
    knobs = knob_overrides(args)
    paths = out_paths(args.out_dir)
    s0 = paths["stage0_dir"]
    os.makedirs(s0, exist_ok=True)
    per_shape = max(1, args.stage0_n // len(gp.SHAPES))
    programs, funnels = [], {}
    for shape in gp.SHAPES:
        progs, funnel = gp.generate_shape_programs(shape, per_shape,
                                                   STAGE0_SEED_BASE[shape], knobs)
        programs.extend(progs)
        funnels[shape] = funnel
    prog_path = os.path.join(s0, "programs.jsonl")
    if not os.path.exists(prog_path):
        for p in programs:
            append_jsonl(prog_path, program_row(p))
    raw_path = os.path.join(s0, "raw_generations.jsonl")
    existing = {r["run_id"] for r in read_jsonl(raw_path)}
    backend = make_backend(args)
    jobs = []
    for p in programs:
        base = {"run_id": gold_run_id(p["program_id"], args.seed),
                "program_id": p["program_id"], "condition": "gold",
                "shape": p["shape"]}
        jobs.append((base, tf.make_user_message(tf.make_listing_text(p["stmt_texts"])),
                     tf.GOLD_PREFILL))
    batched_generate(backend, jobs, raw_path, existing, args.max_new_gold,
                     args.batch_size, "stage0-gold")
    raw_rows = read_jsonl(raw_path)
    # attach run ids
    by_id = {gold_run_id(p["program_id"], args.seed): p for p in programs}
    for r in raw_rows:
        r["condition"] = "gold"
        r["program_id"] = by_id[r["run_id"]]["program_id"] if r["run_id"] in by_id \
            else r.get("program_id")
    per_prog, summ = gold_eval_all(programs, raw_rows)
    gates = {
        "solve_gate_0.70": summ["solve_rate"] is not None and summ["solve_rate"] >= 0.70,
        "parse_gate_0.95": summ["parse_rate_program_level"] is not None
                           and summ["parse_rate_program_level"] >= 0.95,
    }
    report = {"stage": "stage0", "created_at": now_iso(),
              "n_requested": args.stage0_n, "knobs": knobs,
              "generator_funnels": funnels, "gold_eval": summ, "gates": gates,
              "gates_pass": all(gates.values()),
              "model": args.model, "revision": args.revision,
              "backend": args.backend,
              "decoding": {"greedy": True, "max_new_tokens": args.max_new_gold}}
    write_json(os.path.join(s0, "stage0_report.json"), report)
    with open(os.path.join(s0, "per_program_eval.jsonl"), "w") as fh:
        for r in per_prog:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("gold_eval", "gates", "gates_pass", "knobs")}, indent=2))


# ------------------------------------------------------------------- prepare

def prepare(args):
    guard_out_dir(args.out_dir)
    guard_scope(args)
    knobs = knob_overrides(args)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = out_paths(args.out_dir)
    if os.path.exists(paths["programs"]) and not args.overwrite:
        raise SystemExit("programs.jsonl exists; use --overwrite")
    for k in ("programs", "manifest", "raw", "validated", "audit"):
        if os.path.exists(paths[k]) and args.overwrite:
            os.remove(paths[k])
    funnels = {}
    for shape in gp.SHAPES:
        progs, funnel = gp.generate_shape_programs(shape, args.oversample,
                                                   SHAPE_SEED_BASE[shape], knobs)
        funnels[shape] = funnel
        for p in progs:
            append_jsonl(paths["programs"], program_row(p))
    meta = write_metadata(args.out_dir, args, knobs,
                          extra={"generator_funnels": funnels})
    print(json.dumps({"programs": args.oversample * len(gp.SHAPES),
                      "funnels": {s: {"attempts": f["attempts"]}
                                  for s, f in funnels.items()},
                      "metadata_written": True,
                      "experiment": meta["experiment"]}, indent=2))


# ------------------------------------------------------------------ generate

def generate(args):
    guard_out_dir(args.out_dir)
    cells = guard_scope(args)
    paths = out_paths(args.out_dir)
    programs = read_jsonl(paths["programs"])
    if not programs:
        raise SystemExit("run prepare first")
    backend = make_backend(args)
    existing = {r["run_id"] for r in read_jsonl(paths["raw"])}

    # ---- phase 1: gold traces
    jobs = []
    for p in programs:
        base = {"run_id": gold_run_id(p["program_id"], args.seed),
                "program_id": p["program_id"], "condition": "gold",
                "shape": p["shape"]}
        jobs.append((base, tf.make_user_message(p["listing_text"]), tf.GOLD_PREFILL))
    batched_generate(backend, jobs, paths["raw"], existing, args.max_new_gold,
                     args.batch_size, "gold")

    # ---- phase 2: cohort double filter (regime2 2.2)
    raw_rows = read_jsonl(paths["raw"])
    per_prog, summ = gold_eval_all(programs, raw_rows)
    gate_a = {"parse_gate_0.90": (summ["parse_rate_program_level"] or 0) >= 0.90,
              "solve_gate_0.60": (summ["solve_rate"] or 0) >= 0.60}
    cohort = {"per_program": per_prog, "summary": summ, "gate_A": gate_a,
              "gate_A_pass": all(gate_a.values())}
    write_json(paths["cohort"], cohort)
    print("[EXPG] cohort: %s gate_A=%s" % (json.dumps(summ["by_shape"]), gate_a))
    if not cohort["gate_A_pass"] and not args.ignore_gate_a:
        raise SystemExit("GATE G-A FAILED (parse %s, solve %s): loop Stage 0 "
                         "(retune knobs). Use --ignore-gate-a to proceed anyway."
                         % (summ["parse_rate_program_level"], summ["solve_rate"]))

    # ---- phase 3: fail-closed injection planning (manifest)
    gold = {r["program_id"]: r for r in raw_rows
            if r.get("condition") == "gold" and not r.get("failed_generation")}
    eligible = {s: [] for s in gp.SHAPES}
    elig_ids = {r["program_id"] for r in per_prog if r["eligible"]}
    for p in programs:
        if p["program_id"] in elig_ids:
            eligible[p["shape"]].append(p)
    manifest_existing = {r["run_id"] for r in read_jsonl(paths["manifest"])}
    n_rejected = 0
    for cell in cells:
        shape = CELL_SHAPE[cell]
        picked = eligible[shape][:args.cap]
        for p in picked:
            rid = inj_run_id(p["program_id"], cell, args.seed)
            if rid in manifest_existing:
                continue
            try:
                b = inj_mod.build_injection(p, cell, gold_full_text(gold[p["program_id"]]))
            except inj_mod.InjectError as e:
                append_jsonl(paths["audit"], {
                    "stage": "injection_audit", "run_id": rid, "cell": cell,
                    "program_id": p["program_id"], "rejected": True,
                    "reason": e.reason, "created_at": now_iso()})
                n_rejected += 1
                continue
            row = {"run_id": rid, "cell": cell, "condition": cell,
                   "program_id": p["program_id"], "shape": shape,
                   "position": "mid", "position_frac": p["position_frac"],
                   "L": p["L"], "site_line": p["site_line"],
                   "site_var": p["site_var"], "site_kind": p["site_kind"],
                   "k_r": p["k_r"], "k_c": p["k_c"],
                   "r1": p["r1"], "r1_var": p["r1_var"],
                   "r2": p["r2"], "r2_var": p["r2_var"],
                   "seed": args.seed, "created_at": now_iso()}
            row.update({k: b[k] for k in
                        ("family", "injected_trace_line", "original_trace_line",
                         "prefix_text", "step_replaced", "expected_lines_from",
                         "cf", "planted_var", "planted_value", "true_value",
                         "force_after_line", "delta_policy")})
            append_jsonl(paths["manifest"], row)
            manifest_existing.add(rid)
    manifest = read_jsonl(paths["manifest"])
    print("[EXPG] manifest: %d rows (%d injection rejects) %s"
          % (len(manifest), n_rejected,
             json.dumps(dict(Counter(m["cell"] for m in manifest)))))

    # ---- phase 4: injected continuations
    prog_by_id = {p["program_id"]: p for p in programs}
    jobs = []
    for m in manifest:
        p = prog_by_id[m["program_id"]]
        base = {"run_id": m["run_id"], "program_id": m["program_id"],
                "condition": m["cell"], "shape": m["shape"]}
        jobs.append((base, tf.make_user_message(p["listing_text"]),
                     m["prefix_text"]))
    batched_generate(backend, jobs, paths["raw"], existing, args.max_new_cont,
                     args.batch_size, "cont")
    print("[EXPG] generate complete")


# ------------------------------------------------------------------ validate

def validate(args):
    guard_out_dir(args.out_dir)
    paths = out_paths(args.out_dir)
    programs = {p["program_id"]: p for p in read_jsonl(paths["programs"])}
    manifest = read_jsonl(paths["manifest"])
    raw = {r["run_id"]: r for r in read_jsonl(paths["raw"])}
    if os.path.exists(paths["validated"]):
        os.remove(paths["validated"])
    n = 0
    for m in manifest:
        p = programs[m["program_id"]]
        stmts = parse_program(p["stmt_texts"])
        r = raw.get(m["run_id"])
        rec = dict(m)
        rec["gold_run_id"] = gold_run_id(m["program_id"], m["seed"])
        if r is None or r.get("failed_generation"):
            rec.update({"generation_failed": True, "class": "generation_failed"})
            append_jsonl(paths["validated"], rec)
            n += 1
            continue
        inj = {k: m[k] for k in ("cf", "planted_var", "planted_value",
                                 "force_after_line", "expected_lines_from")}
        metrics = val_mod.classify_run(stmts, p, inj, r["continuation"])
        rec.update(metrics)
        rec["generation_failed"] = False
        rec["continuation"] = r["continuation"]
        append_jsonl(paths["validated"], rec)
        n += 1
    print("[EXPG] validated %d rows -> %s" % (n, paths["validated"]))


# --------------------------------------------------------------------- judge

JUDGE_DOUBT_PROMPT = """You will see the continuation of a line-by-line execution trace of a short Python program. Decide whether the text EXPRESSES DOUBT, HESITATION, SELF-CORRECTION, or flags an inconsistency about any earlier step (e.g., questioning a stated value, noting a contradiction or a bug, revising a claim). Mechanical trace lines with no commentary = NO.

Continuation:
---
{cont}
---
Answer with exactly one word: YES or NO."""

JUDGE_REJECT_PROMPT = """A value claim was planted into the middle of a line-by-line execution trace of a short Python program. You will see the planted trace line and the continuation that follows it. Decide whether the continuation REJECTS the planted claim. Answer YES if the continuation does EITHER of the following:
(1) verbally flags, questions, doubts, or comments on the planted line (e.g., noting it is wrong, a bug, or inconsistent with an earlier line), OR
(2) explicitly restates the planted variable with a DIFFERENT value than the planted one (a correction).
Answer NO if the continuation merely ignores the planted line, repeats it, or builds on the planted value.

Planted line:
---
{stmt}
---
Continuation:
---
{cont}
---
Answer with exactly one word: YES or NO."""

CONT_TRUNC = 1500  # legacy judge convention


def judge(args):
    guard_out_dir(args.out_dir)
    paths = out_paths(args.out_dir)
    rows = [r for r in read_jsonl(paths["validated"])
            if not r.get("generation_failed")]
    if not rows:
        raise SystemExit("run validate first")
    tooling = find_tooling()
    if tooling not in sys.path:
        sys.path.insert(0, tooling)
    import vllm_gen
    jobs, meta = [], []
    for r in rows:
        cont = (r.get("continuation") or "")[:CONT_TRUNC]
        jobs.append({"messages": [{"role": "user", "content":
                                   JUDGE_DOUBT_PROMPT.format(cont=cont)}]})
        meta.append((r["run_id"], "judge_doubt"))
        jobs.append({"messages": [{"role": "user", "content":
                                   JUDGE_REJECT_PROMPT.format(
                                       stmt=r["injected_trace_line"], cont=cont)}]})
        meta.append((r["run_id"], "judge_reject"))
    outs = vllm_gen.generate_continuations(JUDGE_MODEL, jobs, max_new_tokens=4)
    verdicts = defaultdict(dict)
    for (rid, kind), o in zip(meta, outs):
        txt = (o["text"] or "").strip().upper()
        verdicts[rid][kind] = txt.startswith("YES")
        verdicts[rid][kind + "_raw"] = txt[:16]
    if os.path.exists(paths["judge"]):
        os.remove(paths["judge"])
    for rid, v in verdicts.items():
        append_jsonl(paths["judge"], dict(v, run_id=rid, judge_model=JUDGE_MODEL,
                                          created_at=now_iso()))
    # merge into validated_outputs
    all_rows = read_jsonl(paths["validated"])
    os.remove(paths["validated"])
    for r in all_rows:
        v = verdicts.get(r["run_id"], {})
        r["judge_doubt"] = v.get("judge_doubt")
        r["judge_reject"] = v.get("judge_reject")
        append_jsonl(paths["validated"], r)
    print("[EXPG] judged %d rows (x2 prompts) with %s" % (len(rows), JUDGE_MODEL))


# ----------------------------------------------------------------- summarize

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / float(n)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cluster_boot(rows, metric, seed=0, n_boot=1000, cluster_key="program_id"):
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
        num = den = 0.0
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
    return [round(vals[int(0.025 * (len(vals) - 1))], 4),
            round(vals[int(0.975 * (len(vals) - 1))], 4)]


BOOL_METRICS = ("trace_valid", "final_output_valid", "next_read_absorbed",
                "final_output_absorbed", "repair_event", "doubt_lex",
                "doubt_broad", "judge_doubt", "judge_reject", "unparsed",
                "injection_dependent", "parroted", "derailed",
                "no_output_claim", "generation_failed")


def _mval(r, metric):
    v = r.get(metric)
    if v is None:
        return None
    return int(bool(v))


def summarize_cell(rows, seed):
    out = {"n": len(rows), "program_n": len({r["program_id"] for r in rows})}
    for metric in BOOL_METRICS:
        vals = [_mval(r, metric) for r in rows]
        known = [v for v in vals if v is not None]
        k, n = sum(known), len(known)
        out[metric] = {
            "n": n, "count": k,
            "rate": round(k / float(n), 4) if n else None,
            "wilson95": wilson(k, n),
            "cluster_boot95": cluster_boot(rows, lambda r, m=metric: _mval(r, m),
                                           seed=seed) if n else [None, None],
        }
    # dual denominator for the absorption/doubt headline (parsed-only)
    parsed = [r for r in rows if not r.get("unparsed") and not r.get("generation_failed")]
    out["parsed_only_n"] = len(parsed)
    for metric in ("next_read_absorbed", "doubt_lex", "judge_doubt", "repair_event"):
        vals = [_mval(r, metric) for r in parsed]
        known = [v for v in vals if v is not None]
        out[metric + "_parsed_only"] = (round(sum(known) / float(len(known)), 4)
                                        if known else None)
    # continuous: line-skip rate
    ls = [r.get("line_skip_rate") for r in rows if r.get("line_skip_rate") is not None]
    out["line_skip_rate"] = {
        "n": len(ls), "mean": round(sum(ls) / float(len(ls)), 4) if ls else None,
        "cluster_boot95": cluster_boot(rows, lambda r: r.get("line_skip_rate"),
                                       seed=seed) if ls else [None, None]}
    out["next_read_breakdown"] = dict(Counter(r.get("next_read_class", "none")
                                              for r in rows))
    out["r1_breakdown"] = dict(Counter(r.get("r1_class", "none") for r in rows))
    return out


def _rate(cell_summary, metric):
    m = cell_summary.get(metric)
    return m.get("rate") if isinstance(m, dict) else None


def summarize(args):
    guard_out_dir(args.out_dir)
    paths = out_paths(args.out_dir)
    rows = read_jsonl(paths["validated"])
    if not rows:
        raise SystemExit("run validate first")
    cohort = json.load(open(paths["cohort"])) if os.path.exists(paths["cohort"]) else {}
    cells = {}
    for cell in PILOT_CELLS:
        crows = [r for r in rows if r["cell"] == cell]
        if crows:
            cells[cell] = summarize_cell(crows, args.seed)

    # ---- gate G-B (floor gate, skeptic W5-1; PLAN2 5.1)
    judged = any(r.get("judge_doubt") is not None for r in rows)
    gb_inputs = {}
    viable = False
    for cell in ("adjacent_contradiction", "opfree_kr1"):
        c = cells.get(cell, {})
        doubt_j = _rate(c, "judge_doubt") if judged else None
        doubt_l = _rate(c, "doubt_lex")
        doubt = doubt_j if doubt_j is not None else doubt_l
        absorb = _rate(c, "next_read_absorbed")
        gb_inputs[cell] = {"judge_doubt": doubt_j, "doubt_lex": doubt_l,
                           "doubt_used": doubt, "next_read_absorbed": absorb}
        if (doubt is not None and doubt >= 0.10) or \
           (absorb is not None and absorb <= 0.85):
            viable = True
    gate_b = {"inputs": gb_inputs, "doubt_dv": "judge" if judged else "lexical",
              "criterion": "doubt >= 0.10 OR next-read absorption <= 0.85 on "
                           "adjacent_contradiction or opfree_kr1",
              "full_grid_viable": viable,
              "branch": "full_grid" if viable else "floor_documentation"}

    # ---- k_r manipulation probe (exploratory at pilot n; H7 shape preview)
    kr_probe = {}
    for metric in ("judge_doubt", "doubt_lex", "doubt_broad", "next_read_absorbed",
                   "final_output_absorbed", "repair_event", "judge_reject",
                   "trace_valid"):
        a = _rate(cells.get("opfree_kr1", {}), metric)
        b = _rate(cells.get("opfree_kr8", {}), metric)
        kr_probe[metric] = {"kr1": a, "kr8": b,
                            "delta_kr1_minus_kr8":
                                round(a - b, 4) if a is not None and b is not None
                                else None}

    summary = {
        "experiment": "EXPG_PROGTRACE_PILOT",
        "created_at": now_iso(),
        "cohort_summary": cohort.get("summary"),
        "gate_A": cohort.get("gate_A"),
        "gate_A_pass": cohort.get("gate_A_pass"),
        "cells": cells,
        "gate_B": gate_b,
        "kr_gradient_probe": kr_probe,
        "denominator_note": ("unparsed and generation-failed runs retained in "
                             "denominators (house convention); *_parsed_only "
                             "columns give the dual denominator"),
        "stats_note": "wilson95 + program-cluster bootstrap95 (n_boot=1000)",
    }
    write_json(paths["summary"], summary)
    print(json.dumps({"gate_B": gate_b, "kr_gradient_probe": kr_probe}, indent=2))
    print("[EXPG] summary -> %s" % paths["summary"])


# -------------------------------------------------------------------- report

def _fmt(m):
    if not isinstance(m, dict) or m.get("rate") is None:
        return "n.m."
    lo, hi = (m.get("cluster_boot95") or [None, None])
    ci = " [%.2f,%.2f]" % (lo, hi) if lo is not None else ""
    return "%.3f%s (%d/%d)" % (m["rate"], ci, m["count"], m["n"])


def report(args):
    guard_out_dir(args.out_dir)
    paths = out_paths(args.out_dir)
    summary = json.load(open(paths["summary"]))
    meta = json.load(open(paths["metadata"])) if os.path.exists(paths["metadata"]) else {}
    s0_path = os.path.join(paths["stage0_dir"], "stage0_report.json")
    s0 = json.load(open(s0_path)) if os.path.exists(s0_path) else None
    lines = []
    ap = lines.append
    ap("# EXPG_PROGTRACE pilot report")
    ap("")
    ap("Generated %s. Model %s @ %s (pinned=%s), backend %s, greedy." %
       (summary["created_at"], meta.get("model"), meta.get("model_revision"),
        meta.get("model_revision_pinned"), meta.get("backend")))
    ap("")
    if s0:
        ap("## Stage-0 (grammar gate)")
        ap("")
        ap("Gates: %s (pass=%s). Gold eval: %s" %
           (json.dumps(s0["gates"]), s0["gates_pass"],
            json.dumps(s0["gold_eval"], sort_keys=True)))
        ap("")
        ap("Final knobs: `%s`" % json.dumps(s0["knobs"], sort_keys=True))
        ap("")
    ap("## Cohort funnel + gate G-A")
    ap("")
    ap("%s, gate G-A: %s (pass=%s)" % (json.dumps(summary.get("cohort_summary")),
                                       json.dumps(summary.get("gate_A")),
                                       summary.get("gate_A_pass")))
    ap("")
    ap("## Stage-1 pilot cells (mid position, n<=35/cell)")
    ap("")
    cols = ["cell", "n", "trace_valid", "final_output_valid", "next_read_absorbed",
            "final_output_absorbed", "repair_event", "line_skip", "doubt_lex",
            "doubt_broad", "judge_doubt", "judge_reject", "unparsed"]
    ap("| " + " | ".join(cols) + " |")
    ap("|" + "---|" * len(cols))
    for cell in PILOT_CELLS:
        c = summary["cells"].get(cell)
        if not c:
            continue
        ls = c["line_skip_rate"]["mean"]
        row = [cell, str(c["n"]), _fmt(c["trace_valid"]),
               _fmt(c["final_output_valid"]), _fmt(c["next_read_absorbed"]),
               _fmt(c["final_output_absorbed"]), _fmt(c["repair_event"]),
               ("%.3f" % ls) if ls is not None else "n.m.",
               _fmt(c["doubt_lex"]), _fmt(c["doubt_broad"]),
               _fmt(c["judge_doubt"]), _fmt(c["judge_reject"]),
               _fmt(c["unparsed"])]
        ap("| " + " | ".join(row) + " |")
    ap("")
    ap("## Gate G-B (floor gate)")
    ap("")
    ap("```json")
    ap(json.dumps(summary["gate_B"], indent=2, sort_keys=True))
    ap("```")
    ap("")
    ap("## k_r manipulation probe (opfree_kr1 vs opfree_kr8; exploratory at pilot n)")
    ap("")
    ap("```json")
    ap(json.dumps(summary["kr_gradient_probe"], indent=2, sort_keys=True))
    ap("```")
    ap("")
    ap("Notes: %s. %s." % (summary["denominator_note"], summary["stats_note"]))
    with open(paths["report"], "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[EXPG] report -> %s" % paths["report"])


# ---------------------------------------------------------------------- test

def run_tests(_args):
    import progtrace_tests
    progtrace_tests.main()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["test", "stage0", "prepare", "generate",
                                     "validate", "judge", "summarize", "report"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=PINNED_REVISION)
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--cells", default=",".join(PILOT_CELLS))
    p.add_argument("--cap", type=int, default=35, help="max programs/cell (pilot <=35)")
    p.add_argument("--oversample", type=int, default=50,
                   help="programs generated per shape before the cohort filter")
    p.add_argument("--stage0-n", type=int, default=100)
    p.add_argument("--max-new-gold", type=int, default=512)
    p.add_argument("--max-new-cont", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--knob", action="append", default=None,
                   help="grammar knob override key=value (repeatable)")
    p.add_argument("--ignore-gate-a", action="store_true")
    p.add_argument("--confirm-timestamped-plan-progtrace", action="store_true",
                   help="required for anything beyond the pilot scope")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stage = {"test": run_tests, "stage0": stage0, "prepare": prepare,
             "generate": generate, "validate": validate, "judge": judge,
             "summarize": summarize, "report": report}[args.stage]
    stage(args)


if __name__ == "__main__":
    main()
