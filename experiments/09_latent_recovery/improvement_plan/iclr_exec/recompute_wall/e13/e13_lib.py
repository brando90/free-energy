"""E13 runner shared library (ADDITIVE; no existing file is modified).

Single source of truth for the Build-2/2 runner seams:
  * world loading            RW/e13/worlds/*.jsonl (dry-run-audited pools)
  * injection                build_injection_e13 -- e11/EXPG transforms BY
                             IMPORT plus the four new grid transforms and the
                             probe transform, per RW/e13/README.md (binding
                             contract written by Build 1/2)
  * probe scoring            first-integer match against probe_answer
  * mock golds               interpreter-true trace (DRY MODE ONLY; live runs
                             always use the model's OWN gold, e11 regime)
  * summaries                Wilson + program-cluster bootstrap via e11_run
  * ledgers / projection     cumulative cost_ledger discovery + per-arm spend
                             projection printed BEFORE any live call

Scoring for every non-probe family goes through the EXISTING
validate.classify_run two-world path (unchanged, by import).
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RW = os.path.abspath(os.path.join(HERE, ".."))
E11_RUN_DIR = os.path.join(RW, "e11_run")
EXP = os.path.abspath(os.path.join(RW, "..", ".."))          # improvement_plan
EXPH_DIR = os.path.join(EXP, "exph")
for _d in (HERE, E11_RUN_DIR, EXPH_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import e11_run as e11                      # noqa: E402  (sets expg/e11_generator paths)
import trace_format as tf                  # noqa: E402
import inject as inj_mod                   # noqa: E402
import validate as val_mod                 # noqa: E402
from interp import execute, parse_program  # noqa: E402
import e13_gen                             # noqa: E402  (E13B_INC_ARM_A4 pin)

WORLDS_DIR = os.path.join(HERE, "worlds")
RESULTS_DIR = os.path.join(HERE, "results")

# ---- family inventory the runners drive (baselines kept in-file, paired) ----
CELL_FAMILIES = {
    "line_k0": ["anchor_k0", "line_k0", "line_k0_swap"],
    "grid_kr1": ["adjacent_contradiction", "opfree_kr1",
                 "grid_line_false_note_true", "grid_note_false_before",
                 "grid_note_true_before_line_false", "grid_line_false_copy_true",
                 "dose_off1", "dose_off10_20", "dose_large"],
    "k1_deference": ["depth_k1_bare", "depth_k1_full", "depth_k1_partial",
                     "k1_full_sum", "k1_tentative", "k1_probe"],
}
ALL_CELLS = list(CELL_FAMILIES)

# note-before-definition corners are CONDITIONAL on a parse-sanity gate
# (spec E13a-grid); the summarize stage computes the gate metrics per family.
CONDITIONAL_FAMILIES = ("grid_note_false_before", "grid_note_true_before_line_false")

PROBE_MAX_TOKENS = 64


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def load_worlds(cell):
    path = os.path.join(WORLDS_DIR, "%s.jsonl" % cell)
    rows = e11.read_jsonl(path)
    if not rows:
        raise SystemExit("world file missing/empty: %s" % path)
    return rows, path


def mock_gold_full(prog):
    """Interpreter-true trace (DRY MODE stand-in for the model's own gold;
    the live runners regenerate golds per model, e11 regime)."""
    stmts = parse_program(prog["stmt_texts"])
    return tf.render_true_trace(stmts, execute(stmts))


# ------------------------------------------------------------- injection

def _locate(raws, line_no, var, value):
    for i, ln in enumerate(raws):
        p = tf.parse_trace_line(ln)
        if p is not None and p.get("line") == line_no and p["kind"] == "assign":
            if p["var"] != var:
                raise inj_mod.InjectError("var_mismatch_line%d" % line_no)
            if p["value"] != value:
                raise inj_mod.InjectError("value_mismatch_line%d" % line_no)
            return i
    raise inj_mod.InjectError("line%d_not_found_in_gold" % line_no)


GRID_TRANSFORMS = ("line_false_note_true_after", "note_false_before_line",
                   "note_true_before_line_false", "line_false_copy_true")


def build_injection_e13(prog, family, gold_full_text):
    """Injection dispatch for every E13 family. Fail-closed (InjectError)
    exactly like inject.build_injection / e11_run.build_injection_e11.

    Contract: RW/e13/README.md section 'New transforms (runner contract)'.
    """
    spec = prog["families"][family]
    t = spec["transform"]
    if t == "value":
        # line_k0 / line_k0_swap / k1_full_sum / k1_tentative / opfree_kr1 /
        # standard opacity arms: splice spec['site_body'] over the model's own
        # gold site line (e11 semantics) or, for EXPG value families without a
        # site_body (opfree), replace_final_value -- both via existing code.
        if "site_body" in spec:
            return e11.build_injection_e11(prog, family, gold_full_text)
        return inj_mod.build_injection(prog, family, gold_full_text)
    if t in ("note", "note_true", "paraphrase"):
        return inj_mod.build_injection(prog, family, gold_full_text)
    if t == "probe":
        b = build_injection_e13(prog, spec["prefix_family"], gold_full_text)
        b = dict(b)
        b.update({"family": family, "probe": True,
                  "probe_question": spec["probe_question"],
                  "probe_answer": spec["probe_answer"],
                  "prefix_family": spec["prefix_family"],
                  "planted_var": spec["planted_var"],
                  "planted_value": spec["planted_value"],
                  "true_value": spec["true_value"],
                  "force_after_line": spec["force_after_line"]})
        return b
    if t in GRID_TRANSFORMS:
        return _build_grid_injection(prog, family, spec, gold_full_text)
    raise inj_mod.InjectError("unknown_transform_%s" % t)


def _build_grid_injection(prog, family, spec, gold_full_text):
    j = prog["site_line"]
    jd = j - 1
    u = prog["root_var"]
    vtrue = spec["true_value"]
    d = spec["planted_value"]
    t = spec["transform"]
    raws = inj_mod.gold_trace_raw_lines(gold_full_text)
    idx_def = _locate(raws, jd, u, vtrue)
    note_line = None
    if spec.get("note_var") is not None:
        note_line = tf.format_note_line(spec["note_var"], spec["note_value"])
    if t == "line_false_note_true_after":
        injected = [inj_mod.replace_final_value(raws[idx_def], d), note_line]
        prefix = raws[:idx_def] + injected
        exp_from = j
    elif t == "note_false_before_line":
        injected = [note_line]
        prefix = raws[:idx_def] + [note_line, raws[idx_def]]
        exp_from = j
    elif t == "note_true_before_line_false":
        injected = [note_line, inj_mod.replace_final_value(raws[idx_def], d)]
        prefix = raws[:idx_def] + injected
        exp_from = j
    elif t == "line_false_copy_true":
        idx_site = _locate(raws, j, prog["site_var"], prog["site_true_value"])
        if idx_site != idx_def + 1:
            raise inj_mod.InjectError("copy_line_not_adjacent_in_gold")
        injected = [inj_mod.replace_final_value(raws[idx_def], d)]
        prefix = raws[:idx_def] + injected + [raws[idx_site]]
        exp_from = j + 1
    else:
        raise inj_mod.InjectError("unknown_grid_transform_%s" % t)
    return {
        "family": family, "site_line": j, "transform": t,
        "injected_trace_line": " || ".join(x for x in injected if x),
        "original_trace_line": raws[idx_def],
        "prefix_text": "\n".join(prefix) + "\n",
        "step_replaced": False, "expected_lines_from": exp_from,
        "cf": True, "planted_var": u, "planted_value": d, "true_value": vtrue,
        "force_after_line": spec["force_after_line"],
        "delta_policy": spec.get("delta_policy"), "out_cf": spec.get("out_cf"),
    }


# ------------------------------------------------------------- probe scoring

_INT_RE = re.compile(r"-?\d+")


def score_probe(text, probe_answer, planted_value):
    """Direct-question scoring: first integer in the reply vs the true sum.
    Raw text is ALWAYS retained by the callers (spec: store raw answers)."""
    ints = [int(x) for x in _INT_RE.findall(text or "")]
    return {"probe_ints": ints[:8],
            "probe_correct": bool(ints) and ints[0] == probe_answer,
            "probe_answer_anywhere": probe_answer in ints,
            "probe_planted_echo": planted_value in ints,
            "probe_no_number": not ints}


def probe_messages(user_msg, prefix_text, question):
    """Same perturbed prefix; the trace-continuation request is REPLACED by the
    direct question, delivered as a follow-up user turn after the (closed)
    assistant turn that contains the perturbed partial trace."""
    return [{"role": "user", "content": user_msg},
            {"role": "assistant", "content": prefix_text.rstrip()},
            {"role": "user", "content": question}]


# ------------------------------------------------------------- validation

def validate_row(prog, b, continuation_text):
    """Standard families -> classify_run + 3-way label; probe -> probe score."""
    if b.get("probe"):
        rec = score_probe(continuation_text, b["probe_answer"], b["planted_value"])
        rec["label"] = "probe_correct" if rec["probe_correct"] else "probe_wrong"
        return rec
    stmts = parse_program(prog["stmt_texts"])
    inj = {k: b[k] for k in ("cf", "planted_var", "planted_value",
                             "force_after_line", "expected_lines_from")}
    met = val_mod.classify_run(stmts, prog, inj, continuation_text)
    rec = {k: met[k] for k in ("final_output_absorbed", "final_output_valid",
                               "repair_event", "next_read_absorbed",
                               "next_read_class", "trace_valid", "doubt_lex",
                               "doubt_broad", "unparsed", "unparsed_line_count",
                               "prose_line_count", "line_skip_rate")}
    rec["judge_reject"] = None
    rec["label"] = e11.three_way_label(rec)
    return rec


# ------------------------------------------------------------- summaries

def summarize_families(rows, seed=0):
    """Per-family: 3-way label rates (Wilson + program-cluster bootstrap) for
    standard families; probe accuracy for the probe family; parse-sanity gate
    metrics for the CONDITIONAL corners."""
    cells = {}
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for fam, frows in sorted(by_fam.items()):
        if any("probe_correct" in r for r in frows):
            n = len(frows)
            k = sum(1 for r in frows if r.get("probe_correct"))
            cells[fam] = {
                "n": n, "program_n": len({r["program_id"] for r in frows}),
                "probe_correct": {
                    "count": k, "rate": round(k / n, 4) if n else None,
                    "wilson95": e11.wilson(k, n),
                    "cluster_boot95": e11.cluster_boot_rate(
                        frows, lambda r: bool(r.get("probe_correct")), seed=seed)},
                "probe_answer_anywhere_rate": round(
                    sum(1 for r in frows if r.get("probe_answer_anywhere")) / n, 4) if n else None,
                "probe_planted_echo_rate": round(
                    sum(1 for r in frows if r.get("probe_planted_echo")) / n, 4) if n else None,
                "probe_no_number_rate": round(
                    sum(1 for r in frows if r.get("probe_no_number")) / n, 4) if n else None,
            }
            continue
        c = e11._cell_summary(frows, seed)
        live = [r for r in frows if not r.get("generation_failed")]
        if live:
            c["parse_gate"] = {
                "unparsed_rate": round(sum(1 for r in live if r.get("unparsed")) /
                                       len(live), 4),
                "mean_unparsed_lines": round(sum(r.get("unparsed_line_count") or 0
                                                 for r in live) / len(live), 3),
                "mean_prose_lines": round(sum(r.get("prose_line_count") or 0
                                              for r in live) / len(live), 3),
                "mean_line_skip": round(sum(r.get("line_skip_rate") or 0
                                            for r in live) / len(live), 4),
            }
        if fam in CONDITIONAL_FAMILIES:
            c["conditional"] = ("note-before-definition corner: CONDITIONAL on "
                                "this parse-sanity gate (spec E13a-grid); "
                                "Review agent signs off before scaling")
        cells[fam] = c
    return cells


# ------------------------------------------------------------- ledgers / spend

def all_ledgers(exclude_path=None):
    """Every cost_ledger.json that must count toward the cumulative hard stop:
    the whole recompute_wall tree + the EXPH/EXPH2 API ledgers."""
    found = []
    for dp, _dn, fn in os.walk(RW):
        if ".venv" in dp or "__pycache__" in dp:
            continue
        for f in fn:
            if f == "cost_ledger.json":
                found.append(os.path.join(dp, f))
    exp_root = os.path.abspath(os.path.join(EXP, ".."))     # 09_latent_recovery
    fixed = [os.path.join(exp_root, "results", "EXPH_API_MODELS", "cost_ledger.json")]
    for sub in ("regime2", "ladder", "instruct_panel", "completions_probe"):
        fixed.append(os.path.join(exp_root, "results", "EXPH2_FRONTIER_FOLLOWUPS",
                                  sub, "cost_ledger.json"))
    found += [p for p in fixed if os.path.exists(p)]
    if exclude_path:
        found = [p for p in found
                 if os.path.abspath(p) != os.path.abspath(exclude_path)]
    return sorted(set(found))


def est_tokens(text):
    """Coarse token estimate for spend projection (chars/4, floor 1)."""
    return max(1, len(text or "") // 4)


def project_spend(tag, jobs_by_arm, price_in, price_out, out_tokens_est):
    """jobs_by_arm: {arm: [(input_text_len_tokens, n_calls_at_that_len), ...]}
    Prints and returns the per-arm projection. Called BEFORE any live call."""
    proj = {}
    for arm, items in sorted(jobs_by_arm.items()):
        n = sum(c for _t, c in items)
        tin = sum(t * c for t, c in items)
        tout = out_tokens_est * n
        usd = tin / 1e6 * price_in + tout / 1e6 * price_out
        proj[arm] = {"n_calls": n, "input_tokens_est": tin,
                     "output_tokens_est": tout, "usd_est": round(usd, 2)}
    total = round(sum(v["usd_est"] for v in proj.values()), 2)
    print("[%s] PROJECTED SPEND (estimates; chars/4 input, %d-tok output):"
          % (tag, out_tokens_est))
    for arm, v in proj.items():
        print("  %-34s calls=%-5d est=$%.2f" % (arm, v["n_calls"], v["usd_est"]))
    print("  %-34s TOTAL est=$%.2f" % ("", total))
    return {"per_arm": proj, "total_usd_est": total}


# ------------------------------------------------------------- example dump

def dump_examples(out_path, examples):
    """examples: {arm_or_family: [ {..fully assembled prompt record..}, ... ]}"""
    e11.write_json(out_path, examples)
    print("[e13] example prompts -> %s" % out_path)
