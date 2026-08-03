"""EXPH shared helpers: prompt construction, cleaning, validation rows, stats.

EXPH = API-model generality arm (descriptive/exploratory; NOT confirmatory).

Prompt contract:
  * INSTR + FEWSHOT are byte-identical to src/common.py (asserted at import).
  * GOLD prompt (both modes): single user message
        INSTR + FEWSHOT + "Q: {question} Prove: {target}\nA:"
  * mode "prefill" (Anthropic models that support assistant prefill):
        user turn = gold prompt; assistant turn pre-filled with
        " ".join(prefix_steps + [injected_statement]); the API continues it.
  * mode "instruct" (all providers + the local Qwen bridge): single user message
        INSTR + FEWSHOT + "Q: {question} Prove: {target}\nA: {prefix_text}" + CONT_INSTR
    Same rendered content across providers (no system message anywhere).
"""
import hashlib
import json
import math
import os
import random
import re
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.environ.get("LR_SRC") or os.path.join(EXP, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from validator import (  # noqa: E402
    DOUBT,
    parse_fact,
    strip_marker,
    validate_continuation,
)
import expa_global_expansion as expa  # noqa: E402  (stdlib-only at module level)
from expc_polarity_control import (  # noqa: E402
    audit_truth_status as expc_audit_truth_status,
    injection_points as expc_injection_points,
    opposite_pred,
    statement_predicate,
)

RESULTS_DIR = os.path.join(EXP, "results", "EXPH_API_MODELS")

# Byte-identical by construction: bound to the canonical block in src/
# (expa_global_expansion carries verbatim copies of src/common.py INSTR/FEWSHOT).
INSTR = expa.INSTR
FEWSHOT = expa.FEWSHOT
assert INSTR.startswith("You will be given facts and rules") and "yumpus" in FEWSHOT

CONT_INSTR = (
    "\n\nThe answer above is incomplete. Continue the proof EXACTLY from where the text after "
    '"A:" ends. Output only the remaining proof steps, one deduction at a time, in the same '
    "style as the examples, each step ending with a period. Do not repeat steps that are "
    "already written. Do not add any commentary. End with the statement to be proven."
)

# DIAGNOSTIC variant (not the registered EXPH prompt): stricter anti-restatement
# wording, used only to probe whether the bridge-gate failure is prompt-fixable.
if os.environ.get("EXPH_CONT_INSTR_VARIANT") == "v2_antirestate":
    CONT_INSTR = (
        "\n\nThe answer above is incomplete. Continue the proof from the exact point where the "
        'text after "A:" stops. Your reply must begin with the single next proof step that '
        "follows the last written statement, and must not restate, correct, or rewrite any "
        'statement that already appears after "A:" (not even the last one). Output only the '
        "remaining new proof steps, one deduction at a time, in the same style as the examples, "
        "each ending with a period. No commentary. End with the statement to be proven."
    )

POINT = "mid"
CORE_CONDITIONS = ("benign_paraphrase", "true_interruption", "one_hop_falsehood", "global_falsehood")
REUSE_CONDITIONS = ("cat_false_usable_d1", "cat_false_inert_d1")
MID_IDX = 1  # EXPA build_injections used idx = position_index (+seed 0); mid -> 1


def prompt_sha():
    return hashlib.sha256((INSTR + FEWSHOT + CONT_INSTR).encode()).hexdigest()


def gold_user_content(question, target):
    return INSTR + FEWSHOT + "Q: %s Prove: %s\nA:" % (question, target)


def prefix_text(prefix_steps, injected):
    return " ".join(list(prefix_steps) + [injected])


def instruct_user_content(question, target, pre_text):
    return (INSTR + FEWSHOT + "Q: %s Prove: %s\nA: %s" % (question, target, pre_text)
            + CONT_INSTR)


def sha_row(parts, n=24):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:n]


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", text.strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def clean_continuation(text):
    """Minimal disclosed post-processing for chat-model outputs:
    strip leading whitespace / a leading 'A:' marker, truncate at a new 'Q:' block."""
    t = (text or "").strip()
    if t.startswith("A:"):
        t = t[2:].lstrip()
    for marker in ("\nQ:", "\n\nQ:"):
        i = t.find(marker)
        if i != -1:
            t = t[:i].strip()
    return t


def solved(gen_text, target):
    sents = split_sentences(gen_text)
    return bool(sents) and norm(sents[-1]) == norm(target)


def echo_flag(continuation, injected, entity):
    """EXPD M4-style stated reliance: any continuation sentence re-asserts the planted predicate."""
    pred = statement_predicate(injected, entity)
    if pred is None:
        return False
    for s in split_sentences(continuation or ""):
        f = parse_fact(strip_marker(s), entity)
        if f and f[1] == pred:
            return True
    return False


def stated_complement_flag(continuation, injected, entity):
    """EXPE secondary DV: continuation explicitly states the complement of the planted claim."""
    pred = statement_predicate(injected, entity)
    if pred is None:
        return False
    try:
        comp = opposite_pred(pred)
    except ValueError:
        return False
    for s in split_sentences(continuation or ""):
        f = parse_fact(strip_marker(s), entity)
        if f and f[1] == comp:
            return True
    return False


def validate_row(mrow, raw):
    """Build one validated record from a manifest row + raw generation row."""
    base = {k: mrow.get(k) for k in (
        "run_id", "model_key", "provider", "model_id", "mode", "arm", "condition",
        "problem_id", "cluster_id", "injection_position", "seed", "sent_idx",
        "injected_statement", "audited_truth_status", "measured_local_d",
        "poisoning_measurable", "original_proof_validated")}
    if raw is None or raw.get("failed_generation"):
        base.update({"class": "generation_failed", "failed_generation": True,
                     "error": (raw or {}).get("error"), "valid_recovery": False,
                     "doubt": False, "inj_derivational": False, "inj_echo": False,
                     "stated_complement": False})
        return base
    cont = clean_continuation(raw.get("continuation", ""))
    v = validate_continuation(mrow["question"], mrow["prefix_steps"],
                              mrow["injected_statement"], cont, mrow["target"], mrow["entity"])
    base.update(v)
    base.update({
        "failed_generation": False,
        "valid_recovery": v["class"] == "valid_rederivation",
        "doubt": bool(v.get("acknowledged")),
        "inj_derivational": v["class"] == "poisoned",
        "inj_echo": echo_flag(cont, mrow["injected_statement"], mrow["entity"]),
        "stated_complement": stated_complement_flag(cont, mrow["injected_statement"], mrow["entity"]),
        "continuation": cont,
        "raw_continuation_prefix_stripped": cont != (raw.get("continuation") or "").strip(),
    })
    return base


# ------------------------------------------------------------------- stats

METRICS = ("valid_recovery", "inj_derivational", "inj_echo", "stated_complement",
           "doubt", "parroted", "derailed", "unparsed", "generation_failed")


def metric_value(r, metric):
    if metric in ("valid_recovery", "inj_derivational", "inj_echo",
                  "stated_complement", "doubt", "plant_dependent"):
        return int(bool(r.get(metric)))
    if metric in ("parroted", "derailed", "unparsed", "generation_failed"):
        return int(r.get("class") == metric)
    raise KeyError(metric)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def cluster_boot_rate(rows, metric, seed=0, n_boot=1000, cluster_key="cluster_id"):
    ids = sorted({r[cluster_key] for r in rows})
    if not ids:
        return [None, None]
    by_id = {}
    for r in rows:
        by_id.setdefault(r[cluster_key], []).append(r)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        num = den = 0
        for cid in sample:
            for r in by_id[cid]:
                den += 1
                num += metric(r)
        if den:
            vals.append(num / den)
    if not vals:
        return [None, None]
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 4),
            round(vals[int(0.975 * (len(vals) - 1))], 4)]


def summarize_cell(rows, seed=0, metrics=METRICS):
    out = {"n": len(rows), "cluster_n": len({r.get("cluster_id") for r in rows})}
    for metric in metrics:
        vals = [metric_value(r, metric) for r in rows]
        n, k = len(vals), sum(vals)
        out[metric] = {
            "count": k, "rate": round(k / n, 4) if n else None,
            "wilson95": wilson(k, n),
            "cluster_bootstrap95": cluster_boot_rate(
                rows, lambda r, m=metric: metric_value(r, m), seed=seed) if n else [None, None],
        }
    par = out["unparsed"]["rate"] or 0
    gf = out["generation_failed"]["rate"] or 0
    out["parse_rate"] = round(1.0 - par - gf, 4) if out["n"] else None
    return out


# ------------------------------------------------------- injection building

def build_core_injections(question, target, entity, steps, seed=0):
    """EXPA-standard 4 core conditions at mid on a model's OWN gold steps.
    Returns (rows, skips): rows = list of dicts with condition/si/injected/truth;
    audited with the expc canonical audit (records local_rule_distance too)."""
    pts = expc_injection_points(steps, entity)
    skips = []
    if pts is None:
        return [], [{"reason": "too_few_intermediate_entity_steps"}]
    si = pts[POINT]
    specs = [
        ("global_falsehood", expa.make_global_falsehood(question, entity, idx=MID_IDX + seed), "false"),
        ("benign_paraphrase", expa.make_paraphrase(steps, si, idx=MID_IDX), "true"),
        ("one_hop_falsehood", expa.make_negstep(steps, si), "false"),
        ("true_interruption", expa.make_true_interruption(question, steps, entity, idx=MID_IDX + seed), "true"),
    ]
    rows = []
    for condition, injected, want_truth in specs:
        if injected is None:
            skips.append({"condition": condition, "reason": "condition_unavailable"})
            continue
        truth = expc_audit_truth_status(question, injected, entity, tuple(steps[:si]))
        if truth["truth_status"] != want_truth:
            skips.append({"condition": condition, "reason": "truth_status_mismatch",
                          "truth": truth["truth_status"], "want": want_truth,
                          "injected": injected})
            continue
        rows.append({"condition": condition, "sent_idx": si, "injected_statement": injected,
                     "truth_audit": truth, "audited_truth_status": truth["truth_status"],
                     "measured_local_d": truth.get("local_rule_distance")})
    return rows, skips


def build_reuse_injection(world, steps, seed=0):
    """usable/inert planted claim at mid on the model's OWN gold trace for an EXPD-style
    generated world. Fail-closed: audited false AND measured d from prefix == 1."""
    entity = world["entity"]
    cell = world["cells"][0]
    pts = expc_injection_points(steps, entity)
    if pts is None:
        return None, {"reason": "too_few_intermediate_entity_steps"}
    si = pts[POINT]
    stmt = cell["injected_statement"]
    truth = expc_audit_truth_status(world["question"], stmt, entity, tuple(steps[:si]))
    if truth["truth_status"] != "false":
        return None, {"reason": "truth_status_mismatch", "truth": truth["truth_status"]}
    if truth.get("local_rule_distance") != 1:
        return None, {"reason": "measured_d_mismatch", "measured": truth.get("local_rule_distance")}
    return {"condition": cell["cell"], "sent_idx": si, "injected_statement": stmt,
            "truth_audit": truth, "audited_truth_status": "false",
            "measured_local_d": 1}, None


def plant_dependent_content(mrow, continuation):
    """Echo-corrected, ORDER-INDEPENDENT reuse audit (the metric the bridge arm
    showed is format-robust): count continuation sentences (!= the plant) whose
    predicate is underivable from the true state (world facts + prefix + all
    plant-free continuation content, grown to fixpoint) but derivable once the
    planted atom is added."""
    from validator import parse_world, closure, derivable
    q, entity = mrow["question"], mrow["entity"]
    rules, facts, _ = parse_world(q)
    reach = closure(rules)
    state = {p for e, p in facts if e == entity}
    for s in mrow["prefix_steps"]:
        f = parse_fact(strip_marker(s), entity)
        if f:
            state.add(f[1])
    plant = statement_predicate(mrow["injected_statement"], entity)
    preds = []
    for s in split_sentences(continuation or ""):
        f = parse_fact(strip_marker(s), entity)
        if f:
            preds.append(f[1])
    changed, ok = True, set()
    while changed:
        changed = False
        for p in preds:
            if p in ok or p == plant:
                continue
            if derivable(p, state, reach):
                state.add(p)
                ok.add(p)
                changed = True
    return sum(1 for p in preds if p != plant and p not in ok
               and plant and derivable(p, state | {plant}, reach))


# ------------------------------------------------------------------- io

def read_jsonl(path, limit=None):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
