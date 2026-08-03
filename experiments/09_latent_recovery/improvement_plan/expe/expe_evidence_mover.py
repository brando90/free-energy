"""EXPE: evidence-mover on natural PrOntoQA worlds (question augmentation).

Design (PLAN2.md section 4 MOD-7/MOD-11, reports/causal2.md section 1.2): a fixed
CWA-false falsehood F ("E is a fcat.") is injected into the model's OWN gold trace
at a fixed position (replacement of the step at si, legacy protocol); the QUESTION
(never the trace) gains exactly one rule sentence per arm, same template, inserted
at the same fixed "back" slot (just before the facts, imitating
context_check.move_rule). The trace-side stimulus is byte-identical across arms.

Arms:
  REFUTING_d1 / _d2 / _d3   "Every g is not a fcat."  antecedent g entailed for the
                            entity at k-1 hops from the prefix state; measured
                            shortest_rule_distance(not_cat fcat) == k; F flips
                            CWA-false -> provably-false (disclosed status change).
  FREQ_MATCHED_NONREFUTING  "Every z is not a fcat."  z never entailed for the
                            entity: same "not a <fcat>" tokens, no derivability.
  TEMPLATE_IRRELEVANT       "Every g is not an icat." entailed antecedent, negative
                            head over a category unreachable from fcat.
  BASELINE_FILLER           "Every x is not a y."     x, y entity-disconnected.

Fail-closed audits per row (MC-4/MC-6): F audited CWA-false in the ORIGINAL
question; measured d == designed d in refuting arms and F becomes provably-false
there; F remains NOT locally refutable (no finite d) in the control arms; the
ORIGINAL gold continuation still validates against the augmented world (re-solve /
re-validate rate reported per arm, MOD-11); the added rule can never derive the
target (no accidental shortcut).

Protocol deviation disclosed in print (MOD-11): the own-trace was generated under
the UNAUGMENTED question and all arms replay it under an augmented question --
equally off-policy across arms.

Staged usage (from $EXP/improvement_plan/expe/, one pinned GPU):

  PYV=$EXP/improvement_plan/tooling/.venv-vllm/bin/python
  $PYV expe_evidence_mover.py test
  $PYV expe_evidence_mover.py prepare  --out-dir $EXP/results/EXPE_EVIDENCE_MOVER_SMOKE --target 20 --overwrite
  source $EXP/improvement_plan/tooling/env.sh
  CUDA_VISIBLE_DEVICES=0 $PYV expe_evidence_mover.py generate --out-dir ... --target 20 --backend vllm
  $PYV expe_evidence_mover.py validate  --out-dir ...
  $PYV expe_evidence_mover.py summarize --out-dir ...
  $PYV expe_evidence_mover.py report    --out-dir ...

or `smoke` to run all of the above (target hard-capped at SMOKE_MAX_TARGET).

PRE-REGISTRATION GATE: any run with --target > SMOKE_MAX_TARGET refuses to start
unless --prereg-evidence "<public commit hash / OSF url>" is supplied (PLAN2
section 9.10: external timestamp precedes the first EXPE generation).
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
import stat
import sys
import traceback
from collections import Counter, defaultdict

sys.dont_write_bytecode = True  # never write __pycache__ into read-only src/

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.environ.get("LR_EXP_BASE") or os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(EXP, "src")
TOOLING = os.environ.get("LR_TOOLING") or os.path.abspath(os.path.join(HERE, "..", "tooling"))
sys.path.insert(0, SRC)

from validator import (  # noqa: E402  (read-only import from $EXP/src)
    DOUBT,
    closure,
    derivable,
    parse_fact,
    parse_rule,
    parse_world,
    strip_marker,
    validate_continuation,
)

DATA_PILOT = os.path.join(EXP, "data", "pilot.jsonl")
GOLD_ROLLOUTS = os.path.join(EXP, "results", "gold", "rollouts.jsonl")
# Expanded gold cohort (PLAN2 v1.1 section C.1): 220 six-arm-available problems,
# produced by results/EXPE_GOLD_EXPANSION/ (own-trace gold under vLLM, double
# filter re-applied at load time below). Selected with --cohort expanded (the
# full-run default); --cohort legacy keeps the original pilot.jsonl +
# results/gold/rollouts.jsonl path byte-for-byte.
EXPANDED_COHORT = os.path.join(EXP, "results", "EXPE_GOLD_EXPANSION", "cohort.jsonl")
DEFAULT_OUT = os.path.join(EXP, "results", "EXPE_EVIDENCE_MOVER_SMOKE")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SEED = 0
SMOKE_MAX_TARGET = 20  # per-arm n cap without a pre-registration timestamp

ARM_REFUTING_D1 = "REFUTING_d1"
ARM_REFUTING_D2 = "REFUTING_d2"
ARM_REFUTING_D3 = "REFUTING_d3"
ARM_FREQ = "FREQ_MATCHED_NONREFUTING"
ARM_TEMPLATE = "TEMPLATE_IRRELEVANT"
ARM_FILLER = "BASELINE_FILLER"
ARMS = (ARM_REFUTING_D1, ARM_REFUTING_D2, ARM_REFUTING_D3, ARM_FREQ, ARM_TEMPLATE, ARM_FILLER)
DESIGNED_D = {
    ARM_REFUTING_D1: 1,
    ARM_REFUTING_D2: 2,
    ARM_REFUTING_D3: 3,
    ARM_FREQ: None,
    ARM_TEMPLATE: None,
    ARM_FILLER: None,
}

# Prompt block copied verbatim from src/common.py (SHA-pinned in run_metadata).
INSTR = ("You will be given facts and rules about fictional creatures, then asked to prove a statement. "
         "Answer with only the proof: a sequence of statements, one deduction at a time, in the exact "
         "style of the examples. End with the statement to be proven.\n\n")
FEWSHOT = """Q: Every yumpus is a dumpus. Dumpuses are tumpuses. Tumpuses are not bright. Sam is a yumpus. Prove: Sam is not bright.
A: Sam is a yumpus. Every yumpus is a dumpus. Sam is a dumpus. Dumpuses are tumpuses. Sam is a tumpus. Tumpuses are not bright. Sam is not bright.

Q: Each gorpus is a sterpus. Sterpuses are red. Every borpus is a gorpus. Alex is a borpus. Prove: Alex is red.
A: Alex is a borpus. Every borpus is a gorpus. Alex is a gorpus. Each gorpus is a sterpus. Alex is a sterpus. Sterpuses are red. Alex is red.

"""


# ---------------------------------------------------------------- small utils

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


def prompt_hash():
    return hashlib.sha256((INSTR + FEWSHOT).encode()).hexdigest()


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


# ------------------------------------------------------------- world analysis

def direct_rule_map(rules):
    direct = defaultdict(set)
    for lhs, rhs in rules:
        direct[lhs].add(rhs)
    return direct


def world_categories(question):
    return sorted({singular(c) for c in re.findall(r"\b([a-z]+pus(?:es)?)\b", question)})


def question_world(question, entity, prefix_steps=()):
    """Parse the question; return closure structures + premise and prefix states."""
    rules, facts, unparsed = parse_world(question)
    direct = direct_rule_map(rules)
    reach = closure(rules)
    premises = {p for e, p in facts if e == entity}
    state = set(premises)
    for s in prefix_steps:
        f = parse_fact(strip_marker(s), entity)
        if f:
            state.add(f[1])
    return {"rules": rules, "direct": direct, "reach": reach,
            "premises": premises, "prefix_state": state, "unparsed": unparsed}


def shortest_rule_distance(pred, state, direct, max_depth=8):
    """Verbatim from src/expc_polarity_control.py:340-356 (canonical d, PLAN2 s3)."""
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


def hop_distance(cat, state, direct):
    """Hops needed to entail ('cat', cat) from state (0 if already in state)."""
    return shortest_rule_distance(("cat", cat), state, direct)


def entity_sentences(steps, entity):
    return [i for i, s in enumerate(steps) if s.split() and s.split()[0] == entity]


def injection_points(steps, entity):
    """Legacy protocol (perturb.py / context_check.py): interior entity sentences."""
    ents = entity_sentences(steps, entity)
    inter = ents[1:-1]
    if len(inter) < 3:
        return None
    return {"early": inter[0], "mid": inter[len(inter) // 2], "late": inter[-1]}


# --------------------------------------------------------------- construction

def rule_sentence(body_cat, head_cat):
    """Single shared template for every arm (+1 rule, same polarity/shape)."""
    return f"Every {body_cat} is not {article(head_cat)} {head_cat}."


# Fresh nonce nouns (repo -pus style) for control-arm slots when the natural
# world's unentailed-category pool is exhausted. MEASURED availability on the
# legacy gold cohort: 95/130 injectable worlds have ZERO unentailed categories
# and the rest have 1-4, while a fully in-vocabulary distinct six-arm set needs
# 5 -- so without this disclosed fallback the paired availability is 0.
# The FREQ arm's antecedent is NEVER fresh (frequency matching is its point).
FRESH_NOUNS = ("florpus", "blorpus", "chorpus", "dorpus", "frompus", "glimpus",
               "hompus", "kerpus", "morpus", "plumpus", "quimpus", "storpus",
               "trompus", "urpus", "velpus", "whompus")
VOCAB_TIERS = ("A_invocab_distinct", "B_invocab_reused", "C_fresh_nouns")


def fresh_noun_pool(question, k=4):
    """Deterministic fresh -pus nouns absent from question AND prompt block."""
    text = (question + " " + INSTR + " " + FEWSHOT).lower()
    out = []
    for n in FRESH_NOUNS:
        if re.search(rf"\b{n}(?:es)?\b", text):
            continue
        out.append(n)
        if len(out) >= k:
            break
    return out


def augment_question(question, rule_sent):
    """Insert one rule at the fixed 'back' slot: just before the first fact
    (context_check.move_rule position='back'). Returns (aug_q, index, offset)."""
    sents = split_sentences(question)
    if rule_sent in sents:
        return None, None, None
    fact_idxs = [i for i, s in enumerate(sents) if parse_fact(s) and not parse_rule(s)]
    ins = min(fact_idxs) if fact_idxs else len(sents)
    new = sents[:ins] + [rule_sent] + sents[ins:]
    aug_q = " ".join(new)
    off = round(aug_q.index(rule_sent) / max(1, len(aug_q)), 3)
    return aug_q, ins, off


def falsehood_candidates(question, entity, prefix_steps):
    """CWA-false categories for E, complement not entailed and not locally
    refutable from the prefix state (original world). Ordered so that
    poisoning-measurable (fcat has outgoing rules) candidates come first."""
    w = question_world(question, entity, prefix_steps)
    ranked = []
    for c in world_categories(question):
        if derivable(("cat", c), w["premises"], w["reach"]):
            continue
        if derivable(("not_cat", c), w["premises"], w["reach"]):
            continue
        if shortest_rule_distance(("not_cat", c), w["prefix_state"], w["direct"]) is not None:
            continue
        measurable = any(lhs == c for lhs, _ in w["rules"])
        ranked.append((0 if measurable else 1, c, measurable))
    return [(c, m) for _, c, m in sorted(ranked)]


def gold_suffix_cats(steps, si, entity):
    """Positive categories the ORIGINAL gold trace states after the injection
    point; preferred antecedents avoid these so d stays static under replay."""
    out = set()
    for s in steps[si:]:
        f = parse_fact(strip_marker(s), entity)
        if f and f[1][0] == "cat":
            out.add(f[1][1])
    return out


def pick_preferring_off_suffix(pool, suffix_cats):
    """Deterministic: sorted, preferring categories the gold suffix never states."""
    pool = sorted(pool)
    off = [g for g in pool if g not in suffix_cats]
    if off:
        return off[0]
    return pool[0] if pool else None


def build_arm_specs(question, steps, entity, si, fcat):
    """Construct all six arm specs for one (problem, position, F). Returns
    (specs_by_arm, unavailable_reasons_by_arm). Construction is deterministic;
    every spec is subsequently re-audited fail-closed (audit_arm)."""
    w = question_world(question, entity, steps[:si])
    existing = set(w["rules"])
    cats = world_categories(question)
    hop = {c: hop_distance(c, w["prefix_state"], w["direct"]) for c in cats}
    suffix = gold_suffix_cats(steps, si, entity)
    unentailed = [c for c in cats
                  if not derivable(("cat", c), w["premises"], w["reach"]) and c != fcat]

    specs, reasons = {}, {}

    for arm, k in ((ARM_REFUTING_D1, 1), (ARM_REFUTING_D2, 2), (ARM_REFUTING_D3, 3)):
        pool = [g for g in cats if hop.get(g) == k - 1
                and (g, ("not_cat", fcat)) not in existing and g != fcat]
        g = pick_preferring_off_suffix(pool, suffix)
        if g is None:
            reasons[arm] = f"no_antecedent_at_hop_{k - 1}"
            continue
        specs[arm] = {"body": g, "head": fcat, "antecedent_hop": k - 1,
                      "antecedent_in_gold_suffix": g in suffix}

    # Control-arm slot assignment with a disclosed vocabulary-tier ladder:
    # prefer unused in-vocabulary unentailed categories (tier A), then reuse
    # in-vocabulary ones across arms (tier B), then fresh nonce nouns for
    # filler/template slots only (tier C). z (FREQ antecedent) is NEVER fresh.
    fresh = list(fresh_noun_pool(question))
    used_invocab, any_reuse, any_fresh = set(), False, False

    def take(pred, allow_fresh):
        nonlocal any_reuse, any_fresh
        cand = [c for c in unentailed if pred(c) and c not in used_invocab]
        if cand:
            used_invocab.add(cand[0])
            return cand[0], "question_vocab"
        cand = [c for c in unentailed if pred(c)]
        if cand:
            any_reuse = True
            return cand[0], "question_vocab_reused"
        if allow_fresh and fresh:
            any_fresh = True
            return fresh.pop(0), "fresh"
        return None, None

    z, z_src = take(lambda c: (c, ("not_cat", fcat)) not in existing,
                    allow_fresh=False)
    if z is None:
        reasons[ARM_FREQ] = "no_invocab_unentailed_freq_antecedent"
    else:
        specs[ARM_FREQ] = {"body": z, "head": fcat, "antecedent_hop": None,
                           "antecedent_in_gold_suffix": z in suffix,
                           "body_source": z_src, "head_source": "question_vocab"}

    if ARM_REFUTING_D1 in specs:
        g0 = specs[ARM_REFUTING_D1]["body"]
        icat, i_src = take(
            lambda c: ("cat", c) not in w["reach"].get(fcat, set())
            and (g0, ("not_cat", c)) not in existing,
            allow_fresh=True)
        if icat is None:
            reasons[ARM_TEMPLATE] = "no_irrelevant_head_category"
        else:
            specs[ARM_TEMPLATE] = {"body": g0, "head": icat, "antecedent_hop": 0,
                                   "antecedent_in_gold_suffix": g0 in suffix,
                                   "body_source": "question_vocab",
                                   "head_source": i_src}
    else:
        reasons[ARM_TEMPLATE] = "no_d1_antecedent_for_template"

    x, x_src = take(lambda c: True, allow_fresh=True)
    y, y_src = (None, None)
    if x is not None:
        y, y_src = take(lambda c: c != x and (x, ("not_cat", c)) not in existing,
                        allow_fresh=True)
    if x is None or y is None:
        reasons[ARM_FILLER] = "no_disconnected_filler_pair"
    else:
        specs[ARM_FILLER] = {"body": x, "head": y, "antecedent_hop": None,
                             "antecedent_in_gold_suffix": x in suffix,
                             "body_source": x_src, "head_source": y_src}

    tier = (VOCAB_TIERS[2] if any_fresh else
            VOCAB_TIERS[1] if any_reuse else VOCAB_TIERS[0])
    for arm, spec in specs.items():
        spec["arm"] = arm
        spec.setdefault("body_source", "question_vocab")
        spec.setdefault("head_source", "question_vocab")
        spec["vocab_tier"] = tier
        spec["rule_sentence"] = rule_sentence(spec["body"], spec["head"])
    return specs, reasons


# --------------------------------------------------------------------- audits

def audit_arm(question, entity, target, steps, si, gold_text, fcat, arm, spec):
    """Fail-closed audit for one arm row. Returns dict of named boolean checks,
    measured quantities, and 'fully_audited'. Every check must be True."""
    rule_sent = spec["rule_sentence"]
    aug_q, ins_idx, offset = augment_question(question, rule_sent)
    checks, meta = {}, {}
    checks["augmentation_constructible"] = aug_q is not None
    if aug_q is None:
        return {"checks": checks, "fully_audited": False, "aug_question": None,
                "insert_index": None, "char_offset": None, **meta}

    wo = question_world(question, entity, steps[:si])
    wa = question_world(aug_q, entity, steps[:si])
    opp_f = ("not_cat", fcat)
    head_pred = ("not_cat", spec["head"])
    designed = DESIGNED_D[arm]

    # (1) F audited CWA-false in the ORIGINAL question.
    checks["falsehood_unentailed_original"] = not derivable(("cat", fcat), wo["premises"], wo["reach"])
    checks["falsehood_complement_unentailed_original"] = not derivable(opp_f, wo["premises"], wo["reach"])
    d_orig = shortest_rule_distance(opp_f, wo["prefix_state"], wo["direct"])
    checks["falsehood_not_locally_refutable_original"] = d_orig is None
    meta["measured_d_original"] = d_orig

    # (2) the added rule is new, parses back to the intended rule, and the
    # augmented world parses with exactly one extra rule and no new unparsed.
    checks["added_rule_not_duplicate"] = (
        (spec["body"], head_pred) not in set(wo["rules"]) and rule_sent not in question)
    checks["added_rule_parses"] = parse_rule(rule_sent) == (spec["body"], head_pred)
    checks["augmented_world_parses"] = (
        len(wa["unparsed"]) == len(wo["unparsed"])
        and len(wa["rules"]) == len(wo["rules"]) + 1)

    # (3) F status in the augmented world; measured d vs designed d.
    measured_d = shortest_rule_distance(opp_f, wa["prefix_state"], wa["direct"])
    meta["measured_d_augmented"] = measured_d
    checks["falsehood_still_unentailed_augmented"] = not derivable(("cat", fcat), wa["premises"], wa["reach"])
    if designed is not None:
        checks["measured_d_equals_designed"] = measured_d == designed
        # disclosed status change: CWA-false -> provably-false
        checks["falsehood_becomes_provably_false"] = derivable(opp_f, wa["premises"], wa["reach"])
        checks["antecedent_hop_as_designed"] = (
            hop_distance(spec["body"], wo["prefix_state"], wo["direct"]) == designed - 1)
        checks["refuting_head_is_falsehood_cat"] = spec["head"] == fcat
    else:
        checks["remains_not_locally_refutable"] = measured_d is None
        checks["falsehood_stays_cwa_false"] = not derivable(opp_f, wa["premises"], wa["reach"])

    # (4) arm-specific invariants.
    if arm == ARM_FREQ:
        checks["freq_head_is_falsehood_cat"] = spec["head"] == fcat
        checks["freq_antecedent_never_entailed"] = not derivable(
            ("cat", spec["body"]), wa["premises"], wa["reach"])
        # frequency matching is this arm's point: antecedent must come from the
        # question's own vocabulary, never a fresh noun.
        checks["freq_antecedent_in_question_vocab"] = (
            spec["body"] in world_categories(question))
    elif arm == ARM_TEMPLATE:
        checks["template_antecedent_entailed_hop0"] = ("cat", spec["body"]) in wo["prefix_state"]
        checks["template_head_not_falsehood_cat"] = spec["head"] != fcat
        checks["template_head_unreachable_from_falsehood"] = (
            ("cat", spec["head"]) not in wo["reach"].get(fcat, set()))
        checks["template_head_not_entailed"] = not derivable(
            ("cat", spec["head"]), wa["premises"], wa["reach"])
    elif arm == ARM_FILLER:
        checks["filler_body_never_entailed"] = not derivable(
            ("cat", spec["body"]), wa["premises"], wa["reach"])
        checks["filler_head_never_entailed"] = not derivable(
            ("cat", spec["head"]), wa["premises"], wa["reach"])
        checks["filler_head_not_falsehood_cat"] = spec["head"] != fcat

    # (5) MOD-11 / MC-6: the ORIGINAL gold continuation re-validates against the
    # augmented question's world (fail-closed AND reported as re-solve rate).
    gv = validate_continuation(aug_q, [], None, gold_text, target, entity)
    checks["gold_revalidates_under_augmentation"] = gv["class"] == "valid_rederivation"
    meta["gold_revalidation_class"] = gv["class"]

    # (6) the added rule is never usable to derive the target (no shortcut).
    tf = parse_fact(target, entity)
    checks["target_parses"] = tf is not None
    checks["no_target_shortcut"] = tf is not None and tf[1] != head_pred
    checks["target_still_derivable_augmented"] = (
        tf is not None and derivable(tf[1], wa["premises"], wa["reach"]))

    return {"checks": checks, "fully_audited": all(checks.values()),
            "aug_question": aug_q, "insert_index": ins_idx, "char_offset": offset, **meta}


def build_problem_arms(question, steps, entity, target, si, gold_text):
    """Evaluate every falsehood candidate; among those admitting all six
    fully-audited arms, select by (poisoning-measurable first, best vocabulary
    tier, candidate order). Returns (result_or_None, diagnostics)."""
    diags = {"n_falsehood_candidates": 0, "arm_available_under_some_F": set(),
             "audit_failures": []}
    cands = falsehood_candidates(question, entity, steps[:si])
    diags["n_falsehood_candidates"] = len(cands)
    successes = []
    for idx, (fcat, measurable) in enumerate(cands):
        specs, reasons = build_arm_specs(question, steps, entity, si, fcat)
        audits = {}
        ok = True
        for arm in ARMS:
            if arm not in specs:
                ok = False
                continue
            a = audit_arm(question, entity, target, steps, si, gold_text, fcat, arm, specs[arm])
            audits[arm] = a
            if a["fully_audited"]:
                diags["arm_available_under_some_F"].add(arm)
            else:
                ok = False
                diags["audit_failures"].append({
                    "fcat": fcat, "arm": arm,
                    "failed_checks": [k for k, v in a["checks"].items() if not v]})
        if ok:
            tier = specs[ARM_REFUTING_D1]["vocab_tier"]
            falsehood = f"{entity} is {article(fcat)} {fcat}."
            successes.append((
                (0 if measurable else 1, VOCAB_TIERS.index(tier), idx),
                {"fcat": fcat, "falsehood_statement": falsehood,
                 "poisoning_measurable": measurable, "vocab_tier": tier,
                 "specs": specs, "audits": audits}))
    if successes:
        return min(successes, key=lambda t: t[0])[1], diags
    return None, diags


# ------------------------------------------------------------------- cohort

def load_gold_cohort(source="legacy"):
    """Gold cohort under the double filter (PLAN2 section 10).

    source="expanded" (full-run default via --cohort): the PLAN2 v1.1 C.1
    cohort, results/EXPE_GOLD_EXPANSION/cohort.jsonl (already double-filtered
    at expansion time; the validator half of the filter is re-applied here
    fail-closed).
    source="legacy": the original pilot.jsonl + results/gold/rollouts.jsonl
    path, unchanged.
    """
    rows, ladder = [], Counter()
    if source == "expanded":
        if not os.path.exists(EXPANDED_COHORT):
            raise SystemExit(
                f"REFUSING: --cohort expanded but {EXPANDED_COHORT} is missing. "
                "Run the gold expansion first, or select --cohort legacy.")
        gold = read_jsonl(EXPANDED_COHORT)
        ladder["expanded_cohort_rows"] = len(gold)
        for g in gold:
            steps = split_sentences(g["gen_text"])
            if not steps:
                ladder["empty_trace"] += 1
                continue
            entity = g["entity"]
            gv = validate_continuation(g["question"], [], None, g["gen_text"],
                                       g["target"], entity)
            if gv["class"] != "valid_rederivation":
                ladder["gold_not_validator_valid"] += 1
                continue
            ladder["gold_validated_cohort"] += 1
            rows.append({"problem_id": g["problem_id"], "question": g["question"],
                         "target": g["target"], "entity": entity,
                         "gold_text": g["gen_text"], "steps": steps})
        return rows, ladder
    data = {r["id"]: r for r in read_jsonl(DATA_PILOT)}
    gold = read_jsonl(GOLD_ROLLOUTS)
    ladder["gold_rollouts_total"] = len(gold)
    for g in gold:
        if not g.get("solved"):
            ladder["gold_not_solved"] += 1
            continue
        inst = data.get(g["id"])
        if inst is None:
            ladder["no_dataset_instance"] += 1
            continue
        steps = split_sentences(g["gen_text"])
        if not steps:
            ladder["empty_trace"] += 1
            continue
        entity = steps[0].split()[0]
        gv = validate_continuation(inst["question"], [], None, g["gen_text"],
                                   inst["target"], entity)
        if gv["class"] != "valid_rederivation":
            ladder["gold_not_validator_valid"] += 1
            continue
        ladder["gold_validated_cohort"] += 1
        rows.append({"problem_id": g["id"], "question": inst["question"],
                     "target": inst["target"], "entity": entity,
                     "gold_text": g["gen_text"], "steps": steps})
    return rows, ladder


# -------------------------------------------------------------------- stages

def out_paths(out_dir):
    return {
        "candidate_pool": os.path.join(out_dir, "candidate_pool.jsonl"),
        "eligibility": os.path.join(out_dir, "eligibility_audit.jsonl"),
        "truth_audit": os.path.join(out_dir, "truth_audit.jsonl"),
        "availability": os.path.join(out_dir, "availability.json"),
        "manifest": os.path.join(out_dir, "manifest.jsonl"),
        "raw": os.path.join(out_dir, "raw_generations.jsonl"),
        "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
        "summary": os.path.join(out_dir, "summary_tables.json"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
        "report": os.path.join(out_dir, "EXPE_REPORT.md"),
        "tests": os.path.join(out_dir, "EXPE_TESTS_PASSED"),
        "complete": os.path.join(out_dir, "RUN_COMPLETE"),
    }


def ensure_out_dir(out_dir, overwrite=False):
    if os.path.exists(out_paths(out_dir)["complete"]) and not overwrite:
        raise SystemExit(f"{out_dir} is complete; refusing to modify without --overwrite")
    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def arms_for_position(args, position):
    """Arm subset generated at a position (PLAN2 v1.1 section C.5: all six arms
    at mid; REFUTING_d1 + BASELINE_FILLER additionally at early/late). Parsed
    from --position-arms 'early:REFUTING_d1+BASELINE_FILLER,...'; positions not
    named there run all six arms. Eligibility is NOT relaxed: a problem still
    must admit all six arms under one shared F at the position (paired-
    availability constraint held constant across positions)."""
    spec = getattr(args, "position_arms", None) or ""
    mapping = {}
    for part in [x for x in spec.split(",") if x.strip()]:
        pos, arms = part.split(":", 1)
        arm_list = tuple(a.strip() for a in arms.split("+") if a.strip())
        for a in arm_list:
            if a not in ARMS:
                raise SystemExit(f"--position-arms names unknown arm {a}")
        mapping[pos.strip()] = arm_list
    return mapping.get(position, ARMS)


def enforce_prereg_gate(args):
    if args.target > SMOKE_MAX_TARGET and not args.prereg_evidence:
        raise SystemExit(
            f"REFUSING: --target {args.target} exceeds the smoke cap "
            f"({SMOKE_MAX_TARGET}/arm). The EXPE full run is gated on the PLAN2 "
            "external pre-registration timestamp (PLAN2 section 9.10). Re-run with "
            "--prereg-evidence '<public commit hash / OSF url>' once it exists.")


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
        "experiment": "EXPE_EVIDENCE_MOVER",
        "created_at": now_iso(),
        "model": args.model,
        "model_revision": model_revision or args.model_revision,
        "model_revision_pinned": bool(revision_pinned),
        "model_revision_error": revision_error,
        "backend": args.backend,
        "seed": args.seed,
        "target_problems_per_position": args.target,
        "positions": list(args.positions),
        "smoke_cap": SMOKE_MAX_TARGET,
        "prereg_evidence": args.prereg_evidence or None,
        "cohort_source": getattr(args, "cohort", "legacy"),
        "cohort_path": (EXPANDED_COHORT if getattr(args, "cohort", "legacy") == "expanded"
                        else GOLD_ROLLOUTS),
        "position_arms": {p: list(arms_for_position(args, p)) for p in args.positions},
        "arms": list(ARMS),
        "designed_d": {k: v for k, v in DESIGNED_D.items()},
        "decoding": {"do_sample": False, "temperature": 0.0, "num_return_sequences": 1,
                     "max_new_tokens": args.max_new_tokens},
        "prompt": {"instruction": INSTR, "fewshot": FEWSHOT, "sha256": prompt_hash()},
        "protocol_disclosure_mod11": (
            "The own-trace was generated under the unaugmented question and all arms "
            "replay it under an augmented question -- equally off-policy across arms; "
            "the re-solve/re-validate rate on augmented questions is reported per arm "
            "(MC-6). EXPE<->EXPB comparisons are cross-protocol."),
        "python": sys.version,
    }
    write_json(out_paths(out_dir)["metadata"], meta)
    return meta


def run_id_for(problem_id, arm, position, seed):
    return sha_row(["EXPE", problem_id, arm, position, seed], 24)


def prepare(args):
    enforce_prereg_gate(args)
    ensure_out_dir(args.out_dir, overwrite=args.overwrite)
    paths = out_paths(args.out_dir)
    write_metadata(args.out_dir, args)

    cohort, ladder = load_gold_cohort(getattr(args, "cohort", "legacy"))
    eligible = []          # one row per (problem, position) with all arms
    arm_union = Counter()  # problems where arm is available under SOME F
    per_pos_ladder = {p: Counter() for p in args.positions}

    for row in cohort:
        pts = injection_points(row["steps"], row["entity"])
        if pts is None:
            ladder["too_few_injection_points"] += 1
            append_jsonl(paths["eligibility"], {
                "stage": "injection_points", "problem_id": row["problem_id"],
                "eligible": False, "reason": "too_few_intermediate_entity_steps"})
            continue
        for position in args.positions:
            si = pts[position]
            built, diags = build_problem_arms(
                row["question"], row["steps"], row["entity"], row["target"], si,
                row["gold_text"])
            lp = per_pos_ladder[position]
            if diags["n_falsehood_candidates"] == 0:
                lp["no_falsehood_candidate"] += 1
                append_jsonl(paths["eligibility"], {
                    "stage": "falsehood_availability", "problem_id": row["problem_id"],
                    "injection_position": position, "eligible": False,
                    "reason": "no_cwa_false_unrefuted_category"})
                continue
            for arm in diags["arm_available_under_some_F"]:
                arm_union[f"{position}|{arm}"] += 1
            if built is None:
                lp["not_all_arms_available"] += 1
                missing = [a for a in ARMS
                           if a not in diags["arm_available_under_some_F"]]
                append_jsonl(paths["eligibility"], {
                    "stage": "paired_availability", "problem_id": row["problem_id"],
                    "injection_position": position, "eligible": False,
                    "reason": "no_single_F_admits_all_arms",
                    "arms_never_available": missing,
                    "n_falsehood_candidates": diags["n_falsehood_candidates"],
                    "audit_failures": diags["audit_failures"][:6]})
                continue
            lp["admits_all_arms"] += 1
            lp[f"admits_all_arms_{built['vocab_tier']}"] += 1
            if built["poisoning_measurable"]:
                lp["admits_all_arms_measurable_poisoning"] += 1
            eligible.append({
                "problem_id": row["problem_id"], "question": row["question"],
                "target": row["target"], "entity": row["entity"],
                "gold_text": row["gold_text"], "steps": row["steps"],
                "injection_position": position, "sent_idx": si, **built})
            append_jsonl(paths["eligibility"], {
                "stage": "paired_availability", "problem_id": row["problem_id"],
                "injection_position": position, "eligible": True,
                "reason": "all_arms_available",
                "fcat": built["fcat"], "vocab_tier": built["vocab_tier"],
                "poisoning_measurable": built["poisoning_measurable"]})

    availability = {
        "cohort_ladder": dict(ladder),
        "per_position": {p: dict(c) for p, c in per_pos_ladder.items()},
        "arm_available_under_some_F": dict(arm_union),
        "paired_availability_constraint": {
            p: per_pos_ladder[p].get("admits_all_arms", 0) for p in args.positions},
    }
    write_json(paths["availability"], availability)

    rng = random.Random(args.seed)
    selected = []
    for position in args.positions:
        rows = sorted([r for r in eligible if r["injection_position"] == position],
                      key=lambda r: r["problem_id"])
        rng.shuffle(rows)
        selected.extend(rows[: args.target])

    seen = set()
    with open(paths["manifest"], "w") as mh, open(paths["truth_audit"], "w") as th:
        for rec in selected:
            prefill = " " + " ".join(rec["steps"][: rec["sent_idx"]] + [rec["falsehood_statement"]])
            for arm in arms_for_position(args, rec["injection_position"]):
                spec = rec["specs"][arm]
                audit = rec["audits"][arm]
                rid = run_id_for(rec["problem_id"], arm, rec["injection_position"], args.seed)
                if rid in seen:
                    raise RuntimeError(f"duplicate run_id {rid}")
                seen.add(rid)
                if not audit["fully_audited"]:
                    raise RuntimeError(f"fail-closed: unaudited row selected {rid}")
                aug_q = audit["aug_question"]
                mh.write(json.dumps({
                    "run_id": rid,
                    "problem_id": rec["problem_id"],
                    "arm": arm,
                    "condition": arm,
                    "injection_position": rec["injection_position"],
                    "seed": args.seed,
                    "model": args.model,
                    "entity": rec["entity"],
                    "target": rec["target"],
                    "original_question": rec["question"],
                    "question": aug_q,
                    "added_rule_sentence": spec["rule_sentence"],
                    "added_rule_body": spec["body"],
                    "added_rule_head": spec["head"],
                    "added_rule_insert_index": audit["insert_index"],
                    "added_rule_char_offset": audit["char_offset"],
                    "added_rule_char_len": len(spec["rule_sentence"]),
                    "antecedent_hop": spec["antecedent_hop"],
                    "antecedent_in_gold_suffix": spec["antecedent_in_gold_suffix"],
                    "added_rule_body_source": spec["body_source"],
                    "added_rule_head_source": spec["head_source"],
                    "vocab_tier": spec["vocab_tier"],
                    "designed_d": DESIGNED_D[arm],
                    "measured_d_augmented": audit["measured_d_augmented"],
                    "measured_d_original": audit["measured_d_original"],
                    "falsehood_statement": rec["falsehood_statement"],
                    "falsehood_category": rec["fcat"],
                    "injected_statement": rec["falsehood_statement"],
                    "injected_statement_truth_status": "false",
                    "falsehood_status_original": "cwa_false",
                    "falsehood_status_augmented": (
                        "provably_false" if DESIGNED_D[arm] is not None else "cwa_false"),
                    "poisoning_measurable": rec["poisoning_measurable"],
                    "falsehood_mentions_in_question": len(re.findall(
                        rf"\b{rec['fcat']}(?:es)?\b", aug_q)),
                    "sent_idx": rec["sent_idx"],
                    "correct_step": rec["steps"][rec["sent_idx"]],
                    "prefix_steps": rec["steps"][: rec["sent_idx"]],
                    "gold_text": rec["gold_text"],
                    "assistant_prefill_text": prefill,
                    "gold_revalidates_under_augmentation": audit["checks"][
                        "gold_revalidates_under_augmentation"],
                    "prompt_sha256": prompt_hash(),
                    "original_proof_validated": True,
                    "created_at": now_iso(),
                }, sort_keys=True) + "\n")
                th.write(json.dumps({
                    "run_id": rid, "problem_id": rec["problem_id"], "arm": arm,
                    "injection_position": rec["injection_position"],
                    "seed": args.seed,
                    "falsehood_statement": rec["falsehood_statement"],
                    "added_rule_sentence": spec["rule_sentence"],
                    "fully_audited": audit["fully_audited"],
                    "checks": audit["checks"],
                    "measured_d_original": audit["measured_d_original"],
                    "measured_d_augmented": audit["measured_d_augmented"],
                    "gold_revalidation_class": audit["gold_revalidation_class"],
                    "created_at": now_iso(),
                }, sort_keys=True) + "\n")

    with open(paths["candidate_pool"], "w") as fh:
        for rec in eligible:
            slim = {k: rec[k] for k in ("problem_id", "injection_position", "sent_idx",
                                        "fcat", "falsehood_statement",
                                        "poisoning_measurable")}
            slim["arm_rules"] = {a: rec["specs"][a]["rule_sentence"] for a in ARMS}
            fh.write(json.dumps(slim, sort_keys=True) + "\n")

    print(json.dumps({
        "availability": availability["paired_availability_constraint"],
        "cohort": dict(ladder),
        "eligible_rows": len(eligible),
        "selected_problems": len(selected),
        "manifest_rows": len(seen),
    }, indent=2))


# ------------------------------------------------------------------ generate

def generate_vllm(args, todo, paths):
    sys.path.insert(0, TOOLING)
    import vllm_gen
    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise RuntimeError(f"could not pin model revision: {err}")
    write_metadata(args.out_dir, args, revision, pinned, err)
    tok = vllm_gen.get_tokenizer(args.model, revision)
    rows = []
    for m in todo:
        ids = vllm_gen.render_prompt_ids(tok, m["question"], m["target"],
                                         m["assistant_prefill_text"], INSTR, FEWSHOT)
        rows.append({"prompt_token_ids": ids})
    outs = vllm_gen.generate_continuations(args.model, rows,
                                           max_new_tokens=args.max_new_tokens,
                                           revision=revision)
    eos = set(vllm_gen.eos_token_ids(args.model, tok, revision))
    for m, r, o in zip(todo, rows, outs):
        toks = list(o["token_ids"])
        while toks and toks[-1] in eos:
            toks = toks[:-1]
        append_jsonl(paths["raw"], {
            "run_id": m["run_id"], "problem_id": m["problem_id"], "arm": m["arm"],
            "condition": m["arm"], "injection_position": m["injection_position"],
            "seed": m["seed"], "model": args.model, "model_revision": revision,
            "backend": "vllm", "failed_generation": False,
            "continuation": o["text"], "finish_reason": o["finish_reason"],
            "prompt_token_count": len(r["prompt_token_ids"]),
            "continuation_token_count": len(toks),
            "added_rule_token_count": len(tok(m["added_rule_sentence"],
                                              add_special_tokens=False)["input_ids"]),
            "decoding": {"do_sample": False, "temperature": 0.0,
                         "max_new_tokens": args.max_new_tokens},
            "created_at": now_iso(),
        })


def generate_hf(args, todo, paths):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    revision, pinned, err = resolve_model_revision(args.model, args.model_revision)
    if not pinned and not args.allow_unpinned_model:
        raise RuntimeError(f"could not pin model revision: {err}")
    write_metadata(args.out_dir, args, revision, pinned, err)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, dtype=torch.bfloat16, device_map=args.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    for i, m in enumerate(todo):
        try:
            msgs = [{"role": "user", "content": prompt_user_text(m["question"], m["target"])}]
            out = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            ids = out["input_ids"] if not torch.is_tensor(out) else out
            pre = tok(m["assistant_prefill_text"], return_tensors="pt",
                      add_special_tokens=False)["input_ids"]
            ids = torch.cat([ids, pre], dim=1)
            with torch.no_grad():
                gen = model.generate(ids.to(model.device),
                                     attention_mask=torch.ones_like(ids).to(model.device),
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            cont = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            append_jsonl(paths["raw"], {
                "run_id": m["run_id"], "problem_id": m["problem_id"], "arm": m["arm"],
                "condition": m["arm"], "injection_position": m["injection_position"],
                "seed": m["seed"], "model": args.model, "model_revision": revision,
                "backend": "hf", "failed_generation": False, "continuation": cont,
                "prompt_token_count": int(ids.shape[1]),
                "continuation_token_count": int(gen.shape[1] - ids.shape[1]),
                "added_rule_token_count": len(tok(m["added_rule_sentence"],
                                                  add_special_tokens=False)["input_ids"]),
                "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                "created_at": now_iso(),
            })
        except Exception as e:
            append_jsonl(paths["raw"], {
                "run_id": m["run_id"], "problem_id": m["problem_id"], "arm": m["arm"],
                "condition": m["arm"], "injection_position": m["injection_position"],
                "seed": m["seed"], "model": args.model, "model_revision": revision,
                "backend": "hf", "failed_generation": True, "error": repr(e),
                "traceback": traceback.format_exc(), "created_at": now_iso(),
            })
        if (i + 1) % 25 == 0 or i < 3:
            print(f"[EXPE hf generate] {i + 1}/{len(todo)}", flush=True)


def generate(args):
    enforce_prereg_gate(args)
    paths = out_paths(args.out_dir)
    if not os.path.exists(paths["manifest"]):
        prepare(args)
    manifest = read_jsonl(paths["manifest"])
    existing = {r["run_id"] for r in read_jsonl(paths["raw"])}
    todo = [m for m in manifest if m["run_id"] not in existing]
    print(f"[EXPE generate] backend={args.backend} todo={len(todo)}/{len(manifest)}", flush=True)
    if not todo:
        return
    if args.backend == "vllm":
        generate_vllm(args, todo, paths)
    else:
        generate_hf(args, todo, paths)
    print(f"DONE generate rows={len(manifest)}", flush=True)


# ------------------------------------------------------------------ validate

def validate(args):
    paths = out_paths(args.out_dir)
    manifest = {r["run_id"]: r for r in read_jsonl(paths["manifest"])}
    raw = {r["run_id"]: r for r in read_jsonl(paths["raw"])}
    with open(paths["validated"], "w") as out:
        for rid, m in manifest.items():
            r = raw.get(rid)
            base = {k: m[k] for k in (
                "run_id", "problem_id", "arm", "condition", "injection_position",
                "seed", "model", "sent_idx", "designed_d", "measured_d_augmented",
                "falsehood_statement", "falsehood_category", "added_rule_sentence",
                "poisoning_measurable", "antecedent_in_gold_suffix",
                "gold_revalidates_under_augmentation", "original_proof_validated")}
            if r is None:
                rec = {**base, "class": "generation_missing", "failed_generation": True,
                       "verbalized_doubt": False, "valid_recovery": False,
                       "injection_dependent": None}
            elif r.get("failed_generation"):
                rec = {**base, "class": "generation_failed", "failed_generation": True,
                       "error": r.get("error"), "verbalized_doubt": False,
                       "valid_recovery": False, "injection_dependent": None}
            else:
                v = validate_continuation(
                    m["question"], m["prefix_steps"], m["falsehood_statement"],
                    r.get("continuation", ""), m["target"], m["entity"])
                # Secondary behavioral DV: the continuation explicitly states the
                # complement of F (derivational refutation without any doubt
                # marker is common -- the lexical regex misses it entirely).
                comp = (m["entity"], ("not_cat", m["falsehood_category"]))
                stated_comp = any(
                    parse_fact(strip_marker(s), m["entity"]) == comp
                    for s in split_sentences(r.get("continuation", "")))
                rec = {
                    **base, **v, "failed_generation": False,
                    "valid_recovery": v["class"] == "valid_rederivation",
                    "stated_complement_of_falsehood": stated_comp,
                    "verbalized_doubt": bool(v.get("acknowledged")),
                    "injection_dependent": (
                        (v["class"] == "poisoned") if m["poisoning_measurable"] else None),
                    "continuation": r.get("continuation", ""),
                    "continuation_token_count": r.get("continuation_token_count"),
                    "prompt_token_count": r.get("prompt_token_count"),
                    "added_rule_token_count": r.get("added_rule_token_count"),
                    "backend": r.get("backend"),
                }
            out.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {paths['validated']}")


# ------------------------------------------------------------------- stats

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
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def metric_value(row, metric):
    cls = row.get("class")
    if metric == "valid":
        return int(cls == "valid_rederivation")
    if metric == "doubt":
        return int(bool(row.get("verbalized_doubt")))
    if metric == "inj_dep":
        v = row.get("injection_dependent")
        return None if v is None else int(bool(v))
    if metric == "inj_dep_raw":
        return int(cls == "poisoned")
    if metric == "parroted":
        return int(cls == "parroted")
    if metric == "derailed":
        return int(cls == "derailed")
    if metric == "unparsed":
        return int(cls == "unparsed")
    if metric == "generation_failed":
        return int(cls in ("generation_failed", "generation_missing"))
    if metric == "resolve":
        return int(bool(row.get("gold_revalidates_under_augmentation")))
    if metric == "stated_refutation":
        return int(bool(row.get("stated_complement_of_falsehood")))
    raise KeyError(metric)


METRICS = ("valid", "doubt", "stated_refutation", "inj_dep", "inj_dep_raw",
           "parroted", "derailed", "unparsed", "generation_failed", "resolve")


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
        num = den = 0
        for pid in [rng.choice(ids) for _ in ids]:
            for r in by_id[pid]:
                v = metric_value(r, metric)
                if v is None:
                    continue
                num += v
                den += 1
        if den:
            vals.append(num / den)
    if not vals:
        return [None, None]
    return [round(percentile(vals, 0.025), 4), round(percentile(vals, 0.975), 4)]


def summarize_cell(rows, seed):
    out = {"n": len(rows), "problem_n": len({r["problem_id"] for r in rows})}
    for metric in METRICS:
        vals = [metric_value(r, metric) for r in rows]
        vals = [v for v in vals if v is not None]
        n, k = len(vals), sum(vals)
        out[metric] = {"n": n, "count": k, "rate": round(k / n, 4) if n else None,
                       "wilson95": wilson(k, n),
                       "problem_cluster_bootstrap95": cluster_boot_rate(rows, metric, seed)}
    return out


def paired_rows(rows, arm_a, arm_b):
    by_key = defaultdict(dict)
    for r in rows:
        if r["arm"] in (arm_a, arm_b):
            by_key[(r["problem_id"], r["injection_position"])][r["arm"]] = r
    return [(pid, d[arm_a], d[arm_b]) for (pid, _), d in by_key.items()
            if arm_a in d and arm_b in d]


def mcnemar_pvalue(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def paired_contrast(rows, arm_a, arm_b, metric, seed=0, n_boot=1000):
    pairs = [(pid, a, b) for pid, a, b in paired_rows(rows, arm_a, arm_b)
             if metric_value(a, metric) is not None and metric_value(b, metric) is not None]
    if not pairs:
        return {"n_pairs": 0}
    diffs = [metric_value(a, metric) - metric_value(b, metric) for _, a, b in pairs]
    ids = sorted({pid for pid, _, _ in pairs})
    by_id = defaultdict(list)
    for pid, a, b in pairs:
        by_id[pid].append((a, b))
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        nums = []
        for pid in [rng.choice(ids) for _ in ids]:
            nums.extend(metric_value(a, metric) - metric_value(b, metric)
                        for a, b in by_id[pid])
        boots.append(sum(nums) / len(nums) if nums else 0.0)
    a1b0 = sum(metric_value(a, metric) == 1 and metric_value(b, metric) == 0
               for _, a, b in pairs)
    a0b1 = sum(metric_value(a, metric) == 0 and metric_value(b, metric) == 1
               for _, a, b in pairs)
    return {"n_pairs": len(pairs), "problem_n": len(ids),
            "diff_a_minus_b": round(sum(diffs) / len(diffs), 4),
            "ci95_problem_cluster_bootstrap": [round(percentile(boots, 0.025), 4),
                                               round(percentile(boots, 0.975), 4)],
            "mcnemar_a1_b0": a1b0, "mcnemar_a0_b1": a0b1,
            "mcnemar_exact_p": round(mcnemar_pvalue(a1b0, a0b1), 6)}


def prompt_sanity(manifest, position_arms=None):
    """Trace bytes identical across arms; same insertion slot; rule length spread.

    position_arms: {position: expected arm tuple}; positions absent from the
    map expect all six arms (matches arms_for_position / --position-arms)."""
    position_arms = position_arms or {}
    by_key = defaultdict(dict)
    for m in manifest:
        by_key[(m["problem_id"], m["injection_position"])][m["arm"]] = m
    ok = True
    failures, deltas = [], []
    for key, d in by_key.items():
        expected = set(position_arms.get(key[1], ARMS))
        if set(d) != expected:
            ok = False
            failures.append({"key": list(key), "reason": "missing_arm",
                             "arms": sorted(d), "expected": sorted(expected)})
            continue
        prefills = {m["assistant_prefill_text"] for m in d.values()}
        orig_qs = {m["original_question"] for m in d.values()}
        slots = {m["added_rule_insert_index"] for m in d.values()}
        if len(prefills) != 1:
            ok = False
            failures.append({"key": list(key), "reason": "trace_bytes_differ"})
        if len(orig_qs) != 1:
            ok = False
            failures.append({"key": list(key), "reason": "base_question_differs"})
        if len(slots) != 1:
            ok = False
            failures.append({"key": list(key), "reason": "insertion_slot_differs"})
        lens = [m["added_rule_char_len"] for m in d.values()]
        deltas.append(max(lens) - min(lens))
        # frequency matching: refuting + freq arms mention fcat once more than
        # template + filler arms (checked on whichever arms exist at this key).
        f_ment = {a: d[a]["falsehood_mentions_in_question"] for a in d}
        plus = [a for a in d if a in (ARM_REFUTING_D1, ARM_REFUTING_D2,
                                      ARM_REFUTING_D3, ARM_FREQ)]
        zero = [a for a in d if a in (ARM_TEMPLATE, ARM_FILLER)]
        f_ok = True
        if zero:
            base = f_ment[zero[0]]
            f_ok = (all(f_ment[a] == base for a in zero)
                    and all(f_ment[a] == base + 1 for a in plus))
        elif plus:
            f_ok = len({f_ment[a] for a in plus}) == 1
        if not f_ok:
            ok = False
            failures.append({"key": list(key), "reason": "falsehood_mention_mismatch",
                             "mentions": f_ment})
    return {"trace_and_slot_identical_across_arms": ok,
            "failures": failures[:20],
            "added_rule_char_len_spread": {
                "mean": round(sum(deltas) / len(deltas), 2) if deltas else None,
                "max": max(deltas) if deltas else None}}


def summarize(args):
    paths = out_paths(args.out_dir)
    rows = read_jsonl(paths["validated"])
    manifest = read_jsonl(paths["manifest"])
    truth = read_jsonl(paths["truth_audit"])
    availability = (json.load(open(paths["availability"]))
                    if os.path.exists(paths["availability"]) else {})

    by_arm = {arm: summarize_cell([r for r in rows if r["arm"] == arm], args.seed)
              for arm in ARMS}
    by_arm_pos = {}
    for arm in ARMS:
        for pos in args.positions:
            sub = [r for r in rows if r["arm"] == arm and r["injection_position"] == pos]
            if sub:
                by_arm_pos[f"{arm}|{pos}"] = summarize_cell(sub, args.seed)

    contrasts = {
        "H5a_doubt_REFUTING_d1_vs_FREQ": paired_contrast(
            rows, ARM_REFUTING_D1, ARM_FREQ, "doubt", args.seed),
        "H5b_doubt_REFUTING_d1_vs_TEMPLATE": paired_contrast(
            rows, ARM_REFUTING_D1, ARM_TEMPLATE, "doubt", args.seed),
        "H6_injdep_REFUTING_d1_vs_FILLER": paired_contrast(
            rows, ARM_REFUTING_D1, ARM_FILLER, "inj_dep", args.seed),
        "explor_doubt_d1_vs_d3": paired_contrast(
            rows, ARM_REFUTING_D1, ARM_REFUTING_D3, "doubt", args.seed),
        "explor_valid_REFUTING_d1_vs_FILLER": paired_contrast(
            rows, ARM_REFUTING_D1, ARM_FILLER, "valid", args.seed),
        "explor_statedref_REFUTING_d1_vs_FREQ": paired_contrast(
            rows, ARM_REFUTING_D1, ARM_FREQ, "stated_refutation", args.seed),
        "explor_statedref_d1_vs_d3": paired_contrast(
            rows, ARM_REFUTING_D1, ARM_REFUTING_D3, "stated_refutation", args.seed),
    }

    audit_rates = {}
    for arm in ARMS:
        arows = [t for t in truth if t["arm"] == arm]
        audit_rates[arm] = {
            "n": len(arows),
            "fully_audited_rate": (round(sum(t["fully_audited"] for t in arows)
                                         / len(arows), 4) if arows else None),
            "gold_revalidation_rate": (round(sum(
                t["gold_revalidation_class"] == "valid_rederivation" for t in arows)
                / len(arows), 4) if arows else None),
            "measured_d_values": dict(Counter(
                str(t["measured_d_augmented"]) for t in arows)),
        }

    resolve_per_arm = {arm: by_arm[arm]["resolve"]["rate"] for arm in ARMS}
    sanity = prompt_sanity(
        manifest, {p: arms_for_position(args, p) for p in args.positions})
    run_ids = [m["run_id"] for m in manifest]
    vocab = {
        "problem_tier_distribution": dict(Counter(
            m["vocab_tier"] for m in manifest if m["arm"] == ARM_REFUTING_D1)),
        "noun_sources_by_arm": {arm: dict(Counter(
            f"{m['added_rule_body_source']}|{m['added_rule_head_source']}"
            for m in manifest if m["arm"] == arm)) for arm in ARMS},
    }

    out = {
        "created_at": now_iso(),
        "sample_size": {
            "target_problems_per_position": args.target,
            "manifest_rows": len(manifest),
            "validated_rows": len(rows),
            "problems": len({m["problem_id"] for m in manifest}),
            "rows_per_arm": dict(Counter(m["arm"] for m in manifest)),
        },
        "availability": availability,
        "metrics": {"pooled_by_arm": by_arm, "by_arm_position": by_arm_pos},
        "paired_contrasts": contrasts,
        "audit_pass_rates": audit_rates,
        "resolve_rate_per_arm_mod11": resolve_per_arm,
        "vocabulary_tiers": vocab,
        "sanity_checks": sanity,
        "integrity": {
            "unique_run_ids": len(run_ids) == len(set(run_ids)),
            "every_selected_row_fully_audited": bool(truth) and all(
                t["fully_audited"] for t in truth),
            "all_falsehoods_audited_false": bool(truth) and all(
                t["checks"]["falsehood_unentailed_original"]
                and t["checks"]["falsehood_complement_unentailed_original"]
                for t in truth),
            "all_gold_traces_revalidate": bool(truth) and all(
                t["gold_revalidation_class"] == "valid_rederivation" for t in truth),
            "no_target_shortcut_all_rows": bool(truth) and all(
                t["checks"]["no_target_shortcut"] for t in truth),
            "failed_generations_logged": os.path.exists(paths["raw"]),
        },
        "multiple_comparisons": (
            "H1'/H5'/H6 belong to the PLAN2 v1.1 single confirmatory family "
            "(m=6, Holm; primary DV = strict stated-complement). Contrasts in this "
            "file are runner-level descriptives; the registered adjudications live "
            "in hypothesis_verdicts.json (expe_full_analysis.py). Doubt here is the "
            "LEXICAL regex; the 32B judge pass is run separately."),
        "exclusions": ("Fully paired problems only: all six arms must be constructible "
                       "and fail-closed audited for one shared falsehood F before any "
                       "generation. Failed/unparsed generations retained in denominators."),
        "reproducibility": (json.load(open(paths["metadata"]))
                            if os.path.exists(paths["metadata"]) else {}),
    }
    write_json(paths["summary"], out)
    print(json.dumps({"sample_size": out["sample_size"],
                      "availability": availability.get("paired_availability_constraint"),
                      "resolve_rate_per_arm": resolve_per_arm}, indent=2))


def report(args):
    paths = out_paths(args.out_dir)
    s = json.load(open(paths["summary"]))
    avail = s.get("availability", {})
    meta = s.get("reproducibility", {})
    scale = ("FULL" if (meta.get("target_problems_per_position") or 0) > SMOKE_MAX_TARGET
             else "SMOKE")
    lines = [
        f"# EXPE Evidence-Mover Report ({scale})",
        "",
        f"Generated: {now_iso()}",
        "",
        f"- Cohort source: {meta.get('cohort_source', 'legacy')} "
        f"({meta.get('cohort_path', GOLD_ROLLOUTS)})",
        "  (runner change for the full run: the gold cohort is read from the",
        "  PLAN2 v1.1 C.1 expansion, results/EXPE_GOLD_EXPANSION/cohort.jsonl,",
        "  via --cohort expanded; --cohort legacy preserves the original",
        "  pilot.jsonl + results/gold/rollouts.jsonl path.)",
        f"- Position-arm grid (v1.1 C.5): {json.dumps(meta.get('position_arms', {}))}",
        f"- Pre-registration evidence: {meta.get('prereg_evidence')}",
        "",
        "Protocol disclosure (MOD-11): the own-trace was generated under the",
        "unaugmented question and all arms replay it under an augmented question --",
        "equally off-policy across arms. EXPE<->EXPB comparisons are cross-protocol.",
        "Doubt below is the LEXICAL regex DV; the judge pass is not run at smoke scale.",
        "",
        "## Availability (eligibility ladder)",
        "",
        f"- Cohort ladder: {json.dumps(avail.get('cohort_ladder', {}))}",
        f"- Per-position ladders: {json.dumps(avail.get('per_position', {}))}",
        f"- PAIRED-AVAILABILITY (problems admitting ALL six arms under one F): "
        f"{json.dumps(avail.get('paired_availability_constraint', {}))}",
        f"- Arm available under some F: {json.dumps(avail.get('arm_available_under_some_F', {}))}",
        f"- Vocabulary tiers of selected problems: "
        f"{json.dumps(s.get('vocabulary_tiers', {}).get('problem_tier_distribution', {}))}",
        "",
        "Vocabulary-tier disclosure: the legacy cohort's unentailed-category pools",
        "are tiny (95/130 injectable worlds have zero), so control-arm nonce nouns",
        "use a preference ladder -- A: in-vocab distinct, B: in-vocab reused across",
        "arms, C: fresh nonce nouns for TEMPLATE-head/FILLER slots only. The FREQ",
        "antecedent is always in-question-vocabulary (audited).",
        "",
        "## Per-arm rates (pooled)",
        "",
        "| arm | n | valid | doubt(lex) | stated-not-F | inj-dep | parroted | derailed | unparsed | re-solve |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        c = s["metrics"]["pooled_by_arm"].get(arm, {})
        def r(m):
            v = c.get(m, {})
            return f"{v.get('rate')} (n={v.get('n')})" if v else "-"
        lines.append(f"| {arm} | {c.get('n')} | {r('valid')} | {r('doubt')} | "
                     f"{r('stated_refutation')} | "
                     f"{r('inj_dep')} | {r('parroted')} | {r('derailed')} | "
                     f"{r('unparsed')} | {r('resolve')} |")
    lines += ["", "## Paired contrasts (descriptive at smoke n)", ""]
    for name, c in s["paired_contrasts"].items():
        lines.append(f"- {name}: diff={c.get('diff_a_minus_b')}, "
                     f"CI={c.get('ci95_problem_cluster_bootstrap')}, "
                     f"discordant={c.get('mcnemar_a1_b0')}/{c.get('mcnemar_a0_b1')}, "
                     f"McNemar p={c.get('mcnemar_exact_p')} (n_pairs={c.get('n_pairs')})")
    lines += ["", "## Audit pass rates (fail-closed; selection requires 1.0)", ""]
    for arm, a in s["audit_pass_rates"].items():
        lines.append(f"- {arm}: fully_audited={a['fully_audited_rate']}, "
                     f"gold_revalidation={a['gold_revalidation_rate']}, "
                     f"measured_d={json.dumps(a['measured_d_values'])}")
    lines += ["", "## Sanity", "",
              f"- {json.dumps(s['sanity_checks'], sort_keys=True)[:1200]}",
              "",
              "## Integrity", ""]
    for k, v in s["integrity"].items():
        lines.append(f"- [{'x' if v else ' '}] {k}")
    if scale == "SMOKE":
        lines += ["", "## Limitations", "",
                  "- Smoke run: n <= 20/arm at mid; estimation only, no confirmatory claims.",
                  "- Lexical doubt regex is the DV here; PLAN2's primary doubt DV is the",
                  "  32B judge (doubt_judge_v2), to be run over these continuations later.",
                  "- 'stated-not-F' (continuation explicitly derives the complement of F)",
                  "  is a smoke-added secondary DV: refutation is frequently derivational",
                  "  with zero doubt markers, which the lexical regex cannot see. If kept",
                  "  for the full run it must be added to PLAN2 by timestamped addendum.",
                  "- vLLM backend: locked HF results are not row-level reproducible under",
                  "  vLLM (VLLM_PORT_REPORT.md); new suites run wholly under vLLM.",
                  ""]
    else:
        lines += ["", "## Notes (full run)", "",
                  "- Primary rejection DV (PLAN2 v1.1 section B): strict stated-complement",
                  "  ('stated-not-F' column). Judge explicit-rejection is the registered",
                  "  secondary (doubt_judge_v2 pass); verbalized doubt (lexical + judge)",
                  "  is descriptive only.",
                  "- Registered adjudications (H1'/H5'/H6) are computed by",
                  "  expe_full_analysis.py into hypothesis_verdicts.json /",
                  "  EXPE_FULL_REPORT.md; this file is the runner-level view.",
                  "- vLLM backend: locked HF results are not row-level reproducible under",
                  "  vLLM (VLLM_PORT_REPORT.md); new suites run wholly under vLLM.",
                  ""]
    text = "\n".join(lines)
    with open(paths["report"], "w") as fh:
        fh.write(text)
    print(text)


def test(args):
    import expe_tests
    expe_tests.main()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_paths(args.out_dir)["tests"], "w") as fh:
        fh.write(now_iso() + "\n")


def finalize(args):
    paths = out_paths(args.out_dir)
    need = ("manifest", "truth_audit", "raw", "validated", "summary", "report")
    missing = [k for k in need if not os.path.exists(paths[k])]
    if missing:
        raise SystemExit(f"missing required outputs: {missing}")
    with open(paths["complete"], "w") as fh:
        fh.write(now_iso() + "\n")
    for root, dirs, files in os.walk(args.out_dir):
        for name in dirs + files:
            p = os.path.join(root, name)
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(args.out_dir, os.stat(args.out_dir).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"finalized immutable result directory: {args.out_dir}")


def smoke(args):
    args.target = min(args.target, SMOKE_MAX_TARGET)
    args.overwrite = True
    test(args)
    prepare(args)
    generate(args)
    validate(args)
    rows = read_jsonl(out_paths(args.out_dir)["validated"])
    failed = [r for r in rows if r.get("failed_generation")]
    if failed:
        raise RuntimeError(f"smoke generation failures: {len(failed)}/{len(rows)}")
    summarize(args)
    report(args)


def run_all(args):
    enforce_prereg_gate(args)
    test(args)
    prepare(args)
    generate(args)
    validate(args)
    summarize(args)
    report(args)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EXPE evidence-mover runner")
    p.add_argument("stage", choices=["test", "prepare", "generate", "validate",
                                     "summarize", "report", "finalize", "smoke", "run-all"])
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default="auto")
    p.add_argument("--allow-unpinned-model", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--target", type=int, default=SMOKE_MAX_TARGET,
                   help="problems per position (= n per arm per position)")
    p.add_argument("--positions", default="mid",
                   help="comma-separated subset of early,mid,late")
    p.add_argument("--cohort", choices=["expanded", "legacy"], default="legacy",
                   help="gold cohort source: 'expanded' = PLAN2 v1.1 C.1 cohort "
                        "(results/EXPE_GOLD_EXPANSION/cohort.jsonl, full-run "
                        "choice); 'legacy' (default) = pilot.jsonl + "
                        "results/gold/rollouts.jsonl, byte-identical to the "
                        "smoke-era behavior")
    p.add_argument("--position-arms", default=None,
                   help="optional per-position arm subset, e.g. "
                        "'early:REFUTING_d1+BASELINE_FILLER,late:REFUTING_d1+"
                        "BASELINE_FILLER' (PLAN2 v1.1 C.5); unnamed positions "
                        "run all six arms")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--prereg-evidence", default=None,
                   help="public pre-registration timestamp evidence; required for "
                        f"--target > {SMOKE_MAX_TARGET}")
    args = p.parse_args(argv)
    args.positions = [x.strip() for x in args.positions.split(",") if x.strip()]
    for pos in args.positions:
        if pos not in ("early", "mid", "late"):
            p.error(f"bad position {pos}")
    return args


def main(argv=None):
    args = parse_args(argv)
    stage = {"test": test, "prepare": prepare, "generate": generate,
             "validate": validate, "summarize": summarize, "report": report,
             "finalize": finalize, "smoke": smoke, "run-all": run_all}[args.stage]
    stage(args)


if __name__ == "__main__":
    main()
