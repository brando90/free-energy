#!/usr/bin/env python3
"""Characterize the NATURAL error distribution of Qwen2.5-7B on PrOntoQA gold rollouts.

Zero-GPU, read-only over results/. Answers the reviewer objection: "maybe your
insights apply to how YOU inject falsehoods but not how MODELS actually err."

Sources of failed/invalid GOLD (unperturbed) rollouts, all with full text stored:
  legacy : results/gold/rollouts.jsonl + data/pilot.jsonl              (219 rollouts)
  EXPA   : results/EXPA_GLOBAL_EXPANSION/{eligibility_audit,raw_generations}.jsonl
           gold_filter failures: gold_not_solved + gold_not_validator_valid
  EXPE   : results/EXPE_GOLD_EXPANSION/{validated,screened_candidates,raw_generations}.jsonl
  EXPD   : results/EXPD_MATCHED_GRADIENT/{gold_cohort,raw_generations}.jsonl + worlds/worlds.json
           (synthetic in-grammar worlds; reported as a separate segment)

Semantics come from src/ (imports are read-only): validator.py parse/closure/derivable,
expc_polarity_control.py shortest_rule_distance / opposite_pred / predicate_family.

Per failed rollout we walk the model's own sentences against the TRUE closure:
  valid_fact                : derivable from true state (validator semantics)
  echo_of_error             : re-assertion of an earlier natural error atom
  derived_from_error        : derivable ONLY via earlier error atom(s) (poisoned logic,
                              with the natural error as the corrupted atom)
  derived_via_fabricated_rule: licensed only by a rule the model invented
  fresh_error               : underivable even given all earlier errors  <- NATURAL ERROR
  rule_true / rule_fabricated: rule restatements (fabricated = not a question premise)
  off_entity_fact           : parses as a fact about a different subject
  unparsed_entity_fact      : entity-subject sentence outside the validator grammar
                              (compound-fact diagnostic attempted)
  other_unparsed            : commentary / disjunctive-rule restatements etc.

For each fresh natural error we record:
  (a) statement type   : category/attribute x affirmative/negated
  (b) truth status     : audit_truth_status semantics vs the original world
                         (false_contradicted / false_cwa_unentailed /
                          unknown_negated_cat_open_world / unknown_attr_undecidable)
  (c) usability        : design (can its atom feed the target under closure) and
                         realized (used downstream / echoed / inert)
  (d) refutation dist  : shortest_rule_distance of its complement from the VISIBLE
                         prefix state at the point of error
  (e) position         : sentence-index fraction, bucketed early/mid/late thirds
  (f) kind             : entity-fact vs rule vs off-entity vs unparsed

Only rollouts whose WORLD fully parses (no unparsed premises, parseable target) enter
the truth-based analysis; others (Composed-grammar worlds) are counted separately as
validator-blind, since "invalid" there conflates model error with grammar limits.

Outputs (improvement_plan/natural_errors/):
  natural_errors.jsonl    one row per fresh natural error (+ rule/off-entity errors)
  natural_rollouts.jsonl  one row per failed rollout (failure taxonomy + absorption)
  summary.json            all aggregate tables used in NATURAL_ERRORS_REPORT.md
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
sys.path.insert(0, os.path.join(EXP, "src"))

from validator import (parse_world, parse_fact, parse_rule, closure, derivable,
                       strip_marker, validate_continuation)
from expc_polarity_control import (direct_rule_map, shortest_rule_distance,
                                   opposite_pred, predicate_family,
                                   is_positive_pred, split_sentences)

OUT_DIR = os.path.join("/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/s7_rowcount", "out_baseline")
RESULTS = os.path.join(EXP, "results")
ENTITYISH = re.compile(r"^[A-Z]\w*\s+is\b")
FEWSHOT_ENTITIES = {"Sam", "Alex"}  # entities that appear in the shared few-shot prompt
# rules stated in the shared few-shot prompt (leakage check), rules_key format
FEWSHOT_RULES = {("yumpus", ("cat", "dumpus")), ("dumpus", ("cat", "tumpus")),
                 ("tumpus", ("adj", "bright", False)), ("gorpus", ("cat", "sterpus")),
                 ("sterpus", ("adj", "red", True)), ("borpus", ("cat", "gorpus"))}


def pred_token(pred):
    return pred[1]


# --------------------------------------------------------------------------- loading
def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_cohort():
    """Yield dicts: source, problem_id, question, target, entity, gen_text, solved."""
    rows, seen = [], set()

    # legacy pilot (219 gold rollouts)
    pilot = {r["id"]: r for r in read_jsonl(os.path.join(EXP, "data", "pilot.jsonl"))}
    for r in read_jsonl(os.path.join(RESULTS, "gold", "rollouts.jsonl")):
        inst = pilot[r["id"]]
        steps = split_sentences(r["gen_text"])
        f = parse_fact(strip_marker(inst["target"]))
        entity = f[0] if f else (steps[0].split()[0] if steps else None)
        rows.append(dict(source="legacy", problem_id=r["id"], question=inst["question"],
                         target=inst["target"], entity=entity, gen_text=r["gen_text"],
                         solved=bool(r.get("solved"))))
        seen.add(r["id"])

    # EXPA gold-screen failures (gold_not_solved / gold_not_validator_valid)
    adir = os.path.join(RESULTS, "EXPA_GLOBAL_EXPANSION")
    raw_gold = {}
    for r in read_jsonl(os.path.join(adir, "raw_generations.jsonl")):
        if r["condition"] == "gold":
            raw_gold[r["problem_id"]] = r["continuation"]
    for r in read_jsonl(os.path.join(adir, "eligibility_audit.jsonl")):
        if r.get("stage") != "gold_filter" or r["problem_id"] in seen:
            continue
        rows.append(dict(source="expa", problem_id=r["problem_id"], question=r["question"],
                         target=r["target"], entity=r["entity"],
                         gen_text=raw_gold[r["problem_id"]],
                         solved=(r["reason"] != "gold_not_solved")))
        seen.add(r["problem_id"])

    # EXPE gold-expansion (470 validated rollouts, incl. 334 valid used as controls)
    edir = os.path.join(RESULTS, "EXPE_GOLD_EXPANSION")
    scr = {r["problem_id"]: r for r in read_jsonl(os.path.join(edir, "screened_candidates.jsonl"))}
    raw = {r["problem_id"]: r for r in read_jsonl(os.path.join(edir, "raw_generations.jsonl"))}
    for r in read_jsonl(os.path.join(edir, "validated.jsonl")):
        pid = r["problem_id"]
        if pid in seen:
            continue
        rows.append(dict(source="expe", problem_id=pid, question=scr[pid]["question"],
                         target=scr[pid]["target"], entity=r["entity"],
                         gen_text=raw[pid]["gen_text"], solved=bool(r.get("solved")),
                         gold_valid=bool(r.get("gold_valid"))))
        seen.add(pid)

    # EXPD synthetic-grammar worlds (separate segment)
    ddir = os.path.join(RESULTS, "EXPD_MATCHED_GRADIENT")
    worlds = json.load(open(os.path.join(ddir, "worlds", "worlds.json")))
    raw_gold_d = {}
    for r in read_jsonl(os.path.join(ddir, "raw_generations.jsonl")):
        if r["condition"] == "gold":
            raw_gold_d[r["world_id"]] = r["continuation"]
    for r in read_jsonl(os.path.join(ddir, "gold_cohort.jsonl")):
        if r["reason"] not in ("gold_not_solved", "gold_not_validator_valid"):
            continue
        wid = r["world_id"]
        if wid not in raw_gold_d:
            continue
        rows.append(dict(source="expd", problem_id=wid,
                         question=worlds[wid]["test_example"]["question"],
                         target=r["target"], entity=r["entity"],
                         gen_text=raw_gold_d[wid], solved=bool(r.get("solved"))))
    return rows


# --------------------------------------------------------------------------- walk
def rules_key(rule):
    lhs, rhs = rule
    return (lhs, tuple(rhs))


def try_compound_split(sent, entity):
    """Diagnostic split of 'E is a X, a Y, and a Z.' / 'E is X and Y.' into atomic facts."""
    s = strip_marker(sent).rstrip(".")
    m = re.match(rf"^{re.escape(entity)}\s+is\s+(.+)$", s)
    if not m:
        return None
    body = m.group(1)
    parts = re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+", body)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return None
    preds = []
    for p in parts:
        f = parse_fact(f"{entity} is {p}.", entity)
        if not f:
            return None
        preds.append(f[1])
    disjunctive = (" or " in body)
    return preds, disjunctive


def truth_status(pred, premises, reach):
    """audit_truth_status semantics vs the ORIGINAL world (premises only)."""
    if derivable(pred, premises, reach):
        return "true_entailed"
    if derivable(opposite_pred(pred), premises, reach):
        return "false_contradicted"
    if pred[0] == "cat":
        return "false_cwa_unentailed"
    if pred[0] == "not_cat":
        return "unknown_negated_cat_open_world"
    return "unknown_attr_undecidable"


def pos_bucket(frac):
    return "early" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "late")


def dist_bucket(d):
    if d is None:
        return "unreachable"
    return str(d) if d <= 5 else "6-8"


def walk_rollout(row):
    """Full step-walk for an auditable world. Returns (rollout_record, error_records)."""
    question, target, entity, text = row["question"], row["target"], row["entity"], row["gen_text"]
    rules, facts, _ = parse_world(question)
    reach = closure(rules)
    direct = direct_rule_map(rules)
    premises = {p for e, p in facts if e == entity}
    question_rules = {rules_key(r) for r in rules}
    q_lhs = {lhs for lhs, _ in rules}
    q_rhs = {tuple(rhs) for _, rhs in rules}
    q_words = set(re.findall(r"[a-z]+", question.lower()))
    tf = parse_fact(strip_marker(target), entity)
    target_pred = tf[1] if tf else None

    sents = [strip_marker(x) for x in split_sentences(text)]
    n = len(sents)
    true_state = set(premises)
    visible_state = set(premises)        # everything the entity was CLAIMED to be so far
    tainted = set()                       # error atoms + facts derived through them
    fab_rules = []                        # fabricated rules, in order
    aug_reach = reach
    errors = []                           # fresh natural errors (entity-fact)
    other_errors = []                     # rule_fabricated / off_entity / unparsed
    kind_counts = Counter()
    norm_seen = Counter()

    def err_atoms():
        return [e["pred"] for e in errors]

    for i, s in enumerate(sents):
        norm_seen[re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()] += 1
        frac = i / (n - 1) if n > 1 else 0.0
        f = parse_fact(s, entity)
        if f:
            pred = f[1]
            if derivable(pred, true_state, reach):
                kind_counts["valid_fact"] += 1
                true_state.add(pred)
            elif pred in tainted:
                kind_counts["echo_of_error"] += 1
                for e in errors:
                    if e["pred"] == pred:
                        e["echoed_later"] = True
            elif derivable(pred, true_state | tainted, reach):
                kind_counts["derived_from_error"] += 1
                singles = [e for e in errors
                           if derivable(pred, true_state | {e["pred"]}, reach)]
                for e in singles:
                    e["used_downstream_deriv"] = True
                if not singles:
                    for e in errors:   # joint/chain attribution: mark all as jointly used
                        e["used_downstream_joint"] = True
                tainted.add(pred)
            elif fab_rules and derivable(pred, true_state | tainted, aug_reach):
                kind_counts["derived_via_fabricated_rule"] += 1
                for fr in other_errors:
                    if fr["kind"] == "rule_fabricated":
                        r1 = closure(rules + [fr["rule"]])
                        if derivable(pred, true_state | tainted, r1):
                            fr["used_downstream_deriv"] = True
                tainted.add(pred)
            else:
                kind_counts["fresh_error"] += 1
                dv = (0 if opposite_pred(pred) in visible_state
                      else shortest_rule_distance(opposite_pred(pred), visible_state, direct))
                errors.append(dict(
                    source=row["source"], problem_id=row["problem_id"], sent_idx=i,
                    n_sents=n, sentence=s, kind="entity_fact", pred=pred,
                    stmt_family=predicate_family(pred),
                    polarity="affirmative" if is_positive_pred(pred) else "negated",
                    truth=truth_status(pred, premises, reach),
                    token_in_question=pred_token(pred) in q_words
                        or pred_token(pred) + "es" in q_words,
                    refut_dist_visible=dv,
                    is_target_assertion=(target_pred is not None and pred == target_pred),
                    design_usable_toward_target=bool(
                        pred[0] == "cat" and target_pred is not None
                        and target_pred in reach.get(pred[1], set())),
                    design_usable_any=bool(pred[0] == "cat" and reach.get(pred[1])),
                    position_frac=round(frac, 4), position=pos_bucket(frac),
                    used_downstream_deriv=False, used_downstream_joint=False,
                    echoed_later=False))
                tainted.add(pred)
            visible_state.add(pred)
            continue
        r = parse_rule(s)
        if r:
            if rules_key(r) in question_rules:
                kind_counts["rule_true"] += 1
            else:
                kind_counts["rule_fabricated"] += 1
                fab_rules.append(r)
                aug_reach = closure(rules + fab_rules)
                rk = rules_key(r)
                reversed_real = (r[1][0] == "cat"
                                 and (r[1][1], ("cat", r[0])) in question_rules)
                other_errors.append(dict(
                    source=row["source"], problem_id=row["problem_id"], sent_idx=i,
                    n_sents=n, sentence=s, kind="rule_fabricated", rule=r,
                    is_fewshot_rule=rk in FEWSHOT_RULES,
                    reversed_real_rule=reversed_real,
                    splices_real_vocab=(r[0] in q_lhs or r[0] + "es" in q_words
                                        or r[0] in q_words)
                                       and (tuple(r[1]) in q_rhs
                                            or pred_token(r[1]) in q_words
                                            or pred_token(r[1]) + "es" in q_words),
                    truth_as_implication=("true_entailed"
                                          if tuple(r[1]) in reach.get(r[0], set())
                                          else "not_entailed"),
                    position_frac=round(frac, 4), position=pos_bucket(frac),
                    used_downstream_deriv=False))
            continue
        f_any = parse_fact(s)
        if f_any and f_any[0] != entity:
            kind_counts["off_entity_fact"] += 1
            other_errors.append(dict(
                source=row["source"], problem_id=row["problem_id"], sent_idx=i,
                n_sents=n, sentence=s, kind="off_entity_fact", subject=f_any[0],
                subject_is_fewshot_entity=f_any[0] in FEWSHOT_ENTITIES,
                position_frac=round(frac, 4), position=pos_bucket(frac)))
            continue
        if ENTITYISH.match(s):
            comp = try_compound_split(s, entity)
            status = None
            if comp:
                preds, disjunctive = comp
                if disjunctive:
                    status = "compound_disjunctive"
                elif all(derivable(p, true_state, reach) for p in preds):
                    status = "compound_all_true"
                    for p in preds:
                        true_state.add(p)
                        visible_state.add(p)
                else:
                    status = "compound_contains_underivable"
                    for p in preds:
                        visible_state.add(p)
            kind_counts["unparsed_entity_fact"] += 1
            other_errors.append(dict(
                source=row["source"], problem_id=row["problem_id"], sent_idx=i,
                n_sents=n, sentence=s[:160], kind="unparsed_entity_fact",
                compound_status=status or "not_compound",
                position_frac=round(frac, 4), position=pos_bucket(frac)))
            continue
        kind_counts["other_unparsed"] += 1

    v = validate_continuation(question, [], None, text, target, entity)
    max_rep = max(norm_seen.values()) if norm_seen else 0
    roll = dict(source=row["source"], problem_id=row["problem_id"], solved=row["solved"],
                official_class=v["class"], n_sents=n, kind_counts=dict(kind_counts),
                n_fresh_errors=len(errors),
                n_rule_fabricated=sum(1 for e in other_errors if e["kind"] == "rule_fabricated"),
                n_off_entity=sum(1 for e in other_errors if e["kind"] == "off_entity_fact"),
                n_unparsed_entity=sum(1 for e in other_errors if e["kind"] == "unparsed_entity_fact"),
                any_error_used_deriv=any(e["used_downstream_deriv"] for e in errors),
                any_error_used_joint=any(e["used_downstream_joint"] for e in errors),
                any_error_echoed=any(e["echoed_later"] for e in errors),
                any_fab_rule_used=any(e.get("used_downstream_deriv") for e in other_errors
                                      if e["kind"] == "rule_fabricated"),
                max_sentence_repeat=max_rep,
                truncation_no_falsehood=bool(
                    not row["solved"] and not errors
                    and not any(e["kind"] in ("rule_fabricated", "off_entity_fact")
                                for e in other_errors)))
    return roll, errors, other_errors


# --------------------------------------------------------------------------- main
def world_auditable(question, target, entity):
    rules, facts, unparsed = parse_world(question)
    if unparsed:
        return False
    tf = parse_fact(strip_marker(target), entity)
    return tf is not None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_cohort()
    print(f"cohort rollouts loaded: {len(rows)} "
          f"({Counter(r['source'] for r in rows)})")

    # split valid vs failed using official validator semantics
    failed, valid_ctrl, seg = [], [], Counter()
    for r in rows:
        aud = world_auditable(r["question"], r["target"], r["entity"])
        r["auditable_world"] = aud
        v = validate_continuation(r["question"], [], None, r["gen_text"], r["target"], r["entity"])
        r["official_class"] = v["class"]
        is_fail = (not r["solved"]) or v["class"] != "valid_rederivation"
        seg[(r["source"], aud, is_fail)] += 1
        (failed if is_fail else valid_ctrl).append(r)
    print("segment counts (source, auditable_world, failed):")
    for k in sorted(seg, key=str):
        print("  ", k, seg[k])

    # ---- control sanity: walk VALID rollouts from auditable worlds -> expect ~0 errors
    ctrl = [r for r in valid_ctrl if r["auditable_world"]]
    ctrl_err = 0
    for r in ctrl:
        _, errs, _ = walk_rollout(r)
        ctrl_err += bool(errs)
    print(f"control: {len(ctrl)} valid rollouts walked, {ctrl_err} produced fresh errors "
          f"(false-positive rate {ctrl_err / max(1, len(ctrl)):.3f})")

    # ---- main extraction over failed rollouts on auditable worlds
    err_rows, other_rows, roll_rows = [], [], []
    blind = []
    for r in failed:
        if not r["auditable_world"]:
            blind.append(r)
            continue
        roll, errs, others = walk_rollout(r)
        roll_rows.append(roll)
        err_rows.extend(errs)
        other_rows.extend(others)

    with open(os.path.join(OUT_DIR, "natural_errors.jsonl"), "w") as fh:
        for e in err_rows + other_rows:
            e = dict(e)
            for k in ("pred", "rule"):
                if k in e:
                    e[k] = json.loads(json.dumps(e[k]))  # tuples -> lists
            fh.write(json.dumps(e, sort_keys=True, default=list) + "\n")
    with open(os.path.join(OUT_DIR, "natural_rollouts.jsonl"), "w") as fh:
        for rr in roll_rows:
            fh.write(json.dumps(rr, sort_keys=True) + "\n")

    # ---------------------------------------------------------------- aggregates
    def agg(errors, rolls, label):
        fresh = [e for e in errors if e["kind"] == "entity_fact"]
        fabs = [e for e in errors if e["kind"] == "rule_fabricated"]
        out = {"label": label,
               "fresh_by_source": dict(Counter(e["source"] for e in fresh)),
               "fresh_token_in_question": [sum(1 for e in fresh if e["token_in_question"]),
                                           len(fresh)],
               "fab_rule_fewshot_leak": sum(1 for e in fabs if e["is_fewshot_rule"]),
               "fab_rule_reversed_real": sum(1 for e in fabs if e["reversed_real_rule"]),
               "fab_rule_splices_real_vocab": sum(1 for e in fabs if e["splices_real_vocab"]),
               "fab_rule_true_entailed_implication": sum(
                   1 for e in fabs if e["truth_as_implication"] == "true_entailed"),
               "fab_rule_position": dict(Counter(e["position"] for e in fabs)),
               "n_failed_rollouts_auditable": len(rolls),
               "n_fresh_entity_errors": len(fresh),
               "n_rule_fabricated": sum(1 for e in errors if e["kind"] == "rule_fabricated"),
               "n_off_entity": sum(1 for e in errors if e["kind"] == "off_entity_fact"),
               "n_unparsed_entity": sum(1 for e in errors if e["kind"] == "unparsed_entity_fact"),
               "compound_status": dict(Counter(e.get("compound_status") for e in errors
                                               if e["kind"] == "unparsed_entity_fact")),
               "stmt_type": dict(Counter(f"{e['stmt_family']}_{e['polarity']}" for e in fresh)),
               "truth": dict(Counter(e["truth"] for e in fresh)),
               "type_x_truth": dict(Counter(
                   f"{e['stmt_family']}_{e['polarity']}|{e['truth']}" for e in fresh)),
               "refut_dist": dict(Counter(dist_bucket(e["refut_dist_visible"]) for e in fresh)),
               "refut_dist_contradicted_only": dict(Counter(
                   dist_bucket(e["refut_dist_visible"]) for e in fresh
                   if e["truth"] == "false_contradicted")),
               "position": dict(Counter(e["position"] for e in fresh)),
               "target_assertion": sum(1 for e in fresh if e["is_target_assertion"]),
               "design_usable_toward_target": sum(1 for e in fresh if e["design_usable_toward_target"]),
               "used_downstream_deriv": sum(1 for e in fresh if e["used_downstream_deriv"]),
               "used_downstream_joint_only": sum(1 for e in fresh if e["used_downstream_joint"]
                                                 and not e["used_downstream_deriv"]),
               "echoed_later": sum(1 for e in fresh if e["echoed_later"]),
               "used_deriv_given_design_usable": [
                   sum(1 for e in fresh if e["design_usable_toward_target"] and e["used_downstream_deriv"]),
                   sum(1 for e in fresh if e["design_usable_toward_target"])],
               "used_deriv_given_design_inert": [
                   sum(1 for e in fresh if not e["design_usable_toward_target"] and e["used_downstream_deriv"]),
                   sum(1 for e in fresh if not e["design_usable_toward_target"])],
               "fab_rule_used_downstream": [
                   sum(1 for e in errors if e["kind"] == "rule_fabricated" and e.get("used_downstream_deriv")),
                   sum(1 for e in errors if e["kind"] == "rule_fabricated")],
               "rollout_taxonomy": dict(Counter(
                   "truncation_or_derail_no_falsehood" if rr["truncation_no_falsehood"] else
                   ("has_fresh_entity_error" if rr["n_fresh_errors"] else
                    ("format_only_unparsed" if rr["n_unparsed_entity"] and not rr["n_fresh_errors"] else
                     "other_error_kinds_only")) for rr in rolls)),
               "official_class": dict(Counter(rr["official_class"] for rr in rolls)),
               "rollouts_with_error_any_used": [
                   sum(1 for rr in rolls if rr["n_fresh_errors"] and
                       (rr["any_error_used_deriv"] or rr["any_error_used_joint"])),
                   sum(1 for rr in rolls if rr["n_fresh_errors"])],
               "rollouts_with_error_used_or_echoed": [
                   sum(1 for rr in rolls if rr["n_fresh_errors"] and
                       (rr["any_error_used_deriv"] or rr["any_error_used_joint"] or rr["any_error_echoed"])),
                   sum(1 for rr in rolls if rr["n_fresh_errors"])],
               }
        return out

    pronto_err = [e for e in err_rows + other_rows if e["source"] in ("legacy", "expa", "expe")]
    pronto_roll = [r for r in roll_rows if r["source"] in ("legacy", "expa", "expe")]
    expd_err = [e for e in err_rows + other_rows if e["source"] == "expd"]
    expd_roll = [r for r in roll_rows if r["source"] == "expd"]

    anomalies = [dict(source=rr["source"], problem_id=rr["problem_id"], solved=rr["solved"])
                 for rr in roll_rows if rr["official_class"] == "valid_rederivation"]
    summary = {
        "cohort": {s: c for s, c in Counter(r["source"] for r in rows).items()},
        "segments": {str(k): v for k, v in seg.items()},
        "control_false_positive": [ctrl_err, len(ctrl)],
        "revalidates_valid_but_pipeline_failed": anomalies,
        "validator_blind_failed_rollouts": {
            "n": len(blind),
            "by_source": dict(Counter(r["source"] for r in blind)),
            "by_official_class": dict(Counter(r["official_class"] for r in blind)),
            "note": "worlds with unparsed premises or unparseable targets (Composed grammar); "
                    "'failure' there conflates model error with validator grammar limits"},
        "prontoqa": agg(pronto_err, pronto_roll, "PrOntoQA (legacy+EXPA+EXPE), auditable worlds"),
        "expd_synthetic": agg(expd_err, expd_roll, "EXPD synthetic in-grammar worlds"),
        "combined": agg(err_rows + other_rows, roll_rows, "all auditable sources"),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
