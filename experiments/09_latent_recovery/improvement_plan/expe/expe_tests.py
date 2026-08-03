"""Fixture tests for EXPE evidence-mover: arm construction, fail-closed audits.

Includes one hand-built world per arm where the audit MUST fail closed, plus a
positive-path world where all six arms construct and audit clean.

Run:  python expe_tests.py     (CPU only; imports validator from $EXP/src)
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expe_evidence_mover import (
    ARM_FILLER,
    ARM_FREQ,
    ARM_REFUTING_D1,
    ARM_REFUTING_D2,
    ARM_REFUTING_D3,
    ARM_TEMPLATE,
    ARMS,
    audit_arm,
    augment_question,
    build_arm_specs,
    build_problem_arms,
    falsehood_candidates,
    injection_points,
    rule_sentence,
)
from validator import validate_continuation

# Goal chain lorpus->shumpus->wumpus->tumpus->(not liquid); distractor chain
# dumpus->grimpus->zumpus (never walked); unentailed cats vumpus->yumpus,
# rompus->jompus, impus->numpus (poisoning-measurable falsehood candidates).
QUESTION = (
    "Every lorpus is a shumpus. "
    "Every shumpus is a wumpus. "
    "Every wumpus is a tumpus. "
    "Every tumpus is not liquid. "
    "Every dumpus is a grimpus. "
    "Every grimpus is a zumpus. "
    "Every vumpus is a yumpus. "
    "Every rompus is a jompus. "
    "Each impus is a numpus. "
    "Stella is a lorpus. "
    "Stella is a dumpus."
)
TARGET = "Stella is not liquid."
ENTITY = "Stella"
STEPS = [
    "Stella is a lorpus.",
    "Every lorpus is a shumpus.",
    "Stella is a shumpus.",
    "Every shumpus is a wumpus.",
    "Stella is a wumpus.",
    "Every wumpus is a tumpus.",
    "Stella is a tumpus.",
    "Every tumpus is not liquid.",
    "Stella is not liquid.",
]
GOLD_TEXT = " ".join(STEPS)


def check(name, cond):
    if not cond:
        raise AssertionError(name)


def spec(body, head, arm, hop=None):
    return {"arm": arm, "body": body, "head": head, "antecedent_hop": hop,
            "antecedent_in_gold_suffix": False,
            "rule_sentence": rule_sentence(body, head)}


def main():
    # ---- cohort mechanics ---------------------------------------------------
    pts = injection_points(STEPS, ENTITY)
    check("mid injection point", pts is not None and pts["mid"] == 4)
    si = pts["mid"]
    gv = validate_continuation(QUESTION, [], None, GOLD_TEXT, TARGET, ENTITY)
    check("fixture gold validates", gv["class"] == "valid_rederivation")

    # ---- falsehood candidate ladder ----------------------------------------
    cands = falsehood_candidates(QUESTION, ENTITY, STEPS[:si])
    check("measurable falsehoods first",
          cands[0] == ("impus", True) and ("yumpus", False) in cands)
    check("entailed cats never falsehoods",
          all(c not in ("lorpus", "shumpus", "wumpus", "tumpus", "dumpus",
                        "grimpus", "zumpus") for c, _ in cands))

    # ---- positive path: all six arms construct and audit clean -------------
    built, diags = build_problem_arms(QUESTION, STEPS, ENTITY, TARGET, si, GOLD_TEXT)
    check("all arms available", built is not None)
    check("selected F is measurable-first",
          built["fcat"] == "impus" and built["poisoning_measurable"])
    check("rich pool stays fully in-vocabulary distinct",
          built["vocab_tier"] == "A_invocab_distinct")
    check("F statement", built["falsehood_statement"] == "Stella is an impus.")
    rules = {a: built["specs"][a]["rule_sentence"] for a in ARMS}
    check("d1 rule", rules[ARM_REFUTING_D1] == "Every dumpus is not an impus.")
    check("d2 rule prefers off-suffix antecedent",
          rules[ARM_REFUTING_D2] == "Every grimpus is not an impus.")
    check("d3 rule prefers off-suffix antecedent",
          rules[ARM_REFUTING_D3] == "Every zumpus is not an impus.")
    check("freq rule", rules[ARM_FREQ] == "Every jompus is not an impus.")
    check("template rule shares d1 antecedent",
          rules[ARM_TEMPLATE] == "Every dumpus is not a rompus.")
    check("filler rule", rules[ARM_FILLER] == "Every numpus is not a vumpus.")
    for arm in ARMS:
        a = built["audits"][arm]
        check(f"{arm} fully audited", a["fully_audited"])
        check(f"{arm} gold revalidates",
              a["gold_revalidation_class"] == "valid_rederivation")
    check("measured d ladder",
          built["audits"][ARM_REFUTING_D1]["measured_d_augmented"] == 1
          and built["audits"][ARM_REFUTING_D2]["measured_d_augmented"] == 2
          and built["audits"][ARM_REFUTING_D3]["measured_d_augmented"] == 3)
    for arm in (ARM_FREQ, ARM_TEMPLATE, ARM_FILLER):
        check(f"{arm} no finite d",
              built["audits"][arm]["measured_d_augmented"] is None)
    for arm in (ARM_REFUTING_D1, ARM_REFUTING_D2, ARM_REFUTING_D3):
        check(f"{arm} F becomes provably false",
              built["audits"][arm]["checks"]["falsehood_becomes_provably_false"])
    for arm in (ARM_FREQ, ARM_TEMPLATE, ARM_FILLER):
        check(f"{arm} F stays CWA-false",
              built["audits"][arm]["checks"]["falsehood_stays_cwa_false"])

    # insertion slot: fixed 'back' = just before the first fact (index 9 here),
    # identical across arms; the trace side is untouched.
    slots = {built["audits"][a]["insert_index"] for a in ARMS}
    check("same back slot across arms", slots == {9})
    aug_q, ins, off = augment_question(QUESTION, rules[ARM_REFUTING_D1])
    check("augment inserts before facts",
          ins == 9 and aug_q.split(". ")[9].startswith("Every dumpus is not"))
    check("duplicate rule refused", augment_question(aug_q, rules[ARM_REFUTING_D1])[0] is None)

    # ---- fail-closed audits: one hand-built failure per arm ----------------
    # (1) REFUTING_d1 with a never-entailed antecedent: measured d is None != 1.
    a = audit_arm(QUESTION, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_REFUTING_D1, spec("jompus", "impus", ARM_REFUTING_D1, hop=0))
    check("d1 fail-closed on unentailed antecedent",
          not a["fully_audited"] and not a["checks"]["measured_d_equals_designed"])

    # (2) REFUTING_d2 with a hop-0 antecedent: measured d == 1 != 2.
    a = audit_arm(QUESTION, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_REFUTING_D2, spec("dumpus", "impus", ARM_REFUTING_D2, hop=1))
    check("d2 fail-closed on hop-0 antecedent",
          not a["fully_audited"] and a["measured_d_augmented"] == 1)

    # (3) REFUTING_d3 with a hop-1 antecedent: measured d == 2 != 3.
    a = audit_arm(QUESTION, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_REFUTING_D3, spec("grimpus", "impus", ARM_REFUTING_D3, hop=2))
    check("d3 fail-closed on hop-1 antecedent",
          not a["fully_audited"] and a["measured_d_augmented"] == 2)

    # (4) FREQ arm whose antecedent IS entailed: it silently refutes -> reject.
    a = audit_arm(QUESTION, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_FREQ, spec("shumpus", "impus", ARM_FREQ))
    check("freq fail-closed when rule refutes",
          not a["fully_audited"]
          and not a["checks"]["remains_not_locally_refutable"]
          and not a["checks"]["freq_antecedent_never_entailed"])

    # (5) TEMPLATE arm whose head is reachable from fcat (modus-tollens route).
    a = audit_arm(QUESTION, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_TEMPLATE, spec("dumpus", "numpus", ARM_TEMPLATE, hop=0))
    check("template fail-closed on reachable head",
          not a["fully_audited"]
          and not a["checks"]["template_head_unreachable_from_falsehood"])

    # (6) FILLER arm whose body is entailed: not entity-disconnected -> reject.
    a = audit_arm(QUESTION, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_FILLER, spec("lorpus", "vumpus", ARM_FILLER))
    check("filler fail-closed on entailed body",
          not a["fully_audited"] and not a["checks"]["filler_body_never_entailed"])

    # (7) F not CWA-false (entailed category): every arm must reject it.
    a = audit_arm(QUESTION, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "wumpus",
                  ARM_REFUTING_D1, spec("dumpus", "wumpus", ARM_REFUTING_D1, hop=0))
    check("entailed F fail-closed",
          not a["fully_audited"] and not a["checks"]["falsehood_unentailed_original"])

    # (8) F already provably false in the original world: excluded upstream and
    # rejected by the audit.
    q2 = QUESTION + " Every lorpus is not a pumpus."
    check("pre-refuted F excluded from candidates",
          all(c != "pumpus" for c, _ in falsehood_candidates(q2, ENTITY, STEPS[:si])))
    a = audit_arm(q2, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "pumpus",
                  ARM_FREQ, spec("jompus", "pumpus", ARM_FREQ))
    check("pre-refuted F fail-closed",
          not a["fully_audited"]
          and not a["checks"]["falsehood_not_locally_refutable_original"]
          and not a["checks"]["falsehood_complement_unentailed_original"])

    # (9) accidental shortcut: added rule head == target predicate -> reject.
    a = audit_arm(QUESTION, ENTITY, "Stella is not an impus.", STEPS, si, GOLD_TEXT,
                  "impus", ARM_REFUTING_D1, spec("dumpus", "impus", ARM_REFUTING_D1, hop=0))
    check("target shortcut fail-closed", not a["checks"]["no_target_shortcut"])

    # (10) duplicate rule already in the question -> reject.
    q3 = QUESTION + " Every dumpus is not an impus."
    specs3, _ = build_arm_specs(q3, STEPS, ENTITY, si, "impus")
    check("construction skips pre-existing rule",
          specs3[ARM_REFUTING_D1]["rule_sentence"] != "Every dumpus is not an impus.")
    a = audit_arm(q3, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_REFUTING_D1, spec("dumpus", "impus", ARM_REFUTING_D1, hop=0))
    check("duplicate-rule audit fail-closed", not a["fully_audited"])

    # ---- vocabulary-tier ladder (scarce natural pools) -----------------------
    # Tiny world: only unentailed cats are impus (fcat) and numpus, and numpus
    # is reachable from impus -> TEMPLATE head and FILLER head must fall back to
    # fresh nonce nouns (tier C); FREQ antecedent stays in-vocabulary.
    Q_SMALL = (
        "Every lorpus is a shumpus. "
        "Every shumpus is a wumpus. "
        "Every wumpus is a tumpus. "
        "Every tumpus is not liquid. "
        "Each impus is a numpus. "
        "Stella is a lorpus."
    )
    built_s, _ = build_problem_arms(Q_SMALL, STEPS, ENTITY, TARGET, si, GOLD_TEXT)
    check("tiny pool available via tier C", built_s is not None)
    check("tier C flagged", built_s["vocab_tier"] == "C_fresh_nouns")
    check("freq antecedent stays in-vocab",
          built_s["specs"][ARM_FREQ]["body"] == "numpus"
          and built_s["specs"][ARM_FREQ]["body_source"] == "question_vocab")
    check("template head is fresh",
          built_s["specs"][ARM_TEMPLATE]["head_source"] == "fresh"
          and built_s["specs"][ARM_TEMPLATE]["head"] == "florpus")
    check("filler reuses in-vocab body, fresh head",
          built_s["specs"][ARM_FILLER]["body_source"] == "question_vocab_reused"
          and built_s["specs"][ARM_FILLER]["head_source"] == "fresh")
    for arm in ARMS:
        check(f"tier C {arm} still fully audited",
              built_s["audits"][arm]["fully_audited"])

    # No in-vocabulary unentailed category besides fcat -> FREQ has no legal
    # antecedent (fresh forbidden there) -> whole problem must be unavailable.
    Q_NOZ = (
        "Every lorpus is a shumpus. "
        "Every shumpus is a wumpus. "
        "Every wumpus is a tumpus. "
        "Every tumpus is not liquid. "
        "Every grimpus is a lorpus. "
        "Stella is a lorpus."
    )
    built_n, diags_n = build_problem_arms(Q_NOZ, STEPS, ENTITY, TARGET, si, GOLD_TEXT)
    check("freq starvation fails closed",
          built_n is None and ARM_FREQ not in diags_n["arm_available_under_some_F"])

    # A fresh-noun FREQ antecedent must be rejected by the audit itself too.
    a = audit_arm(Q_SMALL, ENTITY, TARGET, STEPS, si, GOLD_TEXT, "impus",
                  ARM_FREQ, spec("florpus", "impus", ARM_FREQ))
    check("fresh freq antecedent fail-closed",
          not a["fully_audited"]
          and not a["checks"]["freq_antecedent_in_question_vocab"])

    # ---- trace-side invariance ----------------------------------------------
    prefill = " " + " ".join(STEPS[:si] + [built["falsehood_statement"]])
    check("prefill replaces the mid step",
          "Stella is a wumpus." not in prefill and prefill.endswith("Stella is an impus."))

    # ---- validator DV wiring -------------------------------------------------
    v = validate_continuation(aug_q, STEPS[:si], built["falsehood_statement"],
                              "Stella is a wumpus. Stella is a tumpus. Stella is not liquid.",
                              TARGET, ENTITY)
    check("valid recovery classified", v["class"] == "valid_rederivation")
    v = validate_continuation(aug_q, STEPS[:si], built["falsehood_statement"],
                              "Stella is a numpus. Stella is not liquid.",
                              TARGET, ENTITY)
    check("poisoned classified (F reused derivationally)", v["class"] == "poisoned")
    v = validate_continuation(aug_q, STEPS[:si], built["falsehood_statement"],
                              "Wait, Stella is not an impus. Stella is a wumpus. "
                              "Stella is a tumpus. Stella is not liquid.",
                              TARGET, ENTITY)
    check("refutation sentence derivable under REFUTING arm and doubt flagged",
          v["class"] == "valid_rederivation" and v["acknowledged"])

    print("EXPE fixture tests passed")


if __name__ == "__main__":
    main()
