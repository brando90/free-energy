"""Unit tests for Experiment C polarity-control audit and validation.

Run with:
  /lfs/skampere2/0/eobbad/free-energy/.venv/bin/python src/expc_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expc_polarity_control import (
    GLOBAL_FALSE_NEGATIVE,
    GLOBAL_FALSE_POSITIVE,
    LOCAL_FALSE_NEGATIVE,
    LOCAL_FALSE_POSITIVE,
    audit_truth_status,
    build_injections,
    injection_points,
)
from validator import validate_continuation


QUESTION = (
    "Every lorpus is a shumpus. "
    "Every shumpus is a wumpus. "
    "Every wumpus is a tumpus. "
    "Every wumpus is not liquid. "
    "Every grimpus is a zumpus. "
    "Every zumpus is cold. "
    "Stella is a lorpus."
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
    "Every wumpus is not liquid.",
    "Stella is not liquid.",
]


def check(name, cond):
    if not cond:
        raise AssertionError(name)


def planned(condition, point="late"):
    cand = {
        "problem_id": "unit",
        "question": QUESTION,
        "target": TARGET,
        "entity": ENTITY,
        "original_gold_steps": STEPS,
        "injection_points": injection_points(STEPS, ENTITY),
    }
    rows = build_injections(cand)
    for row in rows:
        if row["condition"] == condition and row["injection_position"] == point:
            return row
    raise AssertionError(f"missing planned row {condition} {point}")


def main():
    # Independent truth audit coverage: true, false, local, and global cases.
    check("true category audit", audit_truth_status(QUESTION, "Stella is a shumpus.", ENTITY)["truth_status"] == "true")
    lp = audit_truth_status(QUESTION, "Stella is liquid.", ENTITY, STEPS[:5])
    check("local false positive audit", lp["truth_status"] == "false" and lp["local_refutable_from_prefix"])
    ln = audit_truth_status(QUESTION, "Stella is not a shumpus.", ENTITY, STEPS[:2])
    check("local false negative audit", ln["truth_status"] == "false")
    gp = audit_truth_status(QUESTION, "Stella is a grimpus.", ENTITY, STEPS[:4])
    check("global false positive audit", gp["truth_status"] == "false" and not gp["local_refutable_from_prefix"])
    gn = audit_truth_status(QUESTION, "Stella is not a wumpus.", ENTITY, STEPS[:2])
    check("global false negative audit", gn["truth_status"] == "false" and gn["local_rule_distance"] >= 2)

    # Injection construction labels all four cells and fails closed when unavailable.
    check("local positive available", planned(LOCAL_FALSE_POSITIVE)["available"])
    check("local negative available", planned(LOCAL_FALSE_NEGATIVE, "early")["available"])
    check("global positive available", planned(GLOBAL_FALSE_POSITIVE)["available"])
    check("global negative available", planned(GLOBAL_FALSE_NEGATIVE, "early")["available"])

    # Valid with skipped intermediate steps: closure permits direct target statement.
    v = validate_continuation(QUESTION, [], None, "Stella is not liquid.", TARGET, ENTITY)
    check("skipped valid", v["class"] == "valid_rederivation")

    # Poisoned: the continuation uses the planted false category to derive later facts.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        "Stella is a grimpus.",
        "Every grimpus is a zumpus. Stella is a zumpus. Every zumpus is cold. Stella is cold. Stella is not liquid.",
        TARGET,
        ENTITY,
    )
    check("poisoned", v["class"] == "poisoned")

    # Parroted: final target reached after an underivable intermediate.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        None,
        "Stella is a grimpus. Stella is not liquid.",
        TARGET,
        ENTITY,
    )
    check("parroted", v["class"] == "parroted")

    # False negated category parses as an invalid logical statement rather than
    # being treated as unparsable text.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        None,
        "Stella is not a shumpus. Stella is not liquid.",
        TARGET,
        ENTITY,
    )
    check("false negated category parsed", v["class"] == "parroted")

    # Unparsed entity sentence is preserved rather than silently dropped.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        None,
        "Stella is very shumpus. Stella is not liquid.",
        TARGET,
        ENTITY,
    )
    check("unparsed", v["class"] == "unparsed")

    print("EXPC validator tests passed")


if __name__ == "__main__":
    main()
