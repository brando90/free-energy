"""Unit tests for Experiment A truth audit and validation cases.

Run with:
  /lfs/skampere2/0/eobbad/free-energy/.venv/bin/python src/expa_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expa_global_expansion import audit_truth_status
from validator import validate_continuation


QUESTION = (
    "Every lorpus is a shumpus. "
    "Every shumpus is a wumpus. "
    "Every wumpus is not liquid. "
    "Every grimpus is a zumpus. "
    "Every zumpus is cold. "
    "Stella is a lorpus."
)
TARGET = "Stella is not liquid."
ENTITY = "Stella"


def check(name, cond):
    if not cond:
        raise AssertionError(name)


def main():
    # Truth audit: true, global false/unentailed, local negation false.
    check("true audit", audit_truth_status(QUESTION, "Stella is a shumpus.", ENTITY)["truth_status"] == "true")
    gfalse = audit_truth_status(QUESTION, "Stella is a grimpus.", ENTITY)
    check("global false audit", gfalse["truth_status"] == "false" and gfalse["audit_basis"] == "unentailed_category_closed_world")
    local_false = audit_truth_status(QUESTION, "Stella is not a shumpus.", ENTITY)
    check("local false audit", local_false["truth_status"] == "false")

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

    # Derailed: valid-looking prefix but wrong final answer.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        "Stella is not a shumpus.",
        "Every lorpus is a shumpus. Stella is a shumpus. Every shumpus is a wumpus. Stella is a wumpus.",
        TARGET,
        ENTITY,
    )
    check("derailed", v["class"] == "derailed")

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

    print("EXPA validator tests passed")


if __name__ == "__main__":
    main()
