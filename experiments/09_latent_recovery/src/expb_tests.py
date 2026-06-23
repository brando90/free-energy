"""Unit tests for Experiment B local-certificate audit and closure validation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expb_local_cert_flip import (
    audit_truth_status,
    locally_falsifies,
)
from validator import validate_continuation


QUESTION = (
    "Every lorpus is a shumpus. "
    "Every shumpus is a wumpus. "
    "Every wumpus is not liquid. "
    "Every grimpus is a zumpus. "
    "Every zumpus is cold. "
    "Every brimpus is loud. "
    "Stella is a lorpus."
)
TARGET = "Stella is not liquid."
ENTITY = "Stella"


def check(name, cond):
    if not cond:
        raise AssertionError(name)


def main():
    # Independent truth audit: true proof fact, globally false F, and true local certificate.
    check("true audit", audit_truth_status(QUESTION, "Stella is a shumpus.", ENTITY)["truth_status"] == "true")
    check("global false audit", audit_truth_status(QUESTION, "Stella is a grimpus.", ENTITY)["truth_status"] == "false")
    check("local certificate true", audit_truth_status(QUESTION, "Stella is not a grimpus.", ENTITY)["truth_status"] == "true")
    check("irrelevant certificate true", audit_truth_status(QUESTION, "Stella is not a brimpus.", ENTITY)["truth_status"] == "true")

    # Local/global certificate relation checks.
    check(
        "local falsifies",
        locally_falsifies("Stella is a grimpus.", "Stella is not a grimpus.", QUESTION, ENTITY),
    )
    check(
        "entailed local falsifies",
        locally_falsifies("Stella is a grimpus.", "Stella is not a zumpus.", QUESTION, ENTITY),
    )
    check(
        "irrelevant does not falsify",
        not locally_falsifies("Stella is a grimpus.", "Stella is not a brimpus.", QUESTION, ENTITY),
    )

    # Skipped but valid: closure allows the final target directly.
    v = validate_continuation(QUESTION, [], None, "Stella is not liquid.", TARGET, ENTITY)
    check("skipped valid", v["class"] == "valid_rederivation")

    # Poisoned: the planted globally false category is used to derive later facts.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        "Stella is a grimpus.",
        "Every grimpus is a zumpus. Stella is a zumpus. Every zumpus is cold. Stella is cold. Stella is not liquid.",
        TARGET,
        ENTITY,
    )
    check("poisoned", v["class"] == "poisoned")

    # Parroted: final target appears after an underivable intermediate.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        None,
        "Stella is a grimpus. Stella is not liquid.",
        TARGET,
        ENTITY,
    )
    check("parroted", v["class"] == "parroted")

    # Derailed: final statement is not the target.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        "Stella is not a grimpus.",
        "Every lorpus is a shumpus. Stella is a shumpus. Every shumpus is a wumpus. Stella is a wumpus.",
        TARGET,
        ENTITY,
    )
    check("derailed", v["class"] == "derailed")

    # Unparsed entity-like sentence must be logged as unparsed.
    v = validate_continuation(
        QUESTION,
        ["Stella is a lorpus."],
        None,
        "Stella is very shumpus. Stella is not liquid.",
        TARGET,
        ENTITY,
    )
    check("unparsed", v["class"] == "unparsed")

    print("EXPB validator tests passed")


if __name__ == "__main__":
    main()
