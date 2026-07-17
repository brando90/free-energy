#!/usr/bin/env python3
"""Finite-world experiment for H03.

The goal is to separate operations that only need ordering or score ratios from
operations that require a normalized probability measure.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


CANDIDATES = ["A", "B", "C", "D", "E"]
BASE_SCORES = [2.25, 1.65, 0.20, -0.55, -1.70]

HIDDEN_STATES = [
    ("h0", "A", 1.85),
    ("h1", "A", 1.25),
    ("h2", "B", 1.55),
    ("h3", "C", 0.15),
    ("h4", "D", -0.55),
    ("h5", "E", -1.40),
]

PRIOR_LOG = [0.10, -0.10, 0.00, -0.20, -0.35]
EVIDENCE_1 = [1.20, 0.75, 0.15, -0.25, -0.85]
EVIDENCE_2 = [0.70, 0.95, -0.10, -0.45, -1.10]


def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def entropy(probabilities: Iterable[float]) -> float:
    return -sum(p * math.log(p) for p in probabilities if p > 0.0)


def total_variation(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def kl_divergence(p: list[float], q: list[float]) -> float:
    eps = 1e-15
    return sum(a * math.log(a / max(b, eps)) for a, b in zip(p, q) if a > 0.0)


def expected_multiclass_brier(true_p: list[float], pred_p: list[float]) -> float:
    total = 0.0
    for y, py in enumerate(true_p):
        total += py * sum(
            (pred_p[i] - (1.0 if i == y else 0.0)) ** 2
            for i in range(len(true_p))
        )
    return total


def cross_entropy(true_p: list[float], pred_p: list[float]) -> float:
    eps = 1e-15
    return -sum(p * math.log(max(q, eps)) for p, q in zip(true_p, pred_p))


def rank_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)


def top_k(scores: list[float], k: int) -> set[int]:
    return set(rank_order(scores)[:k])


def rank_preserving_codes(scores: list[float]) -> list[float]:
    order = rank_order(scores)
    values = [0.0] * len(scores)
    # Deliberately compressed: ranking is intact, probability scale is not.
    codebook = [1.00, 0.82, 0.68, 0.55, 0.43, 0.32, 0.22, 0.13]
    for rank, idx in enumerate(order):
        values[idx] = codebook[rank]
    return values


Transform = Callable[[list[float]], list[float]]


def transform_identity(scores: list[float]) -> list[float]:
    return list(scores)


def transform_affine_shift(scores: list[float]) -> list[float]:
    return [2.0 * score + 3.5 for score in scores]


def transform_temperature_hot(scores: list[float]) -> list[float]:
    return [0.35 * score for score in scores]


def transform_temperature_cold(scores: list[float]) -> list[float]:
    return [2.25 * score for score in scores]


def transform_nonlinear_monotone(scores: list[float]) -> list[float]:
    return [score + 0.10 * (score**3) for score in scores]


TRANSFORMS: list[tuple[str, Transform]] = [
    ("identity", transform_identity),
    ("affine_shift_scale", transform_affine_shift),
    ("temperature_hot", transform_temperature_hot),
    ("temperature_cold", transform_temperature_cold),
    ("nonlinear_monotone", transform_nonlinear_monotone),
    ("rank_preserving_bad_calibration", rank_preserving_codes),
]


def action_utilities(probabilities: list[float]) -> dict[str, float]:
    p_a = probabilities[0]
    p_top2 = probabilities[0] + probabilities[1]
    return {
        "commit_A": 1.00 * p_a - 0.60 * (1.0 - p_a),
        "commit_top2": 0.72 * p_top2 - 0.40 * (1.0 - p_top2),
        "collect_more_evidence": 0.25,
        "abstain": 0.00,
    }


def choose_action(probabilities: list[float]) -> str:
    utilities = action_utilities(probabilities)
    return max(utilities, key=utilities.get)


def observed_marginal(hidden_scores: list[float]) -> list[float]:
    hidden_probs = softmax(hidden_scores)
    totals = {candidate: 0.0 for candidate in CANDIDATES}
    for (_, observed, _), p_h in zip(HIDDEN_STATES, hidden_probs):
        totals[observed] += p_h
    return [totals[candidate] for candidate in CANDIDATES]


def posterior(scores: list[float]) -> list[float]:
    return softmax(scores)


def transformed_posterior(transform: Transform) -> list[float]:
    prior = transform(PRIOR_LOG)
    evidence_1 = transform(EVIDENCE_1)
    evidence_2 = transform(EVIDENCE_2)
    return posterior([a + b + c for a, b, c in zip(prior, evidence_1, evidence_2)])


def row_for_transform(name: str, transform: Transform) -> dict[str, object]:
    base_p = softmax(BASE_SCORES)
    transformed_scores = transform(BASE_SCORES)
    transformed_p = softmax(transformed_scores)
    base_action = choose_action(base_p)
    action = choose_action(transformed_p)
    base_utility = action_utilities(base_p)[base_action]
    action_regret = base_utility - action_utilities(base_p)[action]

    hidden_base = [score for _, _, score in HIDDEN_STATES]
    hidden_transformed = transform(hidden_base)
    base_marginal = observed_marginal(hidden_base)
    transformed_marginal = observed_marginal(hidden_transformed)

    base_posterior = posterior(
        [a + b + c for a, b, c in zip(PRIOR_LOG, EVIDENCE_1, EVIDENCE_2)]
    )
    q_posterior = transformed_posterior(transform)

    return {
        "transform": name,
        "rank_agreement": rank_order(BASE_SCORES) == rank_order(transformed_scores),
        "top1_agreement": top_k(BASE_SCORES, 1) == top_k(transformed_scores, 1),
        "top2_agreement": top_k(BASE_SCORES, 2) == top_k(transformed_scores, 2),
        "p_A": transformed_p[0],
        "entropy": entropy(transformed_p),
        "tv_from_base_probability": total_variation(base_p, transformed_p),
        "kl_base_to_transform": kl_divergence(base_p, transformed_p),
        "cross_entropy_under_base": cross_entropy(base_p, transformed_p),
        "expected_brier_under_base": expected_multiclass_brier(base_p, transformed_p),
        "chosen_action": action,
        "action_regret_under_base": action_regret,
        "abstain_if_top_p_below_0_55": transformed_p[0] < 0.55,
        "latent_marginal_tv_from_base": total_variation(base_marginal, transformed_marginal),
        "posterior_tv_from_base": total_variation(base_posterior, q_posterior),
    }


CAPABILITY_ROWS = [
    {
        "task": "argmax / top-k search",
        "minimal_object": "ordering",
        "rank_only": "yes",
        "score_difference": "yes",
        "unnormalized_score": "yes",
        "normalized_probability_required": "no",
        "note": "Any strictly monotone transform preserves the answer.",
    },
    {
        "task": "Metropolis-Hastings accept/reject ratio",
        "minimal_object": "score difference or unnormalized density ratio",
        "rank_only": "no",
        "score_difference": "yes",
        "unnormalized_score": "yes",
        "normalized_probability_required": "no",
        "note": "The global partition constant cancels in same-x ratios.",
    },
    {
        "task": "calibrated abstention",
        "minimal_object": "calibrated probability",
        "rank_only": "no",
        "score_difference": "no",
        "unnormalized_score": "no",
        "normalized_probability_required": "yes",
        "note": "A threshold such as P(top)>0.55 is meaningless without calibration.",
    },
    {
        "task": "expected utility decision",
        "minimal_object": "calibrated probability over outcomes",
        "rank_only": "no",
        "score_difference": "no",
        "unnormalized_score": "no",
        "normalized_probability_required": "yes",
        "note": "Utilities integrate over possible states, not only the top state.",
    },
    {
        "task": "entropy / uncertainty",
        "minimal_object": "normalized probability distribution",
        "rank_only": "no",
        "score_difference": "no",
        "unnormalized_score": "no",
        "normalized_probability_required": "yes",
        "note": "Entropy changes under rank-preserving temperature transforms.",
    },
    {
        "task": "exact categorical sampling",
        "minimal_object": "normalized probability or equivalent exact sampler",
        "rank_only": "no",
        "score_difference": "no",
        "unnormalized_score": "partial",
        "normalized_probability_required": "yes",
        "note": "Finite enumeration can normalize internally; rank alone cannot sample.",
    },
    {
        "task": "latent-variable marginalization",
        "minimal_object": "summed weights and final probability scale",
        "rank_only": "no",
        "score_difference": "partial",
        "unnormalized_score": "partial",
        "normalized_probability_required": "yes",
        "note": "Multiplicity over hidden states is lost under max/rank-only summaries.",
    },
    {
        "task": "Bayesian evidence composition",
        "minimal_object": "calibrated log-ratios plus posterior normalization",
        "rank_only": "no",
        "score_difference": "partial",
        "unnormalized_score": "partial",
        "normalized_probability_required": "yes",
        "note": "Log-ratios combine, but posterior decisions need calibrated scale.",
    },
    {
        "task": "compare likelihoods across different inputs",
        "minimal_object": "input-specific normalized likelihood",
        "rank_only": "no",
        "score_difference": "no",
        "unnormalized_score": "no",
        "normalized_probability_required": "yes",
        "note": "Different partition functions no longer cancel.",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row[header]).replace("|", "/") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def rounded_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rounded = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            if isinstance(value, float):
                new_row[key] = round(value, 6)
            else:
                new_row[key] = value
        rounded.append(new_row)
    return rounded


def make_verdict(task_rows: list[dict[str, object]]) -> str:
    non_identity = [row for row in task_rows if row["transform"] != "identity"]
    all_rank_preserved = all(row["rank_agreement"] for row in non_identity)
    worst_tv = max(float(row["tv_from_base_probability"]) for row in non_identity)
    worst_kl = max(float(row["kl_base_to_transform"]) for row in non_identity)
    changed_actions = [
        row["transform"]
        for row in non_identity
        if row["chosen_action"] != task_rows[0]["chosen_action"]
    ]
    worst_latent_tv = max(float(row["latent_marginal_tv_from_base"]) for row in non_identity)
    worst_posterior_tv = max(float(row["posterior_tv_from_base"]) for row in non_identity)

    return (
        "# Verdict\n\n"
        "H03 is supported in the probability-level sense and weakened in the "
        "ranking-only sense. In this finite world, all non-identity transforms "
        f"preserved ranking: {all_rank_preserved}. However the same rank order "
        "produced materially different probability objects: worst total "
        f"variation from the base distribution was {worst_tv:.3f}, worst KL was "
        f"{worst_kl:.3f}, worst latent-marginal TV was {worst_latent_tv:.3f}, "
        f"and worst composed-posterior TV was {worst_posterior_tv:.3f}. "
        f"Expected-utility actions changed under: {', '.join(changed_actions) or 'none'}. "
        "So normalization is not needed for argmax/top-k search or same-support "
        "ratio tests where constants cancel, but it gives concrete mathematical "
        "power for calibrated abstention, entropy, exact sampling, marginal "
        "probabilities, and posterior/expected-utility decisions.\n"
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    task_rows = rounded_rows([row_for_transform(name, transform) for name, transform in TRANSFORMS])

    write_csv(RESULTS / "task_summary.csv", task_rows)
    write_csv(RESULTS / "capability_matrix.csv", CAPABILITY_ROWS)
    (RESULTS / "capability_matrix.md").write_text(markdown_table(CAPABILITY_ROWS))
    verdict = make_verdict(task_rows)
    (RESULTS / "verdict.md").write_text(verdict)

    summary = {
        "hypothesis": "H03 normalization probability power",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "candidates": CANDIDATES,
        "base_scores": BASE_SCORES,
        "base_probabilities": [round(value, 6) for value in softmax(BASE_SCORES)],
        "transforms": [name for name, _ in TRANSFORMS],
        "task_summary": task_rows,
        "capability_matrix": CAPABILITY_ROWS,
        "verdict": verdict,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(verdict)
    print(f"Wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
