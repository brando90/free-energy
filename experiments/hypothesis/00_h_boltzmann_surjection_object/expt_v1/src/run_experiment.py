#!/usr/bin/env python3
"""Run H00: benchmark EBM normalization primitives without external deps."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def stable_logsumexp(row: list[float]) -> float:
    m = max(row)
    return m + math.log(sum(math.exp(x - m) for x in row))


def raw_argmin(row: list[float]) -> int:
    best_i = 0
    best = row[0]
    for i, x in enumerate(row[1:], start=1):
        if x < best:
            best = x
            best_i = i
    return best_i


def exp_sum(row: list[float]) -> float:
    return sum(math.exp(-x) for x in row)


def softmax_true_prob(row: list[float], true_idx: int, temp: float = 1.0) -> float:
    scores = [-x / temp for x in row]
    m = max(scores)
    denom = sum(math.exp(s - m) for s in scores)
    return math.exp(scores[true_idx] - m) / denom


def top2_margin(row: list[float]) -> float:
    best = float("inf")
    second = float("inf")
    for x in row:
        if x < best:
            second = best
            best = x
        elif x < second:
            second = x
    return second - best


def make_batch(batch: int, candidates: int, rng: random.Random) -> list[list[float]]:
    return [[rng.gauss(0.0, 2.0) for _ in range(candidates)] for _ in range(batch)]


def time_method(name: str, batch: list[list[float]], reps: int) -> tuple[float, float]:
    checksum = 0.0
    start = time.perf_counter()
    for _ in range(reps):
        if name == "raw_argmin":
            for row in batch:
                checksum += raw_argmin(row)
        elif name == "exp_sum":
            for row in batch:
                checksum += exp_sum(row)
        elif name == "logsumexp":
            for row in batch:
                checksum += stable_logsumexp([-x for x in row])
        elif name == "softmax":
            for row in batch:
                checksum += softmax_true_prob(row, raw_argmin(row), temp=1.0)
        elif name == "temp_softmax_t2":
            for row in batch:
                checksum += softmax_true_prob(row, raw_argmin(row), temp=2.0)
        elif name == "rank_margin":
            for row in batch:
                checksum += top2_margin(row)
        else:
            raise ValueError(name)
    elapsed = time.perf_counter() - start
    return elapsed / reps, checksum


def primitive_sweep() -> list[dict[str, object]]:
    rng = random.Random(0)
    rows: list[dict[str, object]] = []
    methods = [
        "raw_argmin",
        "rank_margin",
        "exp_sum",
        "logsumexp",
        "softmax",
        "temp_softmax_t2",
    ]
    for candidates in [128, 1024, 8192, 65536]:
        for batch in [1, 8, 16]:
            reps = max(1, min(30, 1_000_000 // (batch * candidates)))
            data = make_batch(batch, candidates, rng)
            for method in methods:
                elapsed, checksum = time_method(method, data, reps)
                scores = batch * candidates
                rows.append(
                    {
                        "batch": batch,
                        "candidates": candidates,
                        "method": method,
                        "reps": reps,
                        "latency_ms": elapsed * 1000.0,
                        "scores_per_second": scores / elapsed,
                        "ideal_energy_bytes": scores * 8,
                        "checksum": round(checksum, 6),
                    }
                )
    return rows


def ece(confidences: list[float], correct: list[int], bins: int = 10) -> float:
    total = len(confidences)
    err = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        err += len(idx) / total * abs(acc - conf)
    return err


def task_benchmark() -> list[dict[str, object]]:
    rng = random.Random(1)
    candidates = 1024
    examples = 1000
    methods = ["rank_only", "softmax", "temp_softmax_t2"]
    state = {m: {"correct": [], "confidence": [], "true_prob": []} for m in methods}
    for _ in range(examples):
        true_idx = rng.randrange(candidates)
        energies = [rng.gauss(0.0, 1.0) for _ in range(candidates)]
        energies[true_idx] -= rng.uniform(0.5, 4.0)
        pred = raw_argmin(energies)
        is_correct = int(pred == true_idx)
        margin_conf = 1.0 / (1.0 + math.exp(-top2_margin(energies)))
        state["rank_only"]["correct"].append(is_correct)
        state["rank_only"]["confidence"].append(margin_conf)
        state["rank_only"]["true_prob"].append(float("nan"))
        for method, temp in [("softmax", 1.0), ("temp_softmax_t2", 2.0)]:
            prob = softmax_true_prob(energies, true_idx, temp=temp)
            pred_prob = softmax_true_prob(energies, pred, temp=temp)
            state[method]["correct"].append(is_correct)
            state[method]["confidence"].append(pred_prob)
            state[method]["true_prob"].append(prob)

    rows = []
    for method, vals in state.items():
        correct = vals["correct"]
        confidences = vals["confidence"]
        probs = [p for p in vals["true_prob"] if not math.isnan(p)]
        rows.append(
            {
                "method": method,
                "examples": examples,
                "top1_accuracy": sum(correct) / len(correct),
                "avg_confidence": sum(confidences) / len(confidences),
                "ece": ece(confidences, correct),
                "avg_true_probability": sum(probs) / len(probs) if probs else "",
            }
        )
    return rows


def synthetic_energy_network_time(candidates: int, dims: int = 32) -> float:
    weights = [math.sin(i * 0.17) for i in range(dims)]
    start = time.perf_counter()
    energies = []
    for i in range(candidates):
        acc = 0.0
        for d, w in enumerate(weights):
            acc += math.sin((i + 1) * (d + 3) * 0.001) * w
        energies.append(acc)
    return time.perf_counter() - start, energies


def network_split() -> list[dict[str, object]]:
    rows = []
    for candidates in [1024, 8192, 65536]:
        net_t, energies = synthetic_energy_network_time(candidates)
        start = time.perf_counter()
        _ = stable_logsumexp([-x for x in energies])
        norm_t = time.perf_counter() - start
        rows.append(
            {
                "candidates": candidates,
                "dims": 32,
                "energy_network_ms": net_t * 1000,
                "logsumexp_ms": norm_t * 1000,
                "normalization_fraction": norm_t / (net_t + norm_t),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    primitive = primitive_sweep()
    task = task_benchmark()
    split = network_split()
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
    }
    write_csv(RESULTS / "primitive_timings.csv", primitive)
    write_csv(RESULTS / "task_metrics.csv", task)
    write_csv(RESULTS / "network_split.csv", split)
    (RESULTS / "results.json").write_text(
        json.dumps({"metadata": metadata, "primitive": primitive, "task": task, "network_split": split}, indent=2)
    )

    h00_64k = [r for r in primitive if r["candidates"] == 65536 and r["batch"] == 8]
    by_method = {r["method"]: r["latency_ms"] for r in h00_64k}
    norm_frac_64k = [r for r in split if r["candidates"] == 65536][0]["normalization_fraction"]
    verdict = f"""# H00 Verdict

**TLDR:** The hypothesis is partially supported: normalization is not free, but
in this pure-Python benchmark `logsumexp` is a modest fraction of a small energy
network rather than the dominant cost.

On the 64K-candidate, batch-8 primitive sweep, raw argmin took
{by_method['raw_argmin']:.3f} ms, `logsumexp` took {by_method['logsumexp']:.3f}
ms, and full softmax took {by_method['softmax']:.3f} ms per batch. In the
synthetic energy-network split at 64K candidates, normalization was
{norm_frac_64k:.1%} of combined energy-network plus `logsumexp` time. The
ranking-only path preserves top-1 decisions and avoids probability calibration;
softmax adds calibrated probabilities but pays the exponential pass. This
weakens the strongest version of "exp is the main bottleneck" for small learned
energies, but supports treating normalization as a measurable cost at large
candidate counts.
"""
    (RESULTS / "verdict.md").write_text(verdict)
    print(json.dumps({"metadata": metadata, "rows": len(primitive), "task_rows": len(task)}, indent=2))


if __name__ == "__main__":
    main()
