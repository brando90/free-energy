"""Toy control for LeCun's `(1-e)^T` argument.

The point is to make the assumptions explicit:
- blind rollout + independent unrecoverable errors should fit a geometric curve;
- verifier resampling and recoverable errors should not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def blind_rollout(num_trials: int, max_len: int, e: float, rng: np.random.Generator) -> np.ndarray:
    valid_steps = rng.random((num_trials, max_len)) > e
    return np.cumprod(valid_steps, axis=1).mean(axis=0)


def verifier_resampling(
    num_trials: int,
    max_len: int,
    e: float,
    retries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # A step fails only if every proposal in the retry budget is invalid.
    proposals = rng.random((num_trials, max_len, retries + 1)) > e
    accepted_step = proposals.any(axis=2)
    return np.cumprod(accepted_step, axis=1).mean(axis=0)


def recoverable_process(
    num_trials: int,
    max_len: int,
    e: float,
    recovery: float,
    rng: np.random.Generator,
) -> np.ndarray:
    state = np.ones(num_trials, dtype=bool)
    success_by_t = []
    for _ in range(max_len):
        leave = rng.random(num_trials) < e
        recover = rng.random(num_trials) < recovery
        state = np.where(state, ~leave, recover)
        success_by_t.append(state.mean())
    return np.array(success_by_t)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--e", type=float, default=0.03)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--recovery", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="toy/results")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    lengths = np.arange(1, args.max_len + 1)
    geometric = (1 - args.e) ** lengths
    blind = blind_rollout(args.num_trials, args.max_len, args.e, rng)
    resample = verifier_resampling(args.num_trials, args.max_len, args.e, args.retries, rng)
    recoverable = recoverable_process(args.num_trials, args.max_len, args.e, args.recovery, rng)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_trials": args.num_trials,
        "max_len": args.max_len,
        "e": args.e,
        "retries": args.retries,
        "recovery": args.recovery,
        "seed": args.seed,
        "success_at_max_len": {
            "geometric_prediction": float(geometric[-1]),
            "blind_rollout": float(blind[-1]),
            "verifier_resampling": float(resample[-1]),
            "recoverable_process": float(recoverable[-1]),
        },
    }
    (out_dir / "toy_error_process_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(lengths, geometric, label="(1-e)^T prediction", linewidth=2)
    plt.plot(lengths, blind, "--", label="blind AR simulation", linewidth=2)
    plt.plot(lengths, resample, label=f"verifier resampling, retries={args.retries}", linewidth=2)
    plt.plot(lengths, recoverable, label=f"recoverable process, r={args.recovery}", linewidth=2)
    plt.xlabel("sequence length T")
    plt.ylabel("P(success at T)")
    plt.title("LeCun error-compounding premise vs feedback/recovery")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "toy_error_process.png", dpi=180)
    print(f"[toy] wrote {out_dir / 'toy_error_process_summary.json'}")
    print(f"[toy] wrote {out_dir / 'toy_error_process.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
