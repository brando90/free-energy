"""Probe 06 — Error compounding, the (1-e)^n test.

LeCun's argument: P(proof of length n compiles) decays as (1-e)^n because errors
are unrecoverable. Our alternative: a verifier lets us backtrack and resample,
so a 2-state recoverable-Markov chain (on-manifold / off-manifold with a
recovery probability) fits better, and a constant model may even suffice.

Without VeriBench we simulate two ground-truth processes -- one geometric, one
recoverable-Markov -- and fit three competing models to each. We score by
proper *binomial* log-likelihood at each length n (we have N_total trials and
k_n observed all-valid-up-to-n successes), so AIC behaves correctly under the
sample noise.

Control: the geometric simulator should be best fit by the geometric model,
and the recoverable-Markov simulator should NOT be best fit by geometric.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from probes._common import (
    ProbeResult,
    add_common_args,
    default_output_dir,
    gpu_name,
    resolve_device,
    seed_everything,
    write_result,
)


def simulate_geometric(num_seqs: int, max_len: int, e: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((num_seqs, max_len)) > e).astype(np.int8)


def simulate_recoverable_markov(
    num_seqs: int,
    max_len: int,
    e: float,
    recovery: float,
    rng: np.random.Generator,
) -> np.ndarray:
    bits = np.ones((num_seqs, max_len), dtype=np.int8)
    state = np.ones(num_seqs, dtype=np.int8)
    for t in range(max_len):
        bits[:, t] = state
        state = np.where(
            state == 1,
            (rng.random(num_seqs) > e).astype(np.int8),
            (rng.random(num_seqs) < recovery).astype(np.int8),
        )
    return bits


def k_successes_curve(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (k_n: int array length max_len, N_total)."""
    all_valid = np.cumprod(bits, axis=1)
    k = all_valid.sum(axis=0)
    return k, bits.shape[0]


def binom_logL(k: np.ndarray, N: int, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(k * np.log(p) + (N - k) * np.log(1 - p)))


def fit_constant(k: np.ndarray, N: int) -> Dict:
    p_hat = float(np.clip(np.mean(k / N), 1e-6, 1 - 1e-6))
    n = len(k)
    p_pred = np.full(n, p_hat)
    return {"p": p_hat, "logL": binom_logL(k, N, p_pred), "n_params": 1, "pred": p_pred.tolist()}


def fit_geometric(k: np.ndarray, N: int) -> Dict:
    """Fit P(n) = (1-e)^n. MLE: maximise sum [k_n n log(1-e) + (N-k_n) log(1-(1-e)^n)] over e in (0,1)."""
    ns = np.arange(1, len(k) + 1)
    best_e, best_logL = None, -float("inf")
    # grid + golden-section style refine
    for e in np.linspace(1e-3, 0.5, 100):
        p_pred = (1 - e) ** ns
        ll = binom_logL(k, N, p_pred)
        if ll > best_logL:
            best_e, best_logL = float(e), ll
    # local refine
    step = 0.02
    e = best_e
    for _ in range(60):
        improved = False
        for sign in (-1.0, 1.0):
            cand = float(np.clip(e + sign * step, 1e-4, 0.6))
            ll = binom_logL(k, N, (1 - cand) ** ns)
            if ll > best_logL + 1e-9:
                e, best_logL = cand, ll
                improved = True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
    p_pred = (1 - e) ** ns
    return {"e": float(e), "logL": float(best_logL), "n_params": 1, "pred": p_pred.tolist()}


def fit_recoverable_markov(k: np.ndarray, N: int) -> Dict:
    """Model: per-step survival probability s(t) = 1 - e + alpha * exp(-beta * t).

    Three params (e, alpha, beta). At alpha=0 this collapses to geometric, so the model is nested.
    We fit by binomial MLE with grid + coordinate descent.
    """
    ns = np.arange(1, len(k) + 1)

    def predict(params: np.ndarray) -> np.ndarray:
        e, alpha, beta = params
        e = float(np.clip(e, 1e-4, 0.6))
        alpha = float(np.clip(alpha, -0.5, 0.5))
        beta = float(np.clip(beta, 1e-3, 10.0))
        s = np.clip(1 - e + alpha * np.exp(-beta * ns), 1e-6, 1 - 1e-6)
        return np.exp(np.cumsum(np.log(s)))

    best = None
    for e0 in [0.02, 0.05, 0.1, 0.2, 0.3]:
        for a0 in [-0.1, 0.0, 0.05, 0.1, 0.2, 0.3]:
            for b0 in [0.05, 0.2, 1.0, 3.0]:
                params = np.array([e0, a0, b0], dtype=np.float64)
                pred = predict(params)
                ll = binom_logL(k, N, pred)
                if best is None or ll > best[1]:
                    best = (params, ll, pred)
    params, logL, pred = best
    step = np.array([0.02, 0.03, 0.1])
    for _ in range(200):
        improved = False
        for i in range(3):
            for sign in (-1.0, 1.0):
                trial = params.copy()
                trial[i] = trial[i] + sign * step[i]
                p_pred = predict(trial)
                ll = binom_logL(k, N, p_pred)
                if ll > logL + 1e-9:
                    params, logL, pred = trial, ll, p_pred
                    improved = True
        if not improved:
            step *= 0.5
            if step.max() < 1e-5:
                break
    return {"params": params.tolist(), "logL": float(logL), "n_params": 3, "pred": pred.tolist()}


def aic(logL: float, k: int) -> float:
    return float(2 * k - 2 * logL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 06: error compounding")
    add_common_args(parser)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--num-seqs", type=int, default=20000)
    parser.add_argument("--e-true", type=float, default=0.05)
    parser.add_argument("--recovery-true", type=float, default=0.4)
    args = parser.parse_args()

    if args.smoke:
        args.max_len = 40
        args.num_seqs = 4000

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_06", args.tag)

    result = ProbeResult(
        probe="probe_06_error_compounding",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    rng = np.random.default_rng(args.seed)
    bits_geom = simulate_geometric(args.num_seqs, args.max_len, args.e_true, rng)
    bits_mark = simulate_recoverable_markov(args.num_seqs, args.max_len, args.e_true, args.recovery_true, rng)

    k_geom, N = k_successes_curve(bits_geom)
    k_mark, _ = k_successes_curve(bits_mark)

    fits_geom = {
        "geometric": fit_geometric(k_geom, N),
        "constant": fit_constant(k_geom, N),
        "recoverable_markov": fit_recoverable_markov(k_geom, N),
    }
    fits_mark = {
        "geometric": fit_geometric(k_mark, N),
        "constant": fit_constant(k_mark, N),
        "recoverable_markov": fit_recoverable_markov(k_mark, N),
    }

    def winner_and_aics(fits: Dict[str, Dict]) -> Tuple[str, Dict[str, float]]:
        aics = {name: aic(v["logL"], v["n_params"]) for name, v in fits.items()}
        win = min(aics, key=aics.get)
        return win, aics

    win_geom, aic_geom = winner_and_aics(fits_geom)
    win_mark, aic_mark = winner_and_aics(fits_mark)

    control_passed = (win_geom == "geometric") and (win_mark != "geometric")

    result.control_passed = bool(control_passed)
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "max_len": args.max_len,
        "num_seqs": args.num_seqs,
        "e_true": args.e_true,
        "recovery_true": args.recovery_true,
        "k_curves": {
            "geometric_sim": k_geom.tolist(),
            "markov_sim": k_mark.tolist(),
        },
        "N_trials": N,
        "fits_on_geometric_sim": {
            k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in fits_geom.items()
        },
        "fits_on_markov_sim": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in fits_mark.items()},
        "aic_geometric_sim": aic_geom,
        "aic_markov_sim": aic_mark,
        "winner_on_geometric_sim": win_geom,
        "winner_on_markov_sim": win_mark,
    }
    result.notes = {
        "interpretation": (
            "On (1-e)^n data, geometric AIC should win (recoverable_markov's extra params are wasted)."
            " On recoverable-Markov data, geometric should LOSE -- either constant or recoverable_markov"
            " wins. This validates the AIC-based comparison for the real VeriBench setting."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_06] wrote {path}  control_passed={control_passed}  win_geom={win_geom}  win_markov={win_mark}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
