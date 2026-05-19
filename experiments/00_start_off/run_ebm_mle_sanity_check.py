#!/usr/bin/env python3
"""Gradient-level sanity check: EBM negative phase equals finite-support MLE.

For a finite candidate pool C_t per task, the conditional EBM is

    p_theta(y | t) = exp(-E_theta(t, y)) / sum_{z in C_t} exp(-E_theta(t, z)).

The exact finite-support MLE loss is

    E_theta(t, y+) + logsumexp_{y in C_t}(-E_theta(t, y)).

Its gradient is the positive/data energy gradient minus a model expectation.
This script checks that model-sample gradients match the exact MLE gradient:

1. exact finite-support MLE gradient;
2. exact EBM negative-phase gradient using the finite model expectation;
3. exact samples from the finite EBM distribution;
4. Metropolis-Hastings samples with different chain lengths.

The expected result is that the exact EBM negative phase matches MLE up to
floating-point noise, exact model samples converge to that gradient as sample
count grows, and MCMC samples get closer as the chain mixes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import run_mle_mcmc_experiment as exp


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "ebm_mle_sanity"

HYPOTHESIS = (
    "MCMC-based EBM training is normally expensive because the model "
    "expectation is hard to sample exactly, but for Lean/VeriBench it may "
    "still be worth doing with finite candidate pools, short-run MCMC, or "
    "biased negative samplers if the resulting gradient is close enough to "
    "MLE to learn a useful verifier/proof ranker."
)


def flatten_grads(model, torch):
    chunks = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            grad = torch.zeros_like(param)
        chunks.append(grad.detach().cpu().float().reshape(-1))
    return torch.cat(chunks)


def gradient_metrics(reference, estimate, torch) -> dict[str, float]:
    reference = reference.double()
    estimate = estimate.double()
    diff = estimate - reference
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    est_norm = float(torch.linalg.vector_norm(estimate).item())
    diff_norm = float(torch.linalg.vector_norm(diff).item())
    dot = float(torch.dot(reference, estimate).item())
    denom = max(ref_norm * est_norm, 1e-30)
    return {
        "cosine": dot / denom,
        "relative_l2": diff_norm / max(ref_norm, 1e-30),
        "norm_ratio": est_norm / max(ref_norm, 1e-30),
        "dot_over_reference_norm_sq": dot / max(ref_norm * ref_norm, 1e-30),
        "reference_norm": ref_norm,
        "estimate_norm": est_norm,
        "difference_norm": diff_norm,
    }


def gradient_norm(gradient, torch) -> float:
    return float(torch.linalg.vector_norm(gradient.double()).item())


def compute_exact_mle_gradient(model, tokenizer, pools, device: str, max_length: int, batch_size: int):
    torch, _nn, _F, _AutoModel, _AutoTokenizer = exp.train_energy.import_training_deps()
    model.zero_grad(set_to_none=True)
    model.eval()
    total = len(pools)
    loss_value = 0.0
    for batch in exp.batched(pools, batch_size):
        loss = exp.exact_mle_batch_loss(model, tokenizer, batch, device, max_length)
        weight = len(batch) / max(1, total)
        (loss * weight).backward()
        loss_value += float(loss.item()) * weight
    return flatten_grads(model, torch), loss_value


def compute_exact_ebm_negative_phase_gradient(
    model,
    tokenizer,
    pools,
    device: str,
    max_length: int,
    batch_size: int,
):
    torch, _nn, _F, _AutoModel, _AutoTokenizer = exp.train_energy.import_training_deps()
    model.zero_grad(set_to_none=True)
    model.eval()
    total = len(pools)
    surrogate_value = 0.0
    for batch in exp.batched(pools, batch_size):
        for pool in batch:
            texts = [exp.candidate_text(candidate, pool) for candidate in pool.candidates]
            energies = exp.energy_batch(model, tokenizer, texts, device, max_length)
            gold_index = next(i for i, candidate in enumerate(pool.candidates) if candidate.label == 1)
            weights = torch.softmax(-energies.detach(), dim=0)
            surrogate = energies[gold_index] - torch.sum(weights * energies)
            (surrogate / max(1, total)).backward()
            surrogate_value += float(surrogate.detach().cpu().item()) / max(1, total)
    return flatten_grads(model, torch), surrogate_value


def candidate_energy_cache(model, tokenizer, pools, device: str, max_length: int) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    for pool in pools:
        cache[pool.task_id] = exp.all_candidate_energies(model, tokenizer, pool, device, max_length)
    return cache


def exact_model_sample_indices(
    pools,
    energy_cache: dict[str, list[float]],
    samples_per_task: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for pool in pools:
        energies = energy_cache[pool.task_id]
        weights = [math.exp(-energy) for energy in energies]
        choices = list(range(len(energies)))
        indices[pool.task_id] = rng.choices(choices, weights=weights, k=samples_per_task)
    return indices


def mcmc_sample_indices(
    pools,
    energy_cache: dict[str, list[float]],
    samples_per_task: int,
    steps: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for pool in pools:
        energies = energy_cache[pool.task_id]
        n = len(energies)
        task_indices = []
        for _ in range(samples_per_task):
            cur = rng.randrange(n)
            for _step in range(steps):
                prop = rng.randrange(n)
                if n > 1:
                    while prop == cur:
                        prop = rng.randrange(n)
                log_accept = -energies[prop] + energies[cur]
                if math.log(rng.random()) < min(0.0, log_accept):
                    cur = prop
            task_indices.append(cur)
        indices[pool.task_id] = task_indices
    return indices


def compute_sample_gradient(
    model,
    tokenizer,
    pools,
    sample_indices: dict[str, list[int]],
    device: str,
    max_length: int,
    task_batch_size: int,
    negative_batch_size: int,
):
    torch, _nn, _F, _AutoModel, _AutoTokenizer = exp.train_energy.import_training_deps()
    model.zero_grad(set_to_none=True)
    model.eval()
    total_tasks = len(pools)
    samples_per_task = len(next(iter(sample_indices.values())))
    surrogate_loss = 0.0

    for batch in exp.batched(pools, task_batch_size):
        pos_texts = []
        neg_texts = []
        for pool in batch:
            gold = next(candidate for candidate in pool.candidates if candidate.label == 1)
            pos_texts.append(exp.candidate_text(gold, pool))
            for idx in sample_indices[pool.task_id]:
                neg_texts.append(exp.candidate_text(pool.candidates[idx], pool))

        e_pos = exp.energy_batch(model, tokenizer, pos_texts, device, max_length)
        pos_scale = 1.0 / max(1, total_tasks)
        (e_pos.sum() * pos_scale).backward()
        surrogate_loss += float(e_pos.detach().sum().cpu().item()) * pos_scale

        neg_denom = max(1, total_tasks * samples_per_task)
        for start in range(0, len(neg_texts), negative_batch_size):
            chunk = neg_texts[start : start + negative_batch_size]
            e_neg = exp.energy_batch(model, tokenizer, chunk, device, max_length)
            (-(e_neg.sum() / neg_denom)).backward()
            surrogate_loss -= float(e_neg.detach().sum().cpu().item()) / neg_denom

    return flatten_grads(model, torch), surrogate_loss


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veribench-root", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--subsets", nargs="+", default=["easy_set", "cs_set", "humaneval_set"])
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-gradient-tasks", type=int, default=None)
    parser.add_argument("--max-generated-agents", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--task-batch-size", type=int, default=4)
    parser.add_argument("--negative-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--exact-sample-counts", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--mcmc-steps-list", nargs="+", type=int, default=[0, 1, 5, 25, 100])
    parser.add_argument("--mcmc-samples", type=int, default=64)
    args = parser.parse_args()

    torch, _nn, _F, _AutoModel, _AutoTokenizer = exp.train_energy.import_training_deps()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    exp.set_seed(args.seed)
    veribench_root = Path(args.veribench_root).expanduser().resolve()
    pools = exp.discover_task_pools(veribench_root, args.subsets, args.max_tasks, args.max_generated_agents)
    train_pools, test_pools = exp.split_pools(pools, args.train_fraction, args.seed)
    gradient_pools = train_pools[: args.max_gradient_tasks] if args.max_gradient_tasks else train_pools
    if not gradient_pools:
        raise ValueError("No pools selected for gradient sanity check")

    print(
        f"dataset all={exp.count_candidates(pools)} train={exp.count_candidates(train_pools)} "
        f"test={exp.count_candidates(test_pools)} gradient={exp.count_candidates(gradient_pools)}"
    )
    print(f"model={args.model_name} device={device}")
    print(f"hypothesis={HYPOTHESIS}")

    model, tokenizer = exp.make_model_and_tokenizer(args.model_name, args.seed, device)

    started = time.perf_counter()
    exact_grad, exact_loss = compute_exact_mle_gradient(
        model, tokenizer, gradient_pools, device, args.max_length, args.task_batch_size
    )
    print(f"exact_mle_gradient loss={exact_loss:.6f} norm={gradient_norm(exact_grad, torch):.6f}")

    exact_ebm_grad, exact_ebm_surrogate = compute_exact_ebm_negative_phase_gradient(
        model, tokenizer, gradient_pools, device, args.max_length, args.task_batch_size
    )
    exact_ebm_metrics = gradient_metrics(exact_grad, exact_ebm_grad, torch)
    print(
        f"exact_ebm_negative_phase surrogate={exact_ebm_surrogate:.6f} "
        f"cosine={exact_ebm_metrics['cosine']:.9f} rel_l2={exact_ebm_metrics['relative_l2']:.9f}"
    )

    energy_cache = candidate_energy_cache(model, tokenizer, gradient_pools, device, args.max_length)
    rng = random.Random(args.seed)

    exact_sample_reports = []
    for sample_count in args.exact_sample_counts:
        sample_indices = exact_model_sample_indices(gradient_pools, energy_cache, sample_count, rng)
        grad, surrogate = compute_sample_gradient(
            model,
            tokenizer,
            gradient_pools,
            sample_indices,
            device,
            args.max_length,
            args.task_batch_size,
            args.negative_batch_size,
        )
        metrics = gradient_metrics(exact_grad, grad, torch)
        record = {
            "estimator": "exact_model_samples",
            "samples_per_task": sample_count,
            "surrogate_loss": surrogate,
            **metrics,
        }
        exact_sample_reports.append(record)
        print(
            f"exact_samples k={sample_count} cosine={record['cosine']:.6f} "
            f"rel_l2={record['relative_l2']:.6f} norm_ratio={record['norm_ratio']:.6f}"
        )

    mcmc_reports = []
    for steps in args.mcmc_steps_list:
        sample_indices = mcmc_sample_indices(gradient_pools, energy_cache, args.mcmc_samples, steps, rng)
        grad, surrogate = compute_sample_gradient(
            model,
            tokenizer,
            gradient_pools,
            sample_indices,
            device,
            args.max_length,
            args.task_batch_size,
            args.negative_batch_size,
        )
        metrics = gradient_metrics(exact_grad, grad, torch)
        record = {
            "estimator": "finite_support_mh_mcmc",
            "steps": steps,
            "samples_per_task": args.mcmc_samples,
            "surrogate_loss": surrogate,
            **metrics,
        }
        mcmc_reports.append(record)
        print(
            f"mcmc steps={steps} k={args.mcmc_samples} cosine={record['cosine']:.6f} "
            f"rel_l2={record['relative_l2']:.6f} norm_ratio={record['norm_ratio']:.6f}"
        )

    report = {
        "args": jsonable_args(args),
        "hypothesis": HYPOTHESIS,
        "device": device,
        "veribench_root": str(veribench_root),
        "dataset": {
            "all": exp.count_candidates(pools),
            "train": exp.count_candidates(train_pools),
            "test": exp.count_candidates(test_pools),
            "gradient": exp.count_candidates(gradient_pools),
        },
        "exact_mle": {
            "loss": exact_loss,
            "gradient_norm": gradient_norm(exact_grad, torch),
        },
        "exact_ebm_negative_phase": {
            "surrogate_loss": exact_ebm_surrogate,
            **exact_ebm_metrics,
        },
        "exact_model_samples": exact_sample_reports,
        "mcmc": mcmc_reports,
        "seconds": time.perf_counter() - started,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "ebm_mle_sanity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
