#!/usr/bin/env python3
"""Finite-support exact MLE vs MCMC-estimated EBM training for VeriBench.

This script is the controlled sanity check that the three-example ranking
pilot was not. It builds finite candidate pools for VeriBench tasks, trains
two energy models from the same pretrained transformer initialization, and
compares:

1. Exact finite-support MLE:
   L(t) = E_theta(t, y+) + log sum_{y in C_t} exp(-E_theta(t, y)).

2. MCMC-estimated EBM gradient:
   grad L(t) ~= grad E_theta(t, y+) - grad E_theta(t, y_mcmc),
   where y_mcmc is sampled by a persistent Metropolis-Hastings chain over the
   same finite candidate pool C_t.

Because the support is finite, exact MLE is available and MCMC should behave
similarly when the chain mixes. That is the point of this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pilot_ebm_ranking as pilot
import train_transformer_energy as train_energy


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "mle_mcmc_full"


@dataclass(frozen=True)
class TaskPool:
    task_id: str
    subset: str
    gold_path: str
    candidates: list[pilot.Candidate]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_task_pools(
    veribench_root: Path,
    subsets: list[str],
    max_tasks: int | None,
    max_generated_agents: int,
) -> list[TaskPool]:
    lean_root = veribench_root / "veribench_dataset" / "lean_src" / "veribench"
    pools: list[TaskPool] = []
    for subset in subsets:
        subset_dir = lean_root / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"Missing VeriBench subset: {subset_dir}")
        for gold_path in sorted(subset_dir.glob("*.lean")):
            task_id = f"{subset}/{gold_path.stem}"
            gold_text = gold_path.read_text(encoding="utf-8")
            candidates: list[pilot.Candidate] = [
                pilot.Candidate(task_id, "gold", "gold", 1, str(gold_path), gold_text)
            ]
            for agent_path in pilot.generated_agent_paths(veribench_root, gold_path, max_generated_agents):
                candidates.append(
                    pilot.Candidate(
                        task_id=task_id,
                        candidate_id=agent_path.stem,
                        kind="generated_agent",
                        label=0,
                        source_path=str(agent_path.resolve()),
                        text=agent_path.read_text(encoding="utf-8"),
                    )
                )
            candidates.extend(pilot.make_corruptions(task_id, gold_text))
            pools.append(TaskPool(task_id, subset, str(gold_path.resolve()), candidates))

    if max_tasks is not None:
        pools = pools[:max_tasks]
    return pools


def split_pools(pools: list[TaskPool], train_fraction: float, seed: int) -> tuple[list[TaskPool], list[TaskPool]]:
    shuffled = list(pools)
    random.Random(seed).shuffle(shuffled)
    train_size = max(1, int(round(len(shuffled) * train_fraction)))
    train_size = min(train_size, len(shuffled))
    return shuffled[:train_size], shuffled[train_size:]


def task_meta(pool: TaskPool) -> dict[str, Any]:
    return {
        "id": pool.task_id,
        "role": pool.subset,
        "why": "finite-support exact-MLE vs MCMC EBM sanity check",
    }


def candidate_text(candidate: pilot.Candidate, pool: TaskPool) -> str:
    return train_energy.candidate_input_text(candidate, task_meta(pool))


def energy_batch(model, tokenizer, texts: list[str], device: str, max_length: int):
    batch = train_energy.tokenize_batch(tokenizer, texts, device, max_length)
    return model(**batch)


def batched(items: list[TaskPool], size: int) -> Iterable[list[TaskPool]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def exact_mle_batch_loss(model, tokenizer, pools: list[TaskPool], device: str, max_length: int):
    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    losses = []
    for pool in pools:
        texts = [candidate_text(candidate, pool) for candidate in pool.candidates]
        energies = energy_batch(model, tokenizer, texts, device, max_length)
        gold_index = next(i for i, candidate in enumerate(pool.candidates) if candidate.label == 1)
        loss = energies[gold_index] + torch.logsumexp(-energies, dim=0)
        losses.append(loss)
    return torch.stack(losses).mean()


def all_candidate_energies(model, tokenizer, pool: TaskPool, device: str, max_length: int) -> list[float]:
    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    model.eval()
    with torch.no_grad():
        texts = [candidate_text(candidate, pool) for candidate in pool.candidates]
        energies = energy_batch(model, tokenizer, texts, device, max_length)
    return [float(value) for value in energies.detach().cpu()]


def mh_update_indices(
    model,
    tokenizer,
    pools: list[TaskPool],
    chain_state: dict[str, int],
    rng: random.Random,
    steps: int,
    device: str,
    max_length: int,
) -> None:
    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            texts: list[str] = []
            metadata: list[tuple[TaskPool, int, int]] = []
            for pool in pools:
                n = len(pool.candidates)
                cur = chain_state.setdefault(pool.task_id, rng.randrange(n))
                prop = rng.randrange(n)
                if n > 1:
                    while prop == cur:
                        prop = rng.randrange(n)
                texts.append(candidate_text(pool.candidates[cur], pool))
                texts.append(candidate_text(pool.candidates[prop], pool))
                metadata.append((pool, cur, prop))

            energies = energy_batch(model, tokenizer, texts, device, max_length).detach().cpu()
            for i, (pool, cur, prop) in enumerate(metadata):
                e_cur = float(energies[2 * i])
                e_prop = float(energies[2 * i + 1])
                log_accept = -e_prop + e_cur
                if math.log(rng.random()) < min(0.0, log_accept):
                    chain_state[pool.task_id] = prop
                else:
                    chain_state[pool.task_id] = cur


def mcmc_gradient_batch_loss(
    model,
    tokenizer,
    pools: list[TaskPool],
    chain_state: dict[str, int],
    rng: random.Random,
    mcmc_steps: int,
    device: str,
    max_length: int,
):
    mh_update_indices(model, tokenizer, pools, chain_state, rng, mcmc_steps, device, max_length)
    pos_texts = []
    neg_texts = []
    for pool in pools:
        gold = next(candidate for candidate in pool.candidates if candidate.label == 1)
        neg = pool.candidates[chain_state[pool.task_id]]
        pos_texts.append(candidate_text(gold, pool))
        neg_texts.append(candidate_text(neg, pool))
    e_pos = energy_batch(model, tokenizer, pos_texts, device, max_length)
    e_neg = energy_batch(model, tokenizer, neg_texts, device, max_length)
    return (e_pos - e_neg).mean()


def evaluate_model(model, tokenizer, pools: list[TaskPool], device: str, max_length: int) -> dict[str, Any]:
    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    task_rows = []
    nlls = []
    reciprocal_ranks = []
    gold_top = 0
    model.eval()
    with torch.no_grad():
        for pool in pools:
            energies = all_candidate_energies(model, tokenizer, pool, device, max_length)
            ranked_indices = sorted(range(len(pool.candidates)), key=lambda idx: energies[idx])
            gold_index = next(i for i, candidate in enumerate(pool.candidates) if candidate.label == 1)
            gold_rank = ranked_indices.index(gold_index) + 1
            top = pool.candidates[ranked_indices[0]]
            energy_tensor = torch.tensor(energies)
            nll = float(energy_tensor[gold_index] + torch.logsumexp(-energy_tensor, dim=0))
            nlls.append(nll)
            reciprocal_ranks.append(1.0 / gold_rank)
            gold_top += int(gold_rank == 1)
            task_rows.append(
                {
                    "task_id": pool.task_id,
                    "subset": pool.subset,
                    "num_candidates": len(pool.candidates),
                    "gold_rank": gold_rank,
                    "gold_energy": energies[gold_index],
                    "top_candidate": top.candidate_id,
                    "top_kind": top.kind,
                    "top_energy": energies[ranked_indices[0]],
                    "nll": nll,
                }
            )
    denom = max(1, len(pools))
    return {
        "num_tasks": len(pools),
        "top1": gold_top / denom,
        "mrr": sum(reciprocal_ranks) / denom,
        "mean_nll": sum(nlls) / denom,
        "tasks": task_rows,
    }


def make_model_and_tokenizer(model_name: str, seed: int, device: str):
    _torch, _nn, _F, _AutoModel, AutoTokenizer = train_energy.import_training_deps()
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = train_energy.build_energy_model(model_name).to(device)
    return model, tokenizer


def train_exact(args, train_pools: list[TaskPool], test_pools: list[TaskPool], device: str) -> dict[str, Any]:
    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    model, tokenizer = make_model_and_tokenizer(args.model_name, args.seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in batched(train_pools, args.task_batch_size):
            loss = exact_mle_batch_loss(model, tokenizer, batch, device, args.max_length)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        train_eval = evaluate_model(model, tokenizer, train_pools, device, args.max_length)
        test_eval = evaluate_model(model, tokenizer, test_pools, device, args.max_length) if test_pools else None
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, count),
            "train_eval": summarize_eval(train_eval),
            "test_eval": summarize_eval(test_eval) if test_eval else None,
        }
        history.append(record)
        print(
            f"exact epoch={epoch} loss={record['train_loss']:.4f} "
            f"train_mrr={record['train_eval']['mrr']:.4f} "
            f"test_mrr={(record['test_eval'] or {}).get('mrr', float('nan')):.4f}"
        )
    return {
        "method": "exact_finite_support_mle",
        "seconds": time.perf_counter() - start_time,
        "history": history,
        "final_train": evaluate_model(model, tokenizer, train_pools, device, args.max_length),
        "final_test": evaluate_model(model, tokenizer, test_pools, device, args.max_length) if test_pools else None,
    }


def train_mcmc(args, train_pools: list[TaskPool], test_pools: list[TaskPool], device: str) -> dict[str, Any]:
    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    model, tokenizer = make_model_and_tokenizer(args.model_name, args.seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    chain_state = {
        pool.task_id: rng.randrange(len(pool.candidates))
        for pool in train_pools
    }
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in batched(train_pools, args.task_batch_size):
            loss = mcmc_gradient_batch_loss(
                model,
                tokenizer,
                batch,
                chain_state,
                rng,
                args.mcmc_steps,
                device,
                args.max_length,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        train_eval = evaluate_model(model, tokenizer, train_pools, device, args.max_length)
        test_eval = evaluate_model(model, tokenizer, test_pools, device, args.max_length) if test_pools else None
        record = {
            "epoch": epoch,
            "surrogate_loss": total / max(1, count),
            "train_eval": summarize_eval(train_eval),
            "test_eval": summarize_eval(test_eval) if test_eval else None,
        }
        history.append(record)
        print(
            f"mcmc epoch={epoch} surrogate={record['surrogate_loss']:.4f} "
            f"train_mrr={record['train_eval']['mrr']:.4f} "
            f"test_mrr={(record['test_eval'] or {}).get('mrr', float('nan')):.4f}"
        )
    return {
        "method": "persistent_mh_mcmc_gradient",
        "mcmc_steps": args.mcmc_steps,
        "seconds": time.perf_counter() - start_time,
        "history": history,
        "final_train": evaluate_model(model, tokenizer, train_pools, device, args.max_length),
        "final_test": evaluate_model(model, tokenizer, test_pools, device, args.max_length) if test_pools else None,
    }


def summarize_eval(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "num_tasks": report["num_tasks"],
        "top1": report["top1"],
        "mrr": report["mrr"],
        "mean_nll": report["mean_nll"],
    }


def count_candidates(pools: list[TaskPool]) -> dict[str, Any]:
    counts = [len(pool.candidates) for pool in pools]
    by_subset: dict[str, int] = {}
    for pool in pools:
        by_subset[pool.subset] = by_subset.get(pool.subset, 0) + 1
    return {
        "num_tasks": len(pools),
        "num_candidates": sum(counts),
        "min_candidates_per_task": min(counts) if counts else 0,
        "max_candidates_per_task": max(counts) if counts else 0,
        "by_subset": by_subset,
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in vars(args).items():
        serializable[key] = str(value) if isinstance(value, Path) else value
    return serializable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veribench-root", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--subsets", nargs="+", default=["easy_set", "cs_set", "humaneval_set"])
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-generated-agents", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--task-batch-size", type=int, default=4)
    parser.add_argument("--mcmc-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    torch, _nn, _F, _AutoModel, _AutoTokenizer = train_energy.import_training_deps()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    veribench_root = Path(args.veribench_root).expanduser().resolve()
    pools = discover_task_pools(veribench_root, args.subsets, args.max_tasks, args.max_generated_agents)
    if not pools:
        raise ValueError("No task pools discovered")
    train_pools, test_pools = split_pools(pools, args.train_fraction, args.seed)

    print(
        f"dataset all={count_candidates(pools)} train={count_candidates(train_pools)} "
        f"test={count_candidates(test_pools)}"
    )
    print(f"model={args.model_name} device={device} epochs={args.epochs}")

    exact = train_exact(args, train_pools, test_pools, device)
    mcmc = train_mcmc(args, train_pools, test_pools, device)

    report = {
        "args": jsonable_args(args),
        "device": device,
        "veribench_root": str(veribench_root),
        "dataset": {
            "all": count_candidates(pools),
            "train": count_candidates(train_pools),
            "test": count_candidates(test_pools),
        },
        "exact": exact,
        "mcmc": mcmc,
        "comparison": {
            "train": {
                "exact": summarize_eval(exact["final_train"]),
                "mcmc": summarize_eval(mcmc["final_train"]),
            },
            "test": {
                "exact": summarize_eval(exact["final_test"]),
                "mcmc": summarize_eval(mcmc["final_test"]),
            },
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "mle_mcmc_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
