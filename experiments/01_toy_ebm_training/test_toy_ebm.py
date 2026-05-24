from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import run_toy_ebm as toy


def test_enumerate_binary_support_shape_and_values() -> None:
    support = toy.enumerate_binary_support(seq_len=4)
    assert support.shape == (16, 4)
    assert support.dtype == torch.long
    assert set(support.flatten().tolist()) == {0, 1}
    assert len({tuple(row.tolist()) for row in support}) == 16


def test_target_distribution_normalizes() -> None:
    support = toy.enumerate_binary_support(seq_len=5)
    tasks = toy.make_task_bank(num_tasks=3, seq_len=5, seed=0)
    log_probs = toy.target_log_probs(tasks, support, temperature=1.0)
    assert log_probs.shape == (3, 32)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.isfinite(log_probs).all()


@pytest.mark.parametrize("model_name", toy.MODEL_NAMES)
def test_model_forward_shapes(model_name: str) -> None:
    seq_len = 6
    model = toy.build_model(model_name, seq_len=seq_len, hidden_dim=16, num_layers=1, num_heads=4)
    tasks = torch.randint(0, 2, (7, seq_len))
    candidates = torch.randint(0, 2, (7, seq_len))
    energies = model(tasks, candidates)
    assert energies.shape == (7,)
    assert torch.isfinite(energies).all()


def test_exact_training_improves_over_uniform(tmp_path: Path) -> None:
    parser = toy.build_arg_parser()
    args = parser.parse_args(
        [
            "--tag",
            "pytest_smoke",
            "--output-dir",
            str(tmp_path),
            "--models",
            "mlp",
            "--seq-len",
            "6",
            "--num-train-tasks",
            "12",
            "--num-test-tasks",
            "4",
            "--epochs",
            "20",
            "--batch-size",
            "4",
            "--pair-batch-size",
            "512",
            "--hidden-dim",
            "32",
            "--lr",
            "0.01",
            "--device",
            "cpu",
            "--require-improvement",
        ]
    )
    report = toy.run_experiment(args)
    baseline_kl = report["baseline"]["test"]["kl_pstar_model"]
    model_kl = report["results"][0]["test"]["kl_pstar_model"]
    assert math.isfinite(model_kl)
    assert model_kl < baseline_kl
    assert (tmp_path / "pytest_smoke_report.json").exists()

