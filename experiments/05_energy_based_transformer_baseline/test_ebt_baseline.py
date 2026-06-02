from __future__ import annotations

from pathlib import Path

import torch

import run_ebt_toy as toy
import run_veribench_ebt_ranking as veribench


def test_toy_target_shapes() -> None:
    context, target = toy.make_dataset(num_examples=5, seq_len=4, seed=0)
    assert context.shape == (5, 4)
    assert target.shape == (5, 4)
    assert set(context.flatten().tolist()) <= {0, 1}
    assert set(target.flatten().tolist()) <= {0, 1}


def test_toy_ebt_refine_shapes() -> None:
    model = toy.EnergyBasedTransformer(
        seq_len=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        max_steps=2,
    )
    context = torch.randint(0, 2, (3, 4))
    logits, energies = model.refine(
        context,
        num_steps=2,
        alpha=1.0,
        init_scale=0.0,
        create_graph=True,
        detach_between_steps=False,
        grad_clip=1.0,
    )
    assert logits.shape == (3, 4, 2)
    assert len(energies) == 3
    assert all(energy.shape == (3,) for energy in energies)


def test_char_vocab_and_energy_shapes() -> None:
    texts = ["TASK_ID: a\nLEAN_CANDIDATE:\ntheorem x : True := by trivial"]
    vocab = veribench.CharVocab(texts, max_vocab_size=64)
    ids, mask = veribench.encode_texts(vocab, texts, max_length=32, device=torch.device("cpu"))
    model = veribench.CharEnergyTransformer(
        vocab_size=len(vocab.itos),
        max_length=32,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    energies = model(ids, mask)
    assert ids.shape == (1, 32)
    assert mask.shape == (1, 32)
    assert energies.shape == (1,)
    assert torch.isfinite(energies).all()


def test_toy_smoke_run(tmp_path: Path) -> None:
    parser = toy.build_arg_parser()
    args = parser.parse_args(
        [
            "--tag",
            "pytest",
            "--output-dir",
            str(tmp_path),
            "--seq-len",
            "4",
            "--num-train",
            "64",
            "--num-test",
            "32",
            "--hidden-dim",
            "16",
            "--num-layers",
            "1",
            "--num-heads",
            "4",
            "--batch-size",
            "16",
            "--eval-batch-size",
            "32",
            "--direct-epochs",
            "3",
            "--ebt-epochs",
            "3",
            "--ebt-steps",
            "1",
            "--eval-steps",
            "1",
            "--alpha",
            "1.0",
            "--init-scale",
            "0.0",
            "--device",
            "cpu",
        ]
    )
    report = toy.run_experiment(args)
    assert "energy_based_transformer" in report["results"]
    assert (tmp_path / "pytest_toy_report.json").exists()

