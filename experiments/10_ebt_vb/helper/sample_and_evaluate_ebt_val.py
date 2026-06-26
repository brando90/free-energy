#!/usr/bin/env python3
"""Sample EBT checkpoints on VeriBench val and score IC/D metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from ebt import GoedelVocabEBT
from veribench_embedding_dataloader import VeriBenchEmbeddingDataset, collate_veribench_embedding_samples
from veribench_task import VeriBenchTask


VALIDATION_SET_OPTION = "set_option linter.unusedVariables false"


@dataclass(frozen=True)
class ScoreRow:
    run_name: str
    task_name: str
    split: str
    family: str
    pred_tokens: int
    gold_tokens: int
    token_matches: int
    token_accuracy: float
    exact_tokens: bool
    candidate_len: int
    ic1: float
    ic2: float
    te1: float
    d1: float
    d2: float
    s_tilde: float
    compile_candidate_success: bool
    compile_gold_success: bool
    skip_reason: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if checkpoints:
        return checkpoints[-1]
    final = run_dir / "checkpoint_final.pt"
    if final.exists():
        return final
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def load_config(run_dir: Path) -> Any:
    return OmegaConf.load(run_dir / "config_resolved.yaml")


def build_model(cfg: Any, dataset: VeriBenchEmbeddingDataset, device: torch.device) -> GoedelVocabEBT:
    model = GoedelVocabEBT(
        model_name=str(cfg.model.model_name),
        revision=cfg.model.revision,
        vocab_size=int(dataset.vocab["vocab_size"]),
        pad_token_id=int(dataset.vocab["local_pad_id"]),
        context_dim=int(cfg.model.context_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dim_feedforward=int(cfg.model.dim_feedforward),
        dropout=float(cfg.model.dropout),
        mcmc_num_steps=int(cfg.model.mcmc_num_steps),
        mcmc_step_size=float(cfg.model.mcmc_step_size),
        mcmc_step_size_learnable=bool(cfg.model.mcmc_step_size_learnable),
        gaussian_random_noise_scaling=float(cfg.model.gaussian_random_noise_scaling),
        denoising_initial_condition=str(cfg.model.denoising_initial_condition),
        normalize_initial_condition=bool(cfg.model.normalize_initial_condition),
        normalize_initial_condition_only_first_step=bool(cfg.model.normalize_initial_condition_only_first_step),
        langevin_dynamics_noise=float(cfg.model.langevin_dynamics_noise),
        truncate_mcmc=bool(cfg.model.truncate_mcmc),
        no_mcmc_detach=bool(cfg.model.no_mcmc_detach),
        clamp_futures_grad=bool(cfg.model.clamp_futures_grad),
        clamp_futures_grad_max_change=float(cfg.model.clamp_futures_grad_max_change),
        absolute_clamp=float(cfg.model.absolute_clamp),
        sharpen_predicted_distribution=float(cfg.model.sharpen_predicted_distribution),
        norm_pred=bool(cfg.model.norm_pred),
        norm_pred_not_final_step=bool(cfg.model.norm_pred_not_final_step),
        reconstruction_coeff=float(cfg.model.reconstruction_coeff),
        soften_target_prob_dist=float(cfg.model.soften_target_prob_dist),
        loss_on_final_step_only=bool(cfg.model.loss_on_final_step_only),
        use_context_activations=bool(cfg.model.use_context_activations),
        max_task_embeddings=int(cfg.model.max_task_embeddings),
        max_target_positions=int(cfg.model.max_target_positions),
    ).to(device)
    return model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def trim_special(local_ids: list[int], *, eos_id: int, pad_id: int, bos_id: int) -> list[int]:
    out: list[int] = []
    for token_id in local_ids:
        token_id = int(token_id)
        if token_id == eos_id:
            break
        if token_id in {pad_id, bos_id}:
            continue
        out.append(token_id)
    return out


def decode_local_ids(dataset: VeriBenchEmbeddingDataset, tokenizer: AutoTokenizer, local_ids: list[int]) -> str:
    if not local_ids:
        return ""
    local = torch.tensor(local_ids, dtype=torch.long)
    original = dataset.id_mapper.original_ids(local).tolist()
    return tokenizer.decode([int(x) for x in original], skip_special_tokens=True)


def validation_header(source: str) -> str:
    imports = ["import Std"]
    seen = {"import Std"}
    body: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            if stripped not in seen:
                imports.append(stripped)
                seen.add(stripped)
            continue
        if stripped == VALIDATION_SET_OPTION:
            continue
        body.append(line)
    return "\n".join(imports + [VALIDATION_SET_OPTION] + body).strip() + "\n"


def sample_run(
    *,
    run_name: str,
    run_dir: Path,
    checkpoint_path: Path,
    out_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    max_rows: int | None,
) -> Path:
    cfg = load_config(run_dir)
    data_dir = Path(cfg.data.data_dir)
    dataset = VeriBenchEmbeddingDataset(
        data_dir=data_dir,
        split=split,
        max_items=max_rows,
        max_target_tokens=cfg.data.max_target_tokens,
        activation_dtype=str(cfg.data.activation_dtype),
        model_name=cfg.data.model_name or cfg.model.model_name,
        model_revision=cfg.data.model_revision or cfg.model.revision,
        validate_context=bool(cfg.data.validate_context),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_veribench_embedding_samples,
    )
    model = build_model(cfg, dataset, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(dataset.model_name, trust_remote_code=True)
    bos_id = int(dataset.vocab["local_bos_id"])
    eos_id = int(dataset.vocab["local_eos_id"])
    pad_id = int(dataset.vocab["local_pad_id"])

    output_path = out_dir / run_name / "samples.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in tqdm(loader, desc=f"sampling:{run_name}", unit="batch"):
            batch = move_batch(batch, device)
            target_len = int(batch["labels"].shape[1])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                with torch.enable_grad():
                    sampled = model.sample(
                        batch["context_activations"],
                        target_len=target_len,
                        context_attention_mask=batch["context_attention_mask"],
                        task_indices=batch["task_index"],
                    )
            pred_batch = sampled["token_ids"].detach().cpu()
            labels = batch["labels"].detach().cpu()
            masks = batch["label_attention_mask"].detach().cpu().bool()
            for i, task_name in enumerate(batch["task_name"]):
                pred_ids = trim_special(pred_batch[i].tolist(), eos_id=eos_id, pad_id=pad_id, bos_id=bos_id)
                gold_ids = trim_special(labels[i][masks[i]].tolist(), eos_id=eos_id, pad_id=pad_id, bos_id=bos_id)
                compare_len = min(len(pred_ids), len(gold_ids))
                matches = sum(int(a == b) for a, b in zip(pred_ids[:compare_len], gold_ids[:compare_len]))
                generation = validation_header(decode_local_ids(dataset, tokenizer, pred_ids))
                gold = validation_header(decode_local_ids(dataset, tokenizer, gold_ids))
                row = {
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint_path),
                    "task_name": task_name,
                    "split": batch["split"][i],
                    "family": batch["family"][i],
                    "pred_tokens": len(pred_ids),
                    "gold_tokens": len(gold_ids),
                    "token_matches": matches,
                    "token_accuracy": matches / max(1, len(gold_ids)),
                    "exact_tokens": pred_ids == gold_ids,
                    "generation": generation,
                    "gold": gold,
                    "generated_local_ids": pred_ids,
                    "gold_local_ids": gold_ids,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def evaluate_one(item: dict[str, Any]) -> ScoreRow:
    sample = item["sample"]
    try:
        task = VeriBenchTask.from_manifest_path(
            sample["task_name"],
            data_dir=Path(item["data_dir"]),
            split=sample["split"],
            activation_dtype="bf16",
        )
        metrics = task.evaluate_lean_output(
            sample["generation"],
            lake_dir=Path(item["lake_dir"]) if item["lake_dir"] else None,
            compile_timeout=int(item["compile_timeout"]),
            skip_te1=True,
        )
        return ScoreRow(
            run_name=sample["run_name"],
            task_name=sample["task_name"],
            split=sample["split"],
            family=sample["family"],
            pred_tokens=int(sample["pred_tokens"]),
            gold_tokens=int(sample["gold_tokens"]),
            token_matches=int(sample["token_matches"]),
            token_accuracy=float(sample["token_accuracy"]),
            exact_tokens=bool(sample["exact_tokens"]),
            candidate_len=len(sample["generation"]),
            ic1=float(metrics["IC1"]),
            ic2=float(metrics["IC2"]),
            te1=float(metrics["TE1"]),
            d1=float(metrics["D1"]),
            d2=float(metrics["D2"]),
            s_tilde=float(metrics["S_tilde"]),
            compile_candidate_success=bool(metrics["details"]["compile"]["candidate"]["success"]),
            compile_gold_success=bool(metrics["details"]["compile"]["gold"]["success"]),
            skip_reason="",
        )
    except Exception as exc:
        return ScoreRow(
            run_name=sample["run_name"],
            task_name=sample["task_name"],
            split=sample["split"],
            family=sample.get("family", ""),
            pred_tokens=int(sample.get("pred_tokens", 0)),
            gold_tokens=int(sample.get("gold_tokens", 0)),
            token_matches=int(sample.get("token_matches", 0)),
            token_accuracy=float(sample.get("token_accuracy", 0.0)),
            exact_tokens=bool(sample.get("exact_tokens", False)),
            candidate_len=len(sample.get("generation", "")),
            ic1=0.0,
            ic2=0.0,
            te1=0.0,
            d1=0.0,
            d2=0.0,
            s_tilde=0.0,
            compile_candidate_success=False,
            compile_gold_success=False,
            skip_reason=repr(exc),
        )


def write_scores(path: Path, rows: list[ScoreRow]) -> None:
    headers = list(ScoreRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in headers})


def aggregate(rows: list[ScoreRow]) -> list[dict[str, Any]]:
    buckets: dict[str, list[ScoreRow]] = {}
    for row in rows:
        buckets.setdefault(row.run_name, []).append(row)
    out: list[dict[str, Any]] = []
    for run_name, vals in sorted(buckets.items()):
        n = max(1, len(vals))
        out.append(
            {
                "run_name": run_name,
                "rows": len(vals),
                "exact_tokens": sum(int(r.exact_tokens) for r in vals),
                "exact_token_rate": sum(int(r.exact_tokens) for r in vals) / n,
                "token_accuracy": sum(r.token_accuracy for r in vals) / n,
                "compile_candidate_success": sum(int(r.compile_candidate_success) for r in vals),
                "compile_candidate_rate": sum(int(r.compile_candidate_success) for r in vals) / n,
                "compile_gold_success": sum(int(r.compile_gold_success) for r in vals),
                "compile_gold_rate": sum(int(r.compile_gold_success) for r in vals) / n,
                "IC1": sum(r.ic1 for r in vals) / n,
                "IC2": sum(r.ic2 for r in vals) / n,
                "TE1": sum(r.te1 for r in vals) / n,
                "D1": sum(r.d1 for r in vals) / n,
                "D2": sum(r.d2 for r in vals) / n,
                "S_tilde": sum(r.s_tilde for r in vals) / n,
            }
        )
    return out


def write_aggregate(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "run_name",
        "rows",
        "exact_tokens",
        "exact_token_rate",
        "token_accuracy",
        "compile_candidate_success",
        "compile_candidate_rate",
        "compile_gold_success",
        "compile_gold_rate",
        "IC1",
        "IC2",
        "TE1",
        "D1",
        "D2",
        "S_tilde",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-run-dir", type=Path, default=BASE_DIR / "runs/target7k_4h_bs2_val_1m_20260625_232609")
    parser.add_argument("--train-test-run-dir", type=Path, default=BASE_DIR / "runs/target7k_4h_bs2_train_test_1m_20260625_232609")
    parser.add_argument("--val-checkpoint", type=Path, default=None)
    parser.add_argument("--train-test-checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results/ebt_val_sampling_eval")
    parser.add_argument("--lake-dir", type=Path, default=None)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    run_specs = [
        ("val_train", args.val_run_dir.resolve(), args.val_checkpoint),
        ("train_test_train", args.train_test_run_dir.resolve(), args.train_test_checkpoint),
    ]

    sample_paths: list[Path] = []
    if not args.eval_only:
        for run_name, run_dir, ckpt_arg in run_specs:
            ckpt = ckpt_arg.resolve() if ckpt_arg is not None else latest_checkpoint(run_dir)
            print(f"Sampling {run_name}: {ckpt}")
            sample_paths.append(
                sample_run(
                    run_name=run_name,
                    run_dir=run_dir,
                    checkpoint_path=ckpt,
                    out_dir=args.out_dir,
                    split=args.split,
                    batch_size=args.batch_size,
                    device=device,
                    max_rows=args.max_rows,
                )
            )
    else:
        sample_paths = [args.out_dir / run_name / "samples.jsonl" for run_name, _, _ in run_specs]

    if args.sample_only:
        print(json.dumps({"samples": [str(p) for p in sample_paths]}, indent=2))
        return 0

    items: list[dict[str, Any]] = []
    for sample_path in sample_paths:
        samples = read_jsonl(sample_path)
        if args.max_rows is not None:
            samples = samples[: args.max_rows]
        for sample in samples:
            cfg = load_config(Path(sample["run_dir"]))
            items.append(
                {
                    "sample": sample,
                    "data_dir": str(cfg.data.data_dir),
                    "lake_dir": str(args.lake_dir or ""),
                    "compile_timeout": args.compile_timeout,
                }
            )

    scores: list[ScoreRow] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(evaluate_one, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="evaluating", unit="task"):
            scores.append(future.result())
    scores.sort(key=lambda r: (r.run_name, r.task_name))

    scores_path = args.out_dir / "scores.csv"
    aggregate_path = args.out_dir / "aggregate.csv"
    summary_path = args.out_dir / "summary.json"
    write_scores(scores_path, scores)
    agg = aggregate(scores)
    write_aggregate(aggregate_path, agg)
    summary = {"scores": str(scores_path), "aggregate": str(aggregate_path), "runs": agg}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
