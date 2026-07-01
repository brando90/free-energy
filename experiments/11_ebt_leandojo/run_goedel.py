#!/usr/bin/env python3
"""Benchmark Goedel-Prover-V2-8B on Lean Workbook Plus statement recovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from helpers.benchmark import (
    DEFAULT_DATASET,
    DEFAULT_DATA_DIR,
    DEFAULT_SEED,
    DEFAULT_VAL_SIZE,
    LeanWorkbookTask,
    LeanWorkbookCompileSummary,
    benchmark_compile_success,
    download_dataset,
    load_tasks,
    write_validation_indices,
)
from helpers.lean_repl import DEFAULT_LEAN_ENV_DIR, DEFAULT_REPL_DIR, ensure_repl_env


DEFAULT_MODEL = "Goedel-LM/Goedel-Prover-V2-8B"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "goedel_val500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Goedel-Prover-V2-8B on Lean Workbook Plus.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--indices-file", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--num-compile-workers", type=int, default=8)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--expect-timeout", type=int, default=120)
    parser.add_argument("--sample-validation", action="store_true")
    parser.add_argument("--skip-repl-setup", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def extract_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        outputs = response.get("outputs")
        if isinstance(outputs, list) and outputs:
            return str(outputs[0].get("text", ""))
        if "text" in response:
            return str(response["text"])
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    outputs = getattr(response, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return str(getattr(outputs[0], "text", ""))
    return str(response)


class SGLangGenerator:
    def __init__(
        self,
        *,
        model: str,
        gpus: str,
        data_parallel_size: int,
        tensor_parallel_size: int,
        context_length: int,
        mem_fraction_static: float,
        disable_cuda_graph: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        from sglang import Engine

        self.engine = Engine(
            model_path=model,
            trust_remote_code=True,
            dtype="bfloat16",
            dp_size=data_parallel_size,
            tp_size=tensor_parallel_size,
            load_balance_method="round_robin",
            context_length=context_length,
            mem_fraction_static=mem_fraction_static,
            disable_cuda_graph=disable_cuda_graph,
        )
        self.sampling_params: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k >= 0:
            self.sampling_params["top_k"] = top_k
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

    def generate_batch(self, tasks: Sequence[LeanWorkbookTask]) -> list[str]:
        asyncio.set_event_loop(self._event_loop)
        prompts = [task.prompt for task in tasks]
        responses = self.engine.generate(prompt=prompts, sampling_params=self.sampling_params)
        if isinstance(responses, dict):
            responses = [responses]
        return [extract_text(response) for response in responses]

    def shutdown(self) -> None:
        self.engine.shutdown()
        self._event_loop.close()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_file = args.input_file or download_dataset(args.dataset, out_dir=args.data_dir)
    indices_file = args.indices_file or (args.data_dir / "leanworkbook_plus_val500_indices.json")
    if args.sample_validation or not indices_file.exists():
        write_validation_indices(input_file, out_file=indices_file, seed=args.seed, sample_size=args.val_size)
    if args.download_only:
        print(f"dataset={input_file}")
        print(f"indices={indices_file}")
        return 0

    if args.skip_repl_setup:
        repl_path = DEFAULT_REPL_DIR
        lean_env_path = DEFAULT_LEAN_ENV_DIR
    else:
        repl_path, lean_env_path = ensure_repl_env()

    index_payload = json.loads(indices_file.read_text(encoding="utf-8"))
    tasks = load_tasks(input_file, indices=index_payload["indices"], max_items=args.max_items)
    metadata = {
        "dataset": args.dataset,
        "input_file": str(input_file),
        "indices_file": str(indices_file),
        "task_count": len(tasks),
        "model": args.model,
        "gpus": args.gpus,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "disable_cuda_graph": args.disable_cuda_graph,
        "seed": args.seed,
        "val_size": args.val_size,
        "num_compile_workers": args.num_compile_workers,
        "repl_path": str(repl_path),
        "lean_env_path": str(lean_env_path),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    generator = SGLangGenerator(
        model=args.model,
        gpus=args.gpus,
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=args.disable_cuda_graph,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    try:
        summary: LeanWorkbookCompileSummary = benchmark_compile_success(
            dataset_name=args.dataset,
            tasks=tasks,
            generator=generator.generate_batch,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            repl_path=repl_path,
            lean_env_path=lean_env_path,
            num_compile_workers=args.num_compile_workers,
            session_timeout=args.session_timeout,
            expect_timeout=args.expect_timeout,
        )
    finally:
        generator.shutdown()

    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
