#!/usr/bin/env python3
"""Lean Workbook Plus natural-language -> Lean proof benchmark."""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
DEFAULT_DATASET = "internlm/Lean-Workbook"
DEFAULT_SEED = 3407
DEFAULT_VAL_SIZE = 500
FENCE_RE = re.compile(r"```(?:lean4|lean)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
HEADER_RE = re.compile(r"(?s)\btheorem\b.*?:=\s*by\s*")
SORRY_SUFFIX_RE = re.compile(r":=\s*by\s*sorry\s*$", flags=re.DOTALL)


@dataclass(frozen=True)
class LeanWorkbookTask:
    index: int
    task_id: str
    status: str
    natural_language_statement: str
    formal_statement: str
    prompt: str


@dataclass(frozen=True)
class LeanWorkbookCompileSummary:
    dataset_name: str
    task_count: int
    compile_success_count: int
    compile_success_rate: float
    generated_file: str
    summary_file: str


BatchGenerator = Callable[[Sequence[LeanWorkbookTask]], list[str]]


def theorem_skeleton(formal_statement: str) -> str:
    return SORRY_SUFFIX_RE.sub(":= by", formal_statement.strip())


def make_prompt(task_id: str, statement: str, formal_statement: str) -> str:
    return (
        "# Task: Prove a Lean 4 theorem\n\n"
        "You are given a natural language statement and the Lean 4 theorem skeleton.\n"
        "Return only the Lean 4 proof body.\n"
        "Do not repeat the theorem declaration.\n"
        "Do not include markdown, explanation, comments, imports, or surrounding text.\n"
        "If you use a fenced code block, it must contain only the proof body.\n\n"
        f"Theorem name: {task_id}\n\n"
        "Natural language statement:\n"
        f"{statement}\n\n"
        "Lean theorem skeleton:\n"
        "```lean4\n"
        f"{theorem_skeleton(formal_statement)}\n"
        "```"
    )


def extract_proof_body(text: str) -> str:
    text = text.strip()
    match = FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    header_match = HEADER_RE.search(text)
    if header_match:
        text = text[header_match.end() :].strip()
    if text.startswith("by"):
        text = text[2:].lstrip()
    for stop in ("\n```", "\n# ", "\n/-", "\n--"):
        if stop in text:
            text = text.split(stop, 1)[0].rstrip()
    return text.strip()


def build_theorem_code(formal_statement: str, proof_body: str) -> str:
    proof_body = extract_proof_body(proof_body)
    if not proof_body:
        return ""
    body = "\n".join(("  " + line) if line.strip() else "" for line in proof_body.splitlines())
    return theorem_skeleton(formal_statement) + "\n" + body


def download_dataset(
    dataset_name: str = DEFAULT_DATASET,
    *,
    out_dir: Path = DEFAULT_DATA_DIR,
    overwrite: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leanworkbook_plus_train.jsonl"
    if out_path.exists() and not overwrite:
        return out_path
    from datasets import load_dataset

    load_dataset(dataset_name)["train"].to_json(str(out_path))
    return out_path


def load_tasks(
    input_file: Path,
    *,
    indices: Sequence[int] | None = None,
    max_items: int | None = None,
) -> list[LeanWorkbookTask]:
    rows = [json.loads(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if indices is not None:
        index_set = set(indices)
        rows = [row for i, row in enumerate(rows) if i in index_set]
    if max_items is not None:
        rows = rows[:max_items]
    tasks: list[LeanWorkbookTask] = []
    for i, row in enumerate(rows):
        tasks.append(
            LeanWorkbookTask(
                index=i,
                task_id=str(row["id"]),
                status=str(row["status"]),
                natural_language_statement=str(row["natural_language_statement"]),
                formal_statement=str(row["formal_statement"]),
                prompt=make_prompt(str(row["id"]), str(row["natural_language_statement"]), str(row["formal_statement"])),
            )
        )
    return tasks


def write_validation_indices(
    input_file: Path,
    *,
    out_file: Path,
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_VAL_SIZE,
) -> Path:
    total = sum(1 for line in input_file.open(encoding="utf-8") if line.strip())
    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds dataset size {total}")
    rng = random.Random(seed)
    sampled = sorted(rng.sample(range(total), sample_size))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"seed": seed, "sample_size": sample_size, "indices": sampled}, indent=2), encoding="utf-8")
    return out_file


def _compile_chunk(
    rows: list[dict[str, Any]],
    repl_path: str,
    lean_env_path: str,
    timeout: int,
    expect_timeout: int,
) -> list[dict[str, Any]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("fm_verify", str(HERE / "FormalMATH-Bench" / "verify_answers.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    thread = mod.InteractiveThread(
        0,
        repl_path,
        lean_env_path,
        initial_context="import Mathlib",
        timeout=timeout,
        expect_timeout=expect_timeout,
    )
    thread.start()
    thread.init_complete.wait()
    outputs: list[dict[str, Any]] = []
    try:
        for row in rows:
            theorem_code = build_theorem_code(row["formal_statement"], row["prediction_raw"])
            success = False
            outcome = None
            if theorem_code:
                outcome = thread.submit_and_receive({"cmd": theorem_code, "env": 0})
                if outcome is not None:
                    messages = outcome.get("messages", [])
                    has_error = any(msg.get("severity") == "error" for msg in messages)
                    has_sorry = any(msg.get("severity") == "sorries" for msg in messages) or ("sorries" in outcome)
                    success = not has_error and not has_sorry
            row["extracted_proof"] = extract_proof_body(row["prediction_raw"])
            row["compiled_code"] = theorem_code
            row["compile_success"] = success
            row["compile_output"] = outcome
            outputs.append(row)
    finally:
        thread.stop()
        thread.join()
    return outputs


def benchmark_compile_success(
    *,
    dataset_name: str,
    tasks: Sequence[LeanWorkbookTask],
    generator: BatchGenerator,
    out_dir: Path,
    batch_size: int,
    repl_path: Path,
    lean_env_path: Path,
    num_compile_workers: int = 8,
    session_timeout: int = 600,
    expect_timeout: int = 120,
) -> LeanWorkbookCompileSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_path = out_dir / "generated.json"
    summary_path = out_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    for start in range(0, len(tasks), batch_size):
        batch = list(tasks[start : start + batch_size])
        predictions = generator(batch)
        if len(predictions) != len(batch):
            raise ValueError(f"Generator returned {len(predictions)} predictions for batch of size {len(batch)}")
        for task, pred in zip(batch, predictions):
            rows.append(
                {
                    "index": task.index,
                    "task_id": task.task_id,
                    "status": task.status,
                    "natural_language_statement": task.natural_language_statement,
                    "formal_statement": task.formal_statement,
                    "prediction_raw": pred,
                }
            )

    chunks = [rows[i : i + max(1, len(rows) // max(1, num_compile_workers) + 1)] for i in range(0, len(rows), max(1, len(rows) // max(1, num_compile_workers) + 1))]
    compiled_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=num_compile_workers) as ex:
        futures = [
            ex.submit(
                _compile_chunk,
                chunk,
                str(repl_path.resolve()),
                str(lean_env_path.resolve()),
                session_timeout,
                expect_timeout,
            )
            for chunk in chunks
        ]
        for fut in futures:
            compiled_rows.extend(fut.result())

    compiled_rows.sort(key=lambda row: row["index"])
    generated_path.write_text(json.dumps(compiled_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    success_count = sum(int(row["compile_success"]) for row in compiled_rows)
    summary = LeanWorkbookCompileSummary(
        dataset_name=dataset_name,
        task_count=len(compiled_rows),
        compile_success_count=success_count,
        compile_success_rate=(success_count / len(compiled_rows) if compiled_rows else 0.0),
        generated_file=str(generated_path),
        summary_file=str(summary_path),
    )
    summary_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary

