#!/usr/bin/env python3
"""Validate cleaned VeriBench gold Lean targets by compiling generated files."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BASE_DIR / "data" / "context_gold"
DEFAULT_LAKE_DIR = (
    BASE_DIR.parents[0]
    / "09_vb_testing_ipynb"
    / "veribench"
    / "veribench_dataset"
    / "lean_src"
)
DEFAULT_OUT_DIR = BASE_DIR / "results" / "cleaned_lean_validation"
VALIDATION_SET_OPTION = "set_option linter.unusedVariables false"


@dataclass(frozen=True)
class CompileJob:
    index: int
    task_name: str
    split: str
    lean_path: str
    lake_dir: str
    timeout: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_rel_task_path(task_name: str) -> Path:
    return Path(*task_name.split("/")).with_suffix(".lean")


def decode_targets(rows: list[dict[str, Any]], data_dir: Path) -> list[str]:
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(summary["tokenizer"], trust_remote_code=True)
    return [tokenizer.decode(row["target_original_ids"], skip_special_tokens=True) for row in rows]


def with_validation_header(source: str) -> str:
    imports = ["import Std"]
    body_lines: list[str] = []
    seen_imports = {"import Std"}
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            if stripped not in seen_imports:
                imports.append(stripped)
                seen_imports.add(stripped)
            continue
        if stripped == VALIDATION_SET_OPTION:
            continue
        body_lines.append(line)
    return "\n".join(imports + [VALIDATION_SET_OPTION] + body_lines).lstrip("\n") + "\n"


def compile_one(job: CompileJob) -> dict[str, Any]:
    cmd = ["lake", "env", "lean", job.lean_path]
    try:
        proc = subprocess.run(
            cmd,
            cwd=job.lake_dir,
            capture_output=True,
            text=True,
            timeout=job.timeout,
            check=False,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        success = proc.returncode == 0 and "error" not in stderr.lower()
        message = stderr.strip() or stdout.strip()
        return {
            "index": job.index,
            "task_name": job.task_name,
            "split": job.split,
            "success": success,
            "returncode": proc.returncode,
            "lean_path": job.lean_path,
            "message": message[:4000],
        }
    except subprocess.TimeoutExpired:
        return {
            "index": job.index,
            "task_name": job.task_name,
            "split": job.split,
            "success": False,
            "returncode": -1,
            "lean_path": job.lean_path,
            "message": "timeout",
        }
    except FileNotFoundError:
        return {
            "index": job.index,
            "task_name": job.task_name,
            "split": job.split,
            "success": False,
            "returncode": -1,
            "lean_path": job.lean_path,
            "message": "lake not found",
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["index", "task_name", "split", "success", "returncode", "lean_path", "message"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = {}
    for row in results:
        counter = by_split.setdefault(row["split"], Counter())
        counter["rows"] += 1
        counter["success"] += int(bool(row["success"]))
    split_summary = {
        split: {
            "rows": counts["rows"],
            "success": counts["success"],
            "failed": counts["rows"] - counts["success"],
            "pass_rate": counts["success"] / counts["rows"] if counts["rows"] else 0.0,
        }
        for split, counts in sorted(by_split.items())
    }
    total_rows = len(results)
    total_success = sum(int(bool(row["success"])) for row in results)
    return {
        "rows": total_rows,
        "success": total_success,
        "failed": total_rows - total_success,
        "pass_rate": total_success / total_rows if total_rows else 0.0,
        "by_split": split_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--lake-dir", type=Path, default=DEFAULT_LAKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    lean_dir = out_dir / "lean_files"
    lean_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_dir / "manifest.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    decoded = decode_targets(rows, data_dir)

    jobs: list[CompileJob] = []
    for index, (row, source) in enumerate(zip(rows, decoded, strict=True)):
        lean_path = lean_dir / safe_rel_task_path(row["task_name"])
        lean_path.parent.mkdir(parents=True, exist_ok=True)
        lean_path.write_text(with_validation_header(source), encoding="utf-8")
        jobs.append(
            CompileJob(
                index=index,
                task_name=row["task_name"],
                split=row["split"],
                lean_path=str(lean_path),
                lake_dir=str(args.lake_dir.resolve()),
                timeout=int(args.timeout),
            )
        )

    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compile_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if done == len(futures) or done % 25 == 0:
                passed = sum(int(bool(row["success"])) for row in results)
                print(f"compiled={done}/{len(futures)} pass={passed} fail={done - passed}", flush=True)

    results.sort(key=lambda row: int(row["index"]))
    summary = summarize(results)
    write_csv(out_dir / "compile_results.csv", results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote={out_dir}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
