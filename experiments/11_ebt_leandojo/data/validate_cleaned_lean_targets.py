#!/usr/bin/env python3
"""Validate cleaned Lean Workbook Plus targets by compiling them in Mathlib."""

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


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "context_gold"
DEFAULT_LAKE_DIR = HERE.parent / "repl" / "test" / "Mathlib"
DEFAULT_OUT_DIR = HERE.parent / "results" / "cleaned_leanworkbook_validation"
VALIDATION_SET_OPTION = "set_option linter.unusedVariables false"


@dataclass(frozen=True)
class CompileJob:
    index: int
    task_id: str
    status: str
    lean_path: str
    lake_dir: str
    timeout: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def with_validation_header(source: str) -> str:
    imports = ["import Mathlib"]
    body_lines: list[str] = []
    seen_imports = {"import Mathlib"}
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
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", job.lean_path],
            cwd=job.lake_dir,
            capture_output=True,
            text=True,
            timeout=job.timeout,
            check=False,
        )
        return {
            "index": job.index,
            "task_id": job.task_id,
            "status": job.status,
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "lean_path": job.lean_path,
            "message": (proc.stderr or proc.stdout).strip()[:4000],
        }
    except subprocess.TimeoutExpired:
        return {
            "index": job.index,
            "task_id": job.task_id,
            "status": job.status,
            "success": False,
            "returncode": -1,
            "lean_path": job.lean_path,
            "message": "timeout",
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["index", "task_id", "status", "success", "returncode", "lean_path", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, Counter[str]] = {}
    for row in results:
        counter = by_status.setdefault(row["status"], Counter())
        counter["rows"] += 1
        counter["success"] += int(bool(row["success"]))
    status_summary = {
        status: {
            "rows": counts["rows"],
            "success": counts["success"],
            "failed": counts["rows"] - counts["success"],
            "pass_rate": counts["success"] / counts["rows"] if counts["rows"] else 0.0,
        }
        for status, counts in sorted(by_status.items())
    }
    success = sum(int(bool(row["success"])) for row in results)
    return {
        "rows": len(results),
        "success": success,
        "failed": len(results) - success,
        "pass_rate": success / len(results) if results else 0.0,
        "by_status": status_summary,
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

    rows = read_jsonl(args.data_dir.resolve() / "manifest.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    out_dir = args.out_dir.resolve()
    lean_dir = out_dir / "lean_files"
    lean_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[CompileJob] = []
    for row in rows:
        lean_path = lean_dir / f"{int(row['index']):06d}_{row['task_id']}.lean"
        lean_path.write_text(with_validation_header(str(row["target_text"])), encoding="utf-8")
        jobs.append(
            CompileJob(
                index=int(row["index"]),
                task_id=str(row["task_id"]),
                status=str(row["status"]),
                lean_path=str(lean_path),
                lake_dir=str(args.lake_dir.resolve()),
                timeout=int(args.timeout),
            )
        )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(compile_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if done == len(futures) or done % 100 == 0:
                passed = sum(int(bool(row["success"])) for row in results)
                print(f"compiled={done}/{len(futures)} pass={passed} fail={done - passed}", flush=True)

    results.sort(key=lambda row: int(row["index"]))
    summary = summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "compile_results.csv", results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote={out_dir}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
