#!/usr/bin/env python3
"""Compile-check generated Lean outputs from the SGLang Goedel run.

This checker does not use gold Lean contents. It runs each generated `.lean`
file through the VeriBench Lean 4.22/Mathlib project with `lake env lean`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import collections
import json
import pathlib
import subprocess
import time
from dataclasses import asdict, dataclass


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RUN_DIR = HERE / "results" / "goedel_prover_v2_8b_sglang_896"
DEFAULT_LAKE_DIR = HERE / "veribench" / "veribench_dataset" / "lean_src"


@dataclass
class CompileResult:
    lean_path: str
    ok: bool
    elapsed_sec: float
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile generated VeriBench Lean outputs.")
    parser.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--lake-dir", type=pathlib.Path, default=DEFAULT_LAKE_DIR)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser.parse_args()


def compile_one(lean_path: pathlib.Path, lake_dir: pathlib.Path, timeout_sec: int) -> CompileResult:
    lean_path = lean_path.resolve()
    started = time.time()
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(lean_path)],
            cwd=lake_dir,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CompileResult(
            lean_path=str(lean_path),
            ok=False,
            elapsed_sec=time.time() - started,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"TIMEOUT after {timeout_sec}s",
        )
    return CompileResult(
        lean_path=str(lean_path),
        ok=proc.returncode == 0,
        elapsed_sec=time.time() - started,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def main() -> int:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.lake_dir = args.lake_dir.resolve()
    lean_dir = args.run_dir / "lean_outputs"
    if not lean_dir.exists():
        raise FileNotFoundError(f"Missing generated output directory: {lean_dir}")
    if not args.lake_dir.exists():
        raise FileNotFoundError(f"Missing Lake project directory: {args.lake_dir}")

    lean_files = sorted(lean_dir.rglob("*.lean"))
    out_path = args.output or (args.run_dir / "compile_summary.json")

    results: list[CompileResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(compile_one, lean_path, args.lake_dir, args.timeout_sec)
            for lean_path in lean_files
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.lean_path)
    passed = sum(1 for item in results if item.ok)
    summary = {
        "run_dir": str(args.run_dir),
        "lake_dir": str(args.lake_dir),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": [asdict(item) for item in results],
    }
    gen_jsonl = args.run_dir / "generations.jsonl"
    if gen_jsonl.exists():
        by_path = {}
        with gen_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    output_path = row.get("output_lean_path")
                    if output_path:
                        by_path[str(pathlib.Path(output_path).resolve())] = row
        split_totals: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
        for item in results:
            split = by_path.get(item.lean_path, {}).get("split", "unknown")
            split_totals[split]["total"] += 1
            if item.ok:
                split_totals[split]["passed"] += 1
            else:
                split_totals[split]["failed"] += 1
        summary["by_split"] = {
            split: {
                **counts,
                "pass_rate": counts["passed"] / counts["total"] if counts["total"] else 0.0,
            }
            for split, counts in sorted(split_totals.items())
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compile pass: {passed}/{len(results)}")
    print(f"Wrote {out_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
