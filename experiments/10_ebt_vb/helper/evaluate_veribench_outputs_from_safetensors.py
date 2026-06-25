#!/usr/bin/env python3
"""Evaluate VeriBench tasks from safetensor rows and generated outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Keep fallback for environments where veribench_context_gold_dataloader isn't present.
_dataloader_path = BASE_DIR / "veribench_context_gold_dataloader.py"
if not _dataloader_path.exists():
    _fallback = types.ModuleType("veribench_context_gold_dataloader")
    _fallback.DEFAULT_DATA_DIR = BASE_DIR / "data" / "context_gold"
    sys.modules.setdefault("veribench_context_gold_dataloader", _fallback)

from veribench_task import VeriBenchTask

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None  # type: ignore[assignment]


FENCE_RE = re.compile(r"```(?:lean|lean4)?\s*([\s\S]*?)```", re.IGNORECASE)
LEAN_START = re.compile(r"^\s*(import|namespace|def|theorem|lemma|example|inductive|class|structure|instance|open)", re.MULTILINE)


@dataclass
class RowResult:
    task_name: str
    split: str
    family: str
    prompt_tokens: int
    manifest_generated_tokens: int
    tensor_prompt_tokens: int
    tensor_generated_tokens: int
    candidate_source: str
    candidate_len: int
    ic1: float
    ic2: float
    te1: float
    d1: float
    d2: float
    s_tilde: float
    compile_candidate_success: bool
    compile_gold_success: bool
    safe_tensor_path: str
    lake_dir: str
    skip_reason: str | None = None


def parse_args_with_flags() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BASE_DIR / "data/context_gold/manifest.jsonl")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--run-dir", type=Path, default=None, help="Override run directory.")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results/veribench_output_eval")
    parser.add_argument("--lake-dir", type=Path, default=None)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--with-te1", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Parallel evaluation workers")
    return parser.parse_args()


def parse_splits(raw: str) -> set[str]:
    vals = {v.strip() for v in raw.split(",") if v.strip()}
    return set() if not vals or "all" in vals else vals


def read_manifest(path: Path, requested_splits: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if requested_splits and str(row.get("split")) not in requested_splits:
                continue
            out.append(row)
    return out


def tensor_prompt_generated(path: Path) -> tuple[int, int]:
    if safe_open is None:
        return 0, 0
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            return int(meta.get("prompt_tokens", 0)), int(meta.get("generated_tokens", 0))
    except Exception:
        return 0, 0


def clean_lean(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    fences = FENCE_RE.findall(raw)
    if fences:
        raw = max(fences, key=len).strip()
    raw = re.sub(r"(?ms)^\\s*```(?:lean|lean4)?\\s*", "", raw)
    raw = re.sub(r"(?ms)\\s*```\\s*$", "", raw).strip()
    match = LEAN_START.search(raw)
    if match:
        raw = raw[match.start() :].strip()
    if "```" in raw:
        raw = raw.split("```", 1)[0].strip()
    return raw + "\n"


def infer_run_dir(row: dict[str, Any]) -> Path:
    return Path(row["safetensors_path"]).expanduser().resolve().parent.parent


def load_outputs(run_dir: Path) -> dict[str, dict[str, Any]]:
    p = run_dir / "generations.jsonl"
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = str(rec.get("task_name") or "").strip()
            if t:
                out[t] = rec
    return out


def resolve_path(run_dir: Path, candidate: str, row: dict[str, Any], ext: str) -> Path | None:
    if not candidate:
        return None
    p = Path(candidate)
    family = str(row.get("family") or "").strip()
    base = str(row["task_name"]).split("/")[-1]
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([p, run_dir / p])
    if family:
        candidates.extend([run_dir / p / f"{base}{ext}", run_dir / "lean_outputs" / family / f"{base}{ext}", run_dir / "raw_model_outputs" / family / f"{base}{ext}"])
    for c in candidates:
        if c.is_dir():
            c = c / f"{base}{ext}"
        if c.exists():
            return c
    return None


def pick_candidate(row: dict[str, Any], run_outputs: dict[str, dict[str, Any]], run_dir: Path) -> tuple[str, str]:
    t = str(row["task_name"])
    family = str(row.get("family") or "")
    base = t.split("/")[-1]
    candidates: list[tuple[str, str]] = []
    rec = run_outputs.get(t, {})
    if rec.get("raw_output_text"):
        candidates.append((str(rec["raw_output_text"]), "generations.raw_output_text"))
    if rec.get("output_text"):
        candidates.append((str(rec["output_text"]), "generations.output_text"))
    if rec.get("output_lean_path"):
        candidates.append((str(rec["output_lean_path"]), "generations.output_lean_path"))
    if rec.get("raw_output_path"):
        candidates.append((str(rec["raw_output_path"]), "generations.raw_output_path"))
    candidates.append((str(run_dir / "lean_outputs" / family / f"{base}.lean"), "run_dir/lean_outputs"))
    candidates.append((str(run_dir / "raw_model_outputs" / family / f"{base}.txt"), "run_dir/raw_model_outputs"))

    seen: set[str] = set()
    for source, label in candidates:
        if source in seen:
            continue
        seen.add(source)
        if "\n" in source:
            cleaned = clean_lean(source)
            if cleaned.strip():
                return cleaned, label
        # Treat long or unstructured values as literal text, not a filesystem path.
        if ("\n" in source) or (len(source) > 300) or ("\t" in source):
            continue
        p = resolve_path(run_dir, source, row, ".lean" if "lean" in label else ".txt")
        if p is not None:
            try:
                cleaned = clean_lean(p.read_text(encoding="utf-8"))
                if cleaned.strip():
                    return cleaned, label
            except OSError:
                pass
        if "raw" not in label:
            # fallback: treat as literal text if it is not a path-like string.
            if source and "/" not in source:
                cleaned = clean_lean(source)
                if cleaned.strip():
                    return cleaned, label
    return "", "missing"


def evaluate_row_worker(item: dict[str, Any]) -> RowResult:
    row = item["row"]
    candidate = item["candidate_text"]
    lake_dir = Path(item["lake_dir"]) if item["lake_dir"] else None
    compile_timeout = int(item["compile_timeout"])
    skip_te1 = bool(item["skip_te1"])

    safe_tensor_path = str(row["safetensors_path"])
    manifest_prompt = int(row.get("prompt_tokens", 0))
    manifest_generated = int(row.get("generated_tokens", 0))
    tensor_prompt = int(item["tensor_prompt_tokens"])
    tensor_generated = int(item["tensor_generated_tokens"])

    try:
        task = VeriBenchTask.from_manifest_row(row=row, data_dir=Path(item["data_dir"]))
        metrics = task.evaluate_lean_output(
            candidate,
            lake_dir=lake_dir,
            compile_timeout=compile_timeout,
            skip_te1=skip_te1,
        )
        return RowResult(
            task_name=row["task_name"],
            split=row["split"],
            family=row.get("family") or "",
            prompt_tokens=manifest_prompt,
            manifest_generated_tokens=manifest_generated,
            tensor_prompt_tokens=tensor_prompt,
            tensor_generated_tokens=tensor_generated,
            candidate_source=item["candidate_source"],
            candidate_len=len(candidate),
            ic1=float(metrics["IC1"]),
            ic2=float(metrics["IC2"]),
            te1=float(metrics["TE1"]),
            d1=float(metrics["D1"]),
            d2=float(metrics["D2"]),
            s_tilde=float(metrics["S_tilde"]),
            compile_candidate_success=bool(metrics["details"]["compile"]["candidate"]["success"]),
            compile_gold_success=bool(metrics["details"]["compile"]["gold"]["success"]),
            safe_tensor_path=safe_tensor_path,
            lake_dir=str(item["lake_dir"] or task.default_lake_dir),
            skip_reason=None,
        )
    except Exception as exc:  # pragma: no cover - robust runtime handling
        return RowResult(
            task_name=row["task_name"],
            split=row["split"],
            family=row.get("family") or "",
            prompt_tokens=manifest_prompt,
            manifest_generated_tokens=manifest_generated,
            tensor_prompt_tokens=tensor_prompt,
            tensor_generated_tokens=tensor_generated,
            candidate_source=item["candidate_source"],
            candidate_len=len(candidate),
            ic1=0.0,
            ic2=0.0,
            te1=0.0,
            d1=0.0,
            d2=0.0,
            s_tilde=0.0,
            compile_candidate_success=False,
            compile_gold_success=False,
            safe_tensor_path=safe_tensor_path,
            lake_dir=str(item["lake_dir"] or ""),
            skip_reason=repr(exc),
        )


def mean_by_split(rows: list[RowResult]) -> list[dict[str, float]]:
    agg: dict[str, dict[str, float]] = {}
    for r in rows:
        s = agg.setdefault(r.split, {"rows": 0, "IC1": 0.0, "IC2": 0.0, "TE1": 0.0, "D1": 0.0, "D2": 0.0, "S_tilde": 0.0})
        s["rows"] += 1
        s["IC1"] += r.ic1
        s["IC2"] += r.ic2
        s["TE1"] += r.te1
        s["D1"] += r.d1
        s["D2"] += r.d2
        s["S_tilde"] += r.s_tilde

    total = {"rows": 0, "IC1": 0.0, "IC2": 0.0, "TE1": 0.0, "D1": 0.0, "D2": 0.0, "S_tilde": 0.0}
    for s in agg.values():
        total["rows"] += s["rows"]
        for k in ("IC1", "IC2", "TE1", "D1", "D2", "S_tilde"):
            total[k] += s[k]

    out = []
    for split in sorted(agg):
        s = agg[split]
        n = int(s["rows"]) or 1
        out.append(
            {k: (s[k] / n if k != "rows" else int(s["rows"])) for k in ("rows", "IC1", "IC2", "TE1", "D1", "D2", "S_tilde")}
        )
        out[-1]["split"] = split
    n_total = int(total["rows"]) or 1
    out.append(
        {
            "split": "all",
            "rows": int(total["rows"]),
            "IC1": total["IC1"] / n_total,
            "IC2": total["IC2"] / n_total,
            "TE1": total["TE1"] / n_total,
            "D1": total["D1"] / n_total,
            "D2": total["D2"] / n_total,
            "S_tilde": total["S_tilde"] / n_total,
        }
    )
    return out


def write_csv_rows(path: Path, rows: list[RowResult]) -> None:
    headers = [
        "task_name",
        "split",
        "family",
        "prompt_tokens",
        "manifest_generated_tokens",
        "tensor_prompt_tokens",
        "tensor_generated_tokens",
        "candidate_len_chars",
        "candidate_source",
        "IC1",
        "IC2",
        "TE1",
        "D1",
        "D2",
        "S_tilde",
        "compile_candidate_success",
        "compile_gold_success",
        "safe_tensor_path",
        "lake_dir",
        "skip_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "task_name": r.task_name,
                    "split": r.split,
                    "family": r.family,
                    "prompt_tokens": r.prompt_tokens,
                    "manifest_generated_tokens": r.manifest_generated_tokens,
                    "tensor_prompt_tokens": r.tensor_prompt_tokens,
                    "tensor_generated_tokens": r.tensor_generated_tokens,
                    "candidate_len_chars": r.candidate_len,
                    "candidate_source": r.candidate_source,
                    "IC1": f"{r.ic1:.6f}",
                    "IC2": f"{r.ic2:.6f}",
                    "TE1": f"{r.te1:.6f}",
                    "D1": f"{r.d1:.6f}",
                    "D2": f"{r.d2:.6f}",
                    "S_tilde": f"{r.s_tilde:.6f}",
                    "compile_candidate_success": int(r.compile_candidate_success),
                    "compile_gold_success": int(r.compile_gold_success),
                    "safe_tensor_path": r.safe_tensor_path,
                    "lake_dir": r.lake_dir,
                    "skip_reason": r.skip_reason or "",
                }
            )


def write_csv_summary(path: Path, summaries: list[dict[str, float]]) -> None:
    headers = ["split", "rows", "IC1", "IC2", "TE1", "D1", "D2", "S_tilde"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for s in summaries:
            w.writerow(
                {
                    "split": s["split"],
                    "rows": s["rows"],
                    "IC1": f"{s['IC1']:.6f}",
                    "IC2": f"{s['IC2']:.6f}",
                    "TE1": f"{s['TE1']:.6f}",
                    "D1": f"{s['D1']:.6f}",
                    "D2": f"{s['D2']:.6f}",
                    "S_tilde": f"{s['S_tilde']:.6f}",
                }
            )


def main() -> int:
    args = parse_args_with_flags()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = parse_splits(args.splits)
    rows = read_manifest(args.manifest.resolve(), splits)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        print("No rows found")
        return 1

    # cache generation files per run directory
    run_outputs_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    for row in rows:
        run_dir = args.run_dir.resolve() if args.run_dir is not None else infer_run_dir(row)
        if run_dir not in run_outputs_cache:
            run_outputs_cache[run_dir] = load_outputs(run_dir)

    tasks: list[dict[str, Any]] = []
    for row in rows:
        run_dir = args.run_dir.resolve() if args.run_dir is not None else infer_run_dir(row)
        run_outputs = run_outputs_cache[run_dir]
        candidate_text, candidate_source = pick_candidate(row, run_outputs, run_dir)
        prompt_tokens_meta, generated_tokens_meta = tensor_prompt_generated(Path(row["safetensors_path"]).resolve())
        tasks.append(
            {
                "row": row,
                "candidate_text": candidate_text,
                "candidate_source": candidate_source,
                "tensor_prompt_tokens": prompt_tokens_meta,
                "tensor_generated_tokens": generated_tokens_meta,
                "compile_timeout": args.compile_timeout,
                "skip_te1": not args.with_te1,
                "lake_dir": str(args.lake_dir or ""),
                "data_dir": str(args.manifest.parent.resolve()),
            }
        )

    # Parallelize expensive evaluate step (Lean compiles).
    results: list[RowResult] = []
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(evaluate_row_worker, item) for item in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="evaluating", unit="task"):
            results.append(fut.result())

    per_row_path = args.out_dir / "veribench_output_scores.csv"
    agg_path = args.out_dir / "veribench_output_aggregate.csv"
    write_csv_rows(per_row_path, results)
    write_csv_summary(agg_path, mean_by_split(results))
    print(f"Rows processed: {len(results)}")
    print(f"Wrote: {per_row_path}")
    print(f"Wrote: {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
