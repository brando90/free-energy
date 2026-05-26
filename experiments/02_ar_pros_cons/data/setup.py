"""Create deterministic VeriBench manifests for the LeCun error-compounding tests.

The splitter is intentionally lightweight: it does not copy VeriBench data into
this repository. It scans a local VeriBench checkout, writes JSONL manifests with
absolute source paths, and keeps all variants of a task in the same split.

Default input:
    ~/veribench/veribench_dataset

Example:
    python -m data.setup --smoke
    python -m data.setup --veribench-root ~/veribench/veribench_dataset --include-generated-agents
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


THEOREM_RE = re.compile(r"^\s*(?:noncomputable\s+)?(?:@\[[^\n]*\]\s*)?theorem\s+", re.MULTILINE)
SORRY_RE = re.compile(r"\b(?:sorry|admit)\b")
TACTIC_RE = re.compile(
    r"\b(?:simp|omega|aesop|rw|rfl|exact|apply|intro|intros|constructor|cases|induction|linarith|ring|norm_num)\b"
)


@dataclass(frozen=True)
class ManifestRow:
    split: str
    task_id: str
    variant_id: str
    source_kind: str
    family: str
    lean_path: str
    rel_lean_path: str
    py_path: str | None
    line_count: int
    char_count: int
    theorem_count: int
    sorry_count: int
    tactic_count_proxy: int
    sha256: str


def stable_bucket(key: str) -> float:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12)


def split_for_task(task_id: str, train_frac: float, val_frac: float) -> str:
    bucket = stable_bucket(task_id)
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "val"
    return "test"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def row_for_lean_file(
    path: Path,
    dataset_root: Path,
    source_kind: str,
    train_frac: float,
    val_frac: float,
    py_path: Path | None = None,
    variant_id: str | None = None,
) -> ManifestRow:
    text = read_text(path)
    rel = path.relative_to(dataset_root)
    if len(rel.parts) > 2 and rel.parts[0] == "lean_src" and rel.parts[1] == "veribench":
        family = rel.parts[2]
    else:
        family = rel.parts[0]
    task_id = path.stem
    if source_kind == "generated_agent":
        task_id = re.sub(r"__agent\d+$", "", path.stem)
    split = split_for_task(task_id, train_frac, val_frac)
    return ManifestRow(
        split=split,
        task_id=task_id,
        variant_id=variant_id or path.stem,
        source_kind=source_kind,
        family=family,
        lean_path=str(path.resolve()),
        rel_lean_path=str(rel),
        py_path=str(py_path.resolve()) if py_path and py_path.exists() else None,
        line_count=text.count("\n") + 1,
        char_count=len(text),
        theorem_count=len(THEOREM_RE.findall(text)),
        sorry_count=len(SORRY_RE.findall(text)),
        tactic_count_proxy=len(TACTIC_RE.findall(text)),
        sha256=hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
    )


def iter_gold_rows(dataset_root: Path, train_frac: float, val_frac: float) -> Iterable[ManifestRow]:
    lean_root = dataset_root / "lean_src" / "veribench"
    for path in sorted(lean_root.rglob("*.lean")):
        yield row_for_lean_file(path, dataset_root, "gold", train_frac, val_frac)


def localize_generated_path(dataset_root: Path, raw_output_path: str | None, task_id: str, agent: str) -> Path:
    if raw_output_path:
        candidate = dataset_root / "generated_agents" / Path(raw_output_path).name
        if candidate.exists():
            return candidate
    return dataset_root / "generated_agents" / f"{task_id}__{agent}.lean"


def localize_py_path(dataset_root: Path, raw_py_path: str | None) -> Path | None:
    if not raw_py_path:
        return None
    raw = Path(raw_py_path)
    parts = raw.parts
    if "veribench_dataset" in parts:
        suffix = parts[parts.index("veribench_dataset") + 1 :]
        candidate = dataset_root.joinpath(*suffix)
        if candidate.exists():
            return candidate
    matches = list((dataset_root / "py_src").rglob(raw.name))
    return matches[0] if matches else None


def iter_generated_rows(dataset_root: Path, train_frac: float, val_frac: float) -> Iterable[ManifestRow]:
    manifest_path = dataset_root / "generated_agents" / "generation_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        task_id = str(entry.get("task_id", "")).strip()
        agent = str(entry.get("agent", "agent_unknown")).strip()
        if not task_id:
            continue
        lean_path = localize_generated_path(dataset_root, entry.get("output_path"), task_id, agent)
        if not lean_path.exists():
            continue
        py_path = localize_py_path(dataset_root, entry.get("python_path"))
        yield row_for_lean_file(
            lean_path,
            dataset_root,
            "generated_agent",
            train_frac,
            val_frac,
            py_path=py_path,
            variant_id=f"{task_id}__{agent}",
        )


def write_jsonl(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def write_summary(path: Path, rows: list[ManifestRow], args: argparse.Namespace) -> None:
    by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    by_kind: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        by_split[row.split] = by_split.get(row.split, 0) + 1
        by_kind[row.source_kind] = by_kind.get(row.source_kind, 0) + 1
        by_family[row.family] = by_family.get(row.family, 0) + 1
    summary = {
        "veribench_root": str(Path(args.veribench_root).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "smoke": bool(args.smoke),
        "include_generated_agents": bool(args.include_generated_agents),
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - args.train_frac - args.val_frac,
        "n_rows": len(rows),
        "by_split": by_split,
        "by_source_kind": by_kind,
        "by_family": dict(sorted(by_family.items())),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veribench-root", default="~/veribench/veribench_dataset")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--smoke", action="store_true", help="Write a 20-row smoke manifest in addition to full splits.")
    parser.add_argument(
        "--include-generated-agents",
        action="store_true",
        help="Also include generated_agents/*.lean candidates from generation_manifest.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_frac <= 0 or args.val_frac <= 0 or args.train_frac + args.val_frac >= 1:
        raise SystemExit("train/val fractions must be positive and leave room for test")

    dataset_root = Path(args.veribench_root).expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"VeriBench root not found: {dataset_root}")

    rows = list(iter_gold_rows(dataset_root, args.train_frac, args.val_frac))
    if args.include_generated_agents:
        rows.extend(iter_generated_rows(dataset_root, args.train_frac, args.val_frac))
    rows = sorted(rows, key=lambda r: (r.split, r.source_kind, r.family, r.task_id, r.variant_id))

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "veribench_manifest.jsonl", rows)
    for split in ("train", "val", "test"):
        write_jsonl(out_dir / f"{split}.jsonl", [r for r in rows if r.split == split])
    if args.smoke:
        smoke_rows = sorted(rows, key=lambda r: (r.family, r.task_id, r.variant_id))[:20]
        write_jsonl(out_dir / "smoke.jsonl", smoke_rows)
    write_summary(out_dir / "summary.json", rows, args)

    print(f"[data.setup] wrote {len(rows)} rows to {out_dir}")
    print(f"[data.setup] summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
