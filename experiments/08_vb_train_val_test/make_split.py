# TLDR: Build the portable VB train/val/test split (CE-ready content + SCSC-compatible task names) for Elyas' baselines.
"""Create the VeriBench train/val/test split package for training baselines (LM, EBT, Diffusion).

Unlike the manifest-only splitter in experiments/00_ar_pros_cons/data/setup.py, this
script EMBEDS file contents (python source + gold Lean) so the output is portable:
Elyas can train CE baselines from the JSONL alone, on any machine. Task names match
the SCSC metric's discovery convention (``<family>/<stem>``, see
``veribench_metric.utils.discover_gold_tasks``) so test-set generations can be scored
with SCSC against the same gold files.

Split assignment is IDENTICAL to the 2026-05-26 split in
experiments/00_ar_pros_cons (same SHA-256 bucket on ``task_id`` = lean stem,
same 80/10/10 fractions), so results remain comparable.

Usage:
    python make_split.py
    python make_split.py --veribench-root ~/veribench/veribench_dataset --output-dir splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

THEOREM_RE = re.compile(r"^\s*(?:noncomputable\s+)?(?:@\[[^\n]*\]\s*)?theorem\s+", re.MULTILINE)
SORRY_RE = re.compile(r"\b(?:sorry|admit)\b")
TACTIC_RE = re.compile(
    r"\b(?:simp|omega|aesop|rw|rfl|exact|apply|intro|intros|constructor|cases|induction|linarith|ring|norm_num)\b"
)
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


@dataclass(frozen=True)
class Row:
    split: str
    task_id: str          # lean stem; the split key (matches 00_ar_pros_cons assignment)
    task_name: str        # "<family>/<stem>" — SCSC metric task name (discover_gold_tasks)
    variant_id: str
    source_kind: str      # "gold" | "generated_agent"
    family: str
    rel_lean_path: str    # relative to veribench_dataset root
    rel_py_path: str | None
    pairing_method: str | None  # exact | snake | normdir | normfam | None
    paired: bool
    py_code: str | None
    lean_text: str
    line_count: int
    char_count: int
    theorem_count: int
    sorry_count: int
    tactic_count_proxy: int
    sha256_lean: str
    sha256_py: str | None


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


def snake(s: str) -> str:
    return CAMEL_RE.sub("_", s).lower()


def norm(s: str) -> str:
    return s.replace("_", "").lower()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_py(lean_rel: Path, py_root: Path) -> tuple[Path | None, str | None]:
    """Pair a gold lean file (path relative to lean_src/veribench) with its python source."""
    family = lean_rel.parts[0]
    mirror_dir = py_root / lean_rel.parent
    stem = lean_rel.stem
    for method, cand in (("exact", mirror_dir / f"{stem}.py"), ("snake", mirror_dir / f"{snake(stem)}.py")):
        if cand.exists():
            return cand, method
    if mirror_dir.exists():
        matches = sorted(p for p in mirror_dir.glob("*.py") if norm(p.stem) == norm(stem))
        if matches:
            return matches[0], "normdir"
    matches = sorted(p for p in (py_root / family).rglob("*.py") if norm(p.stem) == norm(stem))
    if matches:
        return matches[0], "normfam"
    return None, None


def gold_row(lean_path: Path, dataset_root: Path, train_frac: float, val_frac: float) -> Row:
    lean_root = dataset_root / "lean_src" / "veribench"
    py_root = dataset_root / "py_src"
    rel = lean_path.relative_to(lean_root)
    family = rel.parts[0]
    task_id = lean_path.stem
    task_name = f"{family}/{task_id}"
    py_path, method = find_py(rel, py_root)
    lean_text = read_text(lean_path)
    py_code = read_text(py_path) if py_path else None
    return Row(
        split=split_for_task(task_id, train_frac, val_frac),
        task_id=task_id,
        task_name=task_name,
        variant_id=task_id,
        source_kind="gold",
        family=family,
        rel_lean_path=str(lean_path.relative_to(dataset_root)),
        rel_py_path=str(py_path.relative_to(dataset_root)) if py_path else None,
        pairing_method=method,
        paired=py_path is not None,
        py_code=py_code,
        lean_text=lean_text,
        line_count=lean_text.count("\n") + 1,
        char_count=len(lean_text),
        theorem_count=len(THEOREM_RE.findall(lean_text)),
        sorry_count=len(SORRY_RE.findall(lean_text)),
        tactic_count_proxy=len(TACTIC_RE.findall(lean_text)),
        sha256_lean=sha(lean_text),
        sha256_py=sha(py_code) if py_code is not None else None,
    )


def agent_rows(dataset_root: Path, gold_by_task_id: dict[str, Row], train_frac: float, val_frac: float) -> list[Row]:
    manifest_path = dataset_root / "generated_agents" / "generation_manifest.json"
    if not manifest_path.exists():
        return []
    rows: list[Row] = []
    for entry in json.loads(manifest_path.read_text()):
        task_id = str(entry.get("task_id", "")).strip()
        agent = str(entry.get("agent", "agent_unknown")).strip()
        if not task_id:
            continue
        lean_path = dataset_root / "generated_agents" / f"{task_id}__{agent}.lean"
        if not lean_path.exists() and entry.get("output_path"):
            lean_path = dataset_root / "generated_agents" / Path(entry["output_path"]).name
        if not lean_path.exists():
            continue
        gold = gold_by_task_id.get(task_id)
        lean_text = read_text(lean_path)
        rows.append(
            Row(
                split=split_for_task(task_id, train_frac, val_frac),
                task_id=task_id,
                task_name=gold.task_name if gold else f"generated_agents/{task_id}",
                variant_id=f"{task_id}__{agent}",
                source_kind="generated_agent",
                family="generated_agents",
                rel_lean_path=str(lean_path.relative_to(dataset_root)),
                rel_py_path=gold.rel_py_path if gold else None,
                pairing_method=gold.pairing_method if gold else None,
                paired=bool(gold and gold.paired),
                py_code=gold.py_code if gold else None,
                lean_text=lean_text,
                line_count=lean_text.count("\n") + 1,
                char_count=len(lean_text),
                theorem_count=len(THEOREM_RE.findall(lean_text)),
                sorry_count=len(SORRY_RE.findall(lean_text)),
                tactic_count_proxy=len(TACTIC_RE.findall(lean_text)),
                sha256_lean=sha(lean_text),
                sha256_py=gold.sha256_py if gold else None,
            )
        )
    return rows


def write_jsonl(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veribench-root", default="~/veribench/veribench_dataset")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "splits"))
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    args = parser.parse_args()

    dataset_root = Path(args.veribench_root).expanduser().resolve()
    lean_root = dataset_root / "lean_src" / "veribench"
    if not lean_root.exists():
        raise SystemExit(f"VeriBench gold dir not found: {lean_root}")

    gold = [gold_row(p, dataset_root, args.train_frac, args.val_frac) for p in sorted(lean_root.rglob("*.lean"))]
    names = [r.task_name for r in gold]
    assert len(names) == len(set(names)), "task_name collision in gold set"
    assert all(r.lean_text.strip() for r in gold), "empty gold lean file"
    assert all(r.py_code and r.py_code.strip() for r in gold if r.paired), "paired row with empty py_code"

    gold_by_task_id = {r.task_id: r for r in gold}
    agents = agent_rows(dataset_root, gold_by_task_id, args.train_frac, args.val_frac)
    for r in agents:
        g = gold_by_task_id.get(r.task_id)
        assert g is None or g.split == r.split, f"agent variant {r.variant_id} split mismatch vs gold"

    split_of: dict[str, str] = {}
    for r in gold + agents:
        assert split_of.setdefault(r.task_id, r.split) == r.split, f"task {r.task_id} in two splits"

    gold = sorted(gold, key=lambda r: (r.split, r.family, r.task_name))
    agents = sorted(agents, key=lambda r: (r.split, r.task_name, r.variant_id))

    out_dir = Path(args.output_dir).resolve()
    for split in ("train", "val", "test"):
        write_jsonl(out_dir / f"{split}.jsonl", [r for r in gold if r.split == split])
    write_jsonl(out_dir / "agent_variants.jsonl", agents)
    write_jsonl(out_dir / "manifest.jsonl", gold + agents)
    smoke = [r for r in gold if r.split == "train" and r.paired][:14] \
        + [r for r in gold if r.split == "val" and r.paired][:3] \
        + [r for r in gold if r.split == "test" and r.paired][:3]
    write_jsonl(out_dir / "smoke.jsonl", smoke)

    def stats(rows: list[Row]) -> dict:
        return {
            "n": len(rows),
            "by_split": {s: sum(1 for r in rows if r.split == s) for s in ("train", "val", "test")},
            "paired": sum(1 for r in rows if r.paired),
            "unpaired": sum(1 for r in rows if not r.paired),
        }

    summary = {
        "created_for": "experiments/08_vb_train_val_test — CE + SCSC split for Elyas baselines (LM, EBT, Diffusion)",
        "veribench_root": str(dataset_root),
        "output_dir": str(out_dir),
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(1.0 - args.train_frac - args.val_frac, 10),
        "split_key": "task_id (gold lean stem), SHA-256 stable bucket — identical to experiments/00_ar_pros_cons",
        "gold": stats(gold),
        "agent_variants": stats(agents),
        "pairing_methods": {m: sum(1 for r in gold if r.pairing_method == m) for m in ("exact", "snake", "normdir", "normfam")},
        "unpaired_gold_task_names": [r.task_name for r in gold if not r.paired],
        "by_family_gold": dict(sorted({f: sum(1 for r in gold if r.family == f) for f in {r.family for r in gold}}.items())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"[make_split] gold: {stats(gold)}")
    print(f"[make_split] agent_variants: {stats(agents)}")
    print(f"[make_split] wrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
