#!/usr/bin/env python3
"""Tiny VeriBench EBM pilot smoke test.

This script is deliberately not the scientific result. It verifies that the
three-example manifest resolves to real VeriBench files, builds candidate pools
from gold files, generated agent attempts, and simple corruptions, then ranks
them with a deterministic toy energy.

The cluster run should replace the toy energy with a learned transformer energy
while preserving the same manifest/candidate-pool interface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = EXPERIMENT_DIR / "veribench_three_example_manifest.json"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"


@dataclass(frozen=True)
class Candidate:
    task_id: str
    candidate_id: str
    kind: str
    label: int
    source_path: str | None
    text: str


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_veribench_root(manifest: dict[str, Any], override: str | None) -> Path:
    root = Path(override) if override else Path(manifest["source_repo"])
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"VeriBench root does not exist: {root}. "
            "Pass --veribench-root or set VERIBENCH_ROOT."
        )
    return root


def resolve_source_path(path_text: str, manifest_source_repo: str, veribench_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()

    manifest_root = Path(manifest_source_repo).expanduser()
    if path.is_absolute():
        try:
            rel = path.relative_to(manifest_root)
        except ValueError as exc:
            raise FileNotFoundError(f"Missing absolute source path: {path}") from exc
        candidate = veribench_root / rel
    else:
        candidate = veribench_root / path

    if not candidate.exists():
        raise FileNotFoundError(f"Could not resolve source path: {path_text} -> {candidate}")
    return candidate.resolve()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def first_replacement(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old in text:
            return text.replace(old, new, 1)
    return text + "\n\n/- corrupted: no known replacement site found -/\n"


def make_corruptions(task_id: str, gold_text: str) -> list[Candidate]:
    no_imports = "\n".join(
        line for line in gold_text.splitlines() if not line.strip().startswith("import ")
    )
    no_imports = "/- corrupted: imports removed -/\n" + no_imports

    flipped_expectation = first_replacement(
        gold_text,
        [
            ("= true", "= false"),
            ("= false", "= true"),
            ("= some 0", "= none"),
            ("= none", "= some 0"),
            ("= 0", "= 1"),
            ("= 3", "= 4"),
            ("= 5", "= 6"),
        ],
    )

    broken_statement = first_replacement(
        gold_text,
        [
            ("theorem ", "theorem corrupted_"),
            ("example :", "example : False -- corrupted expected failure\n/-- original follows -/\nexample :"),
            ("def ", "def corrupted_"),
        ],
    )

    return [
        Candidate(task_id, "corrupt_no_imports", "corruption", 0, None, no_imports),
        Candidate(task_id, "corrupt_flipped_expectation", "corruption", 0, None, flipped_expectation),
        Candidate(task_id, "corrupt_broken_statement", "corruption", 0, None, broken_statement),
    ]


def generated_agent_paths(veribench_root: Path, lean_source: Path, limit: int) -> list[Path]:
    generated_root = veribench_root / "veribench_dataset" / "generated_agents"
    if not generated_root.exists():
        return []
    stem = lean_source.stem
    return sorted(generated_root.glob(f"{stem}__agent*.lean"))[:limit]


def build_candidate_pools(
    manifest: dict[str, Any],
    veribench_root: Path,
    max_generated_agents: int,
) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = {}
    manifest_source_repo = manifest["source_repo"]

    for example in manifest["examples"]:
        task_id = example["id"]
        lean_path = resolve_source_path(example["lean_source"], manifest_source_repo, veribench_root)
        gold_text = read_text(lean_path)
        candidates: list[Candidate] = [
            Candidate(task_id, "gold", "gold", 1, str(lean_path), gold_text)
        ]

        for agent_path in generated_agent_paths(veribench_root, lean_path, max_generated_agents):
            candidates.append(
                Candidate(
                    task_id,
                    agent_path.stem,
                    "generated_agent",
                    0,
                    str(agent_path.resolve()),
                    read_text(agent_path),
                )
            )

        candidates.extend(make_corruptions(task_id, gold_text))
        pools[task_id] = candidates

    return pools


def toy_energy(candidate: Candidate) -> float:
    """Deterministic smoke-test energy.

    This intentionally bakes in a source prior so the smoke test checks
    plumbing, not scientific ranking quality. The learned transformer script
    should remove this prior.
    """

    kind_prior = {
        "gold": 0.0,
        "generated_agent": 3.0,
        "corruption": 8.0,
    }[candidate.kind]
    line_count = candidate.text.count("\n") + 1
    sorry_penalty = candidate.text.count("sorry") * 0.02
    error_token_penalty = sum(
        candidate.text.count(token)
        for token in ["corrupted", "False -- corrupted", "no known replacement"]
    ) * 0.5
    return kind_prior + min(line_count, 3000) * 0.0001 + sorry_penalty + error_token_penalty


def summarize_candidate(candidate: Candidate, rank: int, energy: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "energy": energy,
        "task_id": candidate.task_id,
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "label": candidate.label,
        "source_path": candidate.source_path,
        "num_chars": len(candidate.text),
        "num_lines": candidate.text.count("\n") + 1,
        "num_sorry": candidate.text.count("sorry"),
    }


def write_candidate_files(pools: dict[str, list[Candidate]], output_dir: Path) -> None:
    pool_root = output_dir / "candidate_pools"
    for task_id, candidates in pools.items():
        task_dir = pool_root / safe_name(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            filename = f"{safe_name(candidate.candidate_id)}.lean"
            (task_dir / filename).write_text(candidate.text, encoding="utf-8")


def run_smoke_test(
    manifest_path: Path,
    veribench_root_override: str | None,
    output_dir: Path,
    max_generated_agents: int,
    write_files: bool,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    veribench_root = resolve_veribench_root(manifest, veribench_root_override)
    pools = build_candidate_pools(manifest, veribench_root, max_generated_agents)

    output_dir.mkdir(parents=True, exist_ok=True)
    if write_files:
        write_candidate_files(pools, output_dir)

    task_results = []
    all_gold_top = True
    for task_id, candidates in pools.items():
        ranked = sorted(((toy_energy(c), c) for c in candidates), key=lambda pair: pair[0])
        ranked_summaries = [
            summarize_candidate(candidate, rank=i + 1, energy=energy)
            for i, (energy, candidate) in enumerate(ranked)
        ]
        gold_rank = next(item["rank"] for item in ranked_summaries if item["candidate_id"] == "gold")
        all_gold_top = all_gold_top and gold_rank == 1
        task_results.append(
            {
                "task_id": task_id,
                "num_candidates": len(candidates),
                "gold_rank": gold_rank,
                "top_candidate": ranked_summaries[0],
                "ranked_candidates": ranked_summaries,
            }
        )

    report = {
        "status": "pass" if all_gold_top else "fail",
        "manifest": str(manifest_path.resolve()),
        "veribench_root": str(veribench_root),
        "output_dir": str(output_dir.resolve()),
        "energy": "toy_source_prior_smoke_test",
        "task_results": task_results,
    }

    report_path = output_dir / "smoke_test_rankings.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--veribench-root", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-generated-agents", type=int, default=3)
    parser.add_argument("--write-candidate-files", action="store_true")
    parser.add_argument("--require-gold-top", action="store_true")
    args = parser.parse_args()

    report = run_smoke_test(
        manifest_path=args.manifest,
        veribench_root_override=args.veribench_root,
        output_dir=args.output_dir,
        max_generated_agents=args.max_generated_agents,
        write_files=args.write_candidate_files,
    )

    for task in report["task_results"]:
        print(
            f"{task['task_id']}: gold_rank={task['gold_rank']} "
            f"num_candidates={task['num_candidates']} "
            f"top={task['top_candidate']['candidate_id']}"
        )
    print(f"status={report['status']} report={report['output_dir']}/smoke_test_rankings.json")

    if args.require_gold_top and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
