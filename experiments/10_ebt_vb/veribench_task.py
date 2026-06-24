#!/usr/bin/env python3
"""Task-level abstraction for a single VeriBench row.

The class is intentionally small and explicit: it binds one task identifier to
its prompt token count, Goedel hidden states, and target Lean proof tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Iterator

import json
import tempfile
import torch
import subprocess
from statistics import median
import sys
from safetensors.torch import load_file

from veribench_context_gold_dataloader import DEFAULT_DATA_DIR


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_vocab(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _activation_dtype_from_str(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    return torch.float32


def _resolve_safetensors_path(path_value: str, *, data_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidate = (Path(data_dir) / path).resolve()
    if candidate.exists():
        return candidate
    return path


VERIBENCH_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "09_vb_testing_ipynb" / "veribench"

EXAMPLE_DECL_RE = re.compile(r"^\s*example\s+", re.MULTILINE)
THEOREM_DECL_RE = re.compile(r"^(?:noncomputable\s+)?(?:@\[[^\]]*\]\s*)?theorem\s+(\w+)", re.MULTILINE)
SORRY_ADMIT_RE = re.compile(r"\b(sorry|admit)\b")


def _strip_code_fences(text: str) -> str:
    fence_blocks = re.findall(r"```(?:lean)?\s*([\s\S]*?)```", text)
    if not fence_blocks:
        return text.strip()
    return max(fence_blocks, key=len).strip()


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return None


def _count_examples(source: str) -> int:
    return len(EXAMPLE_DECL_RE.findall(source))


def _extract_theorem_blocks(source: str) -> list[str]:
    return [
        block
        for block in re.split(r"(?=^(?:noncomputable\s+)?(?:@\[[^\]]*\]\s*)?theorem\s+)", source, flags=re.MULTILINE)
        if THEOREM_DECL_RE.match(block)
    ]


def _analyze_theorems(source: str) -> tuple[int, int, int, list[str]]:
    blocks = _extract_theorem_blocks(source)
    names = [THEOREM_DECL_RE.match(block).group(1) for block in blocks if THEOREM_DECL_RE.match(block)]  # type: ignore[union-attr]
    total = len(blocks)
    sorry_count = sum(1 for block in blocks if SORRY_ADMIT_RE.search(block))
    return total, total - sorry_count, sorry_count, names


def _compile_lean_file(
    lean_file: str,
    lake_dir: Path | str,
    timeout: int = 600,
) -> tuple[bool, str, str, int]:
    """Compile a Lean file with `lake env lean`. Returns success, stdout, stderr, code."""
    cmd = ["lake", "env", "lean", str(lean_file)]
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(lake_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stderr = result.stderr or ""
        success = result.returncode == 0 and "error" not in stderr.lower()
        return success, result.stdout or "", stderr, result.returncode
    except subprocess.TimeoutExpired:
        return False, "", "timeout", -1
    except FileNotFoundError:
        return False, "", "lake not found", -1


def _judge_equivalence_te1(candidate_src: str, gold_src: str, model: str = "claude-opus-4-20250514", repeats: int = 1) -> tuple[float, dict[str, Any]]:
    # Optional import so this function is usable even without veribench install.
    try:
        from veribench.veribench_metric.judge.llm_judge import (
            JUDGE_MAX_SCORE as _JUDGE_MAX_SCORE,
            judge_file_coverage,
        )
    except Exception:
        try:
            vb_root = VERIBENCH_DEFAULT_ROOT
            if str(vb_root) not in sys.path:
                sys.path.insert(0, str(vb_root))
            from veribench_metric.judge.llm_judge import (
                JUDGE_MAX_SCORE as _JUDGE_MAX_SCORE,
                judge_file_coverage,
            )
        except Exception as e:  # pragma: no cover - import environment dependent
            return 0.0, {"reason": "judge_import_failed", "error": repr(e)}

    scores: list[float] = []
    errors: list[str] = []
    for _ in range(max(1, repeats)):
        try:
            raw = judge_file_coverage(candidate_file=candidate_src, gold_file=gold_src, model=model)
            normalized = max(0.0, min(1.0, float(raw) / float(_JUDGE_MAX_SCORE)))
            scores.append(normalized)
        except Exception as e:  # pragma: no cover - external dependency failure
            errors.append(repr(e))
    if not scores:
        return 0.0, {"reason": "te1_failed", "errors": errors}
    return float(median(scores)), {"scores": scores, "repeats": len(scores), "errors": errors, "model": model}


@dataclass(frozen=True)
class VeriBenchTask:
    """Encapsulates a single VeriBench task row with prompt context and targets."""

    task_name: str
    split: str
    family: str | None
    prompt_tokens: int
    generated_tokens: int
    hidden_dim: int
    safetensors_path: str
    target_local_ids: tuple[int, ...]
    target_original_ids: tuple[int, ...] | None = None
    source_kind: str | None = None
    rel_py_path: str | None = None
    rel_lean_path: str | None = None
    task_id: str | None = None
    bos_id: int = 0
    eos_id: int = 1
    unk_id: int = 2
    pad_id: int = 0
    activation_dtype: str = "bf16"
    data_dir: Path = DEFAULT_DATA_DIR

    _context_activations: torch.Tensor | None = None  # type: ignore[type-arg]

    @property
    def target_tokens(self) -> int:
        return len(self.target_local_ids)

    @property
    def context_activations_path(self) -> Path:
        return _resolve_safetensors_path(self.safetensors_path, data_dir=self.data_dir)

    @property
    def default_veribench_root(self) -> Path:
        """Best-effort location of the local veribench checkout used by this repo."""
        return VERIBENCH_DEFAULT_ROOT

    @property
    def default_lake_dir(self) -> Path:
        return self.default_veribench_root / "veribench_dataset" / "lean_src"

    @property
    def default_lake_dir_or_none(self) -> Path | None:
        return self.default_lake_dir if self.default_lake_dir.exists() else None

    def resolve_gold_lean_path(self, veribench_root: Path | None = None) -> Path | None:
        """Resolve this task's reference gold Lean file path, if available."""
        if not self.rel_lean_path:
            return None
        candidate = Path(self.rel_lean_path).expanduser()
        if candidate.is_absolute():
            return candidate if candidate.exists() else None
        roots = [veribench_root or self.default_veribench_root]
        for root in roots:
            direct = root / candidate
            if direct.exists():
                return direct
            dataset_prefixed = root / "veribench_dataset" / candidate
            if dataset_prefixed.exists():
                return dataset_prefixed
        return None

    def evaluate_lean_output(
        self,
        lean_output: str,
        *,
        lake_dir: Path | None = None,
        veribench_root: Path | None = None,
        compile_timeout: int = 600,
        skip_te1: bool = False,
        te1_model: str = "claude-opus-4-20250514",
        te1_repeats: int = 1,
    ) -> dict[str, Any]:
        """Evaluate one generated Lean output against SCSC-style factors.

        Returns a dictionary with:
        - IC1: fraction of candidate examples that pass on candidate file.
        - IC2: fraction of candidate theorems proved (non-sorry/admit).
        - TE1: theorem-coverage judge score normalized to [0,1]. If unavailable, 0.0 with reason.
        - D1/D2: same checks on gold file.
        """
        source = _strip_code_fences(lean_output)
        lake_dir = Path(lake_dir) if lake_dir is not None else self.default_lake_dir_or_none
        if lake_dir is None:
            lake_dir = Path("/tmp")

        tmp_root = str(lake_dir) if lake_dir.exists() else None
        with tempfile.TemporaryDirectory(prefix="veribench_task_eval_", dir=tmp_root) as work_dir:
            candidate_path = Path(work_dir) / f"{self.task_name.replace('/', '_')}.lean"
            candidate_path.write_text(source if source else "", encoding="utf-8")

            candidate_compile = _compile_lean_file(candidate_path, lake_dir=lake_dir, timeout=compile_timeout)
            candidate_stdout = candidate_compile[1]
            candidate_stderr = candidate_compile[2]
            candidate_success = candidate_compile[0]
            candidate_return_code = candidate_compile[3]

            candidate_examples = _count_examples(source)
            if candidate_examples == 0:
                ic1_score = 0.0
                ic1_info: dict[str, Any] = {
                    "n_tests": 0,
                    "reason": "no_tests",
                }
            elif candidate_success:
                ic1_score = 1.0
                ic1_info = {"n_tests": candidate_examples, "all_pass": True}
            else:
                test_errors = len(re.findall(r"(native_decide.*failed|tactic.*failed|: error:)", candidate_stderr))
                ic1_score = max(0.0, (candidate_examples - test_errors) / candidate_examples)
                ic1_info = {
                    "n_tests": candidate_examples,
                    "n_errors": test_errors,
                    "compile_success": False,
                }

            theorem_total, theorem_proven, theorem_sorry, theorem_names = _analyze_theorems(source)
            if theorem_total == 0:
                ic2_score = 0.0
                ic2_info = {"total": 0, "reason": "no_theorems"}
            elif not candidate_success:
                ic2_score = 0.0
                ic2_info = {
                    "total": theorem_total,
                    "proven": 0,
                    "sorry": theorem_sorry,
                    "compile_success": False,
                }
            else:
                ic2_score = theorem_proven / theorem_total if theorem_total > 0 else 0.0
                ic2_info = {
                    "total": theorem_total,
                    "proven": theorem_proven,
                    "sorry": theorem_sorry,
                    "theorem_names": theorem_names,
                    "compile_success": True,
                }

            gold_path = self.resolve_gold_lean_path(veribench_root=veribench_root)
            gold_stdout = ""
            gold_stderr = ""
            gold_success = False
            gold_return_code = None
            if gold_path is None:
                d1_score = 0.0
                d2_score = 0.0
                d1_info: dict[str, Any] = {"reason": "gold_file_missing"}
                d2_info: dict[str, Any] = {"reason": "gold_file_missing"}
                te1_score = 0.0
                te1_info: dict[str, Any] = {"reason": "gold_file_missing", "skipped": True}
            else:
                gold_source = _read_text_file(gold_path) or ""
                gold_compile = _compile_lean_file(
                    str(gold_path),
                    lake_dir=lake_dir,
                    timeout=compile_timeout,
                )
                gold_success = gold_compile[0]
                gold_stdout = gold_compile[1]
                gold_stderr = gold_compile[2]
                gold_return_code = gold_compile[3]

                gold_examples = _count_examples(gold_source)
                if gold_examples == 0:
                    d1_score = 0.0
                    d1_info = {"n_tests": 0, "reason": "no_tests"}
                elif not gold_success:
                    d1_score = 0.0
                    d1_info = {"n_tests": gold_examples, "compile_success": False}
                else:
                    d1_score = 1.0
                    d1_info = {"n_tests": gold_examples, "compile_success": True}

                gold_total, gold_proven, gold_sorry, gold_names = _analyze_theorems(gold_source)
                if gold_total == 0:
                    d2_score = 0.0
                    d2_info = {"total": 0, "reason": "no_theorems"}
                elif not gold_success:
                    d2_score = 0.0
                    d2_info = {"total": gold_total, "proven": 0, "sorry": gold_sorry, "compile_success": False}
                else:
                    d2_score = gold_proven / gold_total if gold_total > 0 else 0.0
                    d2_info = {
                        "total": gold_total,
                        "proven": gold_proven,
                        "sorry": gold_sorry,
                        "theorem_names": gold_names,
                        "compile_success": True,
                    }

                if skip_te1:
                    te1_score = 0.0
                    te1_info = {"reason": "skipped", "skipped": True}
                else:
                    te1_score, te1_meta = _judge_equivalence_te1(
                        source,
                        gold_source,
                        model=te1_model,
                        repeats=te1_repeats,
                    )
                    te1_info = {"candidate_file": str(candidate_path), "gold_file": str(gold_path), **te1_meta}

        return {
            "IC1": ic1_score,
            "IC2": ic2_score,
            "TE1": te1_score,
            "TC1": te1_score,  # backward-compatible alias used in existing Veribench metric code
            "D1": d1_score,
            "D2": d2_score,
            "S_tilde": (ic1_score * ic2_score * te1_score * d1_score * d2_score) ** 0.2,
            "details": {
                "IC1": ic1_info,
                "IC2": ic2_info,
                "TE1": te1_info,
                "D1": d1_info,
                "D2": d2_info,
                "compile": {
                    "candidate": {
                        "success": candidate_success,
                        "return_code": candidate_return_code,
                        "stderr": candidate_stderr[:1024],
                        "stdout": candidate_stdout[:1024],
                    },
                    "gold": {
                        "path": str(gold_path) if gold_path is not None else None,
                        "success": gold_success if gold_path is not None else None,
                        "return_code": gold_return_code if gold_path is not None else None,
                        "stderr": gold_stderr[:1024] if gold_path is not None else "",
                        "stdout": gold_stdout[:1024] if gold_path is not None else "",
                    },
                },
            },
        }

    @classmethod
    def load_vocab(cls, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[dict[str, Any], int]:
        vocab = _read_vocab(Path(data_dir) / "vocab.json")
        vocab_size = int(vocab["vocab_size"])
        return vocab, vocab_size

    @classmethod
    def from_manifest_row(
        cls,
        row: dict[str, Any],
        *,
        data_dir: Path = DEFAULT_DATA_DIR,
        activation_dtype: str = "bf16",
    ) -> "VeriBenchTask":
        if row["prompt_tokens"] <= 0:
            raise ValueError(f"{row.get('task_name')} has no prompt activations")
        target_local_ids = tuple(int(x) for x in row["target_local_ids"])
        target_original_ids = (
            tuple(int(x) for x in row["target_original_ids"])
            if row.get("target_original_ids") is not None
            else None
        )

        vocab = _read_vocab(Path(data_dir) / "vocab.json")
        return cls(
            task_name=str(row["task_name"]),
            split=str(row["split"]),
            family=row.get("family"),
            prompt_tokens=int(row["prompt_tokens"]),
            generated_tokens=int(row.get("generated_tokens", 0)),
            hidden_dim=int(row.get("hidden_dim", 4096)),
            safetensors_path=str(row["safetensors_path"]),
            target_local_ids=target_local_ids,
            target_original_ids=target_original_ids,
            source_kind=row.get("source_kind"),
            rel_py_path=row.get("rel_py_path"),
            rel_lean_path=row.get("rel_lean_path"),
            task_id=row.get("task_id"),
            bos_id=int(vocab["local_bos_id"]),
            eos_id=int(vocab["local_eos_id"]),
            unk_id=int(vocab["local_unk_id"]),
            pad_id=int(vocab["local_pad_id"]),
            activation_dtype=activation_dtype,
            data_dir=Path(data_dir),
            _context_activations=None,
        )

    @classmethod
    def from_manifest_path(
        cls,
        task_name: str,
        *,
        manifest_path: Path | None = None,
        data_dir: Path = DEFAULT_DATA_DIR,
        split: str | None = None,
        activation_dtype: str = "bf16",
    ) -> "VeriBenchTask":
        if manifest_path is None:
            manifest_path = Path(data_dir) / "manifest.jsonl"
        rows = _read_jsonl(manifest_path)
        for row in rows:
            if row.get("task_name") != task_name:
                continue
            if split is not None and str(row.get("split")) != split:
                continue
            return cls.from_manifest_row(row, data_dir=data_dir, activation_dtype=activation_dtype)
        raise KeyError(f"task_name {task_name!r} not found in {manifest_path}")

    @classmethod
    def iter_tasks(
        cls,
        *,
        split: str | None = None,
        data_dir: Path = DEFAULT_DATA_DIR,
        max_items: int | None = None,
        activation_dtype: str = "bf16",
    ) -> Iterator["VeriBenchTask"]:
        rows = _read_jsonl(Path(data_dir) / "manifest.jsonl")
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if max_items is not None:
            rows = rows[:max_items]
        for row in rows:
            yield cls.from_manifest_row(row, data_dir=data_dir, activation_dtype=activation_dtype)

    def load_prompt_activations(self) -> torch.Tensor:
        if self._context_activations is not None:
            return self._context_activations
        tensors = load_file(self.context_activations_path, device="cpu")
        if "hidden_states" not in tensors:
            raise KeyError(f"hidden_states not found in {self.context_activations_path}")
        activations = tensors["hidden_states"][: self.prompt_tokens]
        activations = activations.to(_activation_dtype_from_str(self.activation_dtype)).contiguous()
        if activations.shape[0] != self.prompt_tokens:
            raise ValueError(
                f"{self.task_name}: expected {self.prompt_tokens} prompt states, got {activations.shape[0]}"
            )
        if activations.shape[1] != self.hidden_dim:
            raise ValueError(
                f"{self.task_name}: expected hidden_dim {self.hidden_dim}, got {activations.shape[1]}"
            )
        # cache by returning a copy-free reference (frozen dataclass requires object mutation below)
        object.__setattr__(self, "_context_activations", activations)
        return activations

    def read_prompt_tokens(self) -> int:
        return self.prompt_tokens

    def read_target_tokens(self) -> tuple[int, ...]:
        return self.target_local_ids

    def as_ebt_sample(self) -> dict[str, torch.Tensor]:
        context = self.load_prompt_activations()
        target = torch.tensor(self.target_local_ids, dtype=torch.long)
        decoder_input_ids = torch.cat([torch.tensor([self.bos_id], dtype=torch.long), target])
        labels = torch.cat([target, torch.tensor([self.eos_id], dtype=torch.long)])
        return {
            "context_activations": context,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "context_token_count": torch.tensor(context.shape[0], dtype=torch.long),
            "target_token_count": torch.tensor(labels.shape[0], dtype=torch.long),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "split": self.split,
            "family": self.family,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "hidden_dim": self.hidden_dim,
            "safetensors_path": self.safetensors_path,
            "target_local_ids": list(self.target_local_ids),
            "target_original_ids": list(self.target_original_ids) if self.target_original_ids is not None else None,
            "source_kind": self.source_kind,
            "rel_py_path": self.rel_py_path,
            "rel_lean_path": self.rel_lean_path,
            "task_id": self.task_id,
        }
