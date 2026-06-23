#!/usr/bin/env python3
"""Generate Lean for VeriBench Python tasks with Goedel Prover V2 8B.

This uses SGLang's in-process Engine. By default it makes GPUs 0, 1, and 2
visible and launches three data-parallel model replicas with tensor parallel
size 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from tqdm import tqdm


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_VERIBENCH_ROOT = HERE / "veribench"
DEFAULT_MODEL = "Goedel-LM/Goedel-Prover-V2-8B"
DEFAULT_OUT_DIR = HERE / "results" / "goedel_prover_v2_8b_sglang_896"
DEFAULT_SPLIT_DIR = HERE.parent / "08_vb_train_val_test" / "splits"
DEFAULT_GOLD_EXAMPLE_PY = DEFAULT_VERIBENCH_ROOT / "veribench_dataset" / "gold_examples" / "py_src" / "my_add.py"
DEFAULT_GOLD_EXAMPLE_LEAN = DEFAULT_VERIBENCH_ROOT / "veribench_dataset" / "gold_examples" / "lean_src" / "my_add.lean"


PROMPT_TEMPLATE = """# Task: Translate Python to a complete VeriBench Lean 4 file

You must translate the target Python program into one complete Lean 4.22 file
in the VeriBench standard format.

## Non-negotiable output rules

- Output only Lean 4 source code. Do not output markdown fences.
- The first non-whitespace character must be `/` from a Lean module comment or
  `i` from an `import` command. Never begin with a number, theorem, explanation,
  or ``` fence.
- Generate a full file, not a single theorem proof.
- Do not write markdown headings such as `### Implementation` or internal
  ```lean code fences. Use Lean comments `/-! ... -/` for section labels.
- Include definitions before theorems. Do not output only `theorem ... := by`.
- Prefer `import Mathlib` so tactics such as `norm_num`, `omega`, and `aesop`
  are available.
- Use Lean 4 syntax compatible with Lean 4.22.
- Use `:= sorry` for difficult theorem proofs. Do not write fake proof scripts
  such as `trivial`, `assumption`, or repeated `have` blocks that do not prove
  the goal.
- Avoid theorem-prover contest style output. This is code translation, not
  theorem proving from a proposition.

## Required VeriBench file shape

Follow this order:

1. Optional `import Std` or `import Mathlib`
2. `set_option linter.unusedVariables false`
3. `/-! ... -/` module docstring with the section list
4. `namespace <TaskName>`
5. Functional implementation `def ...`
6. Examples and tests using `example ... := by native_decide` and optional `#eval`
7. `def Pre ... : Prop := ...`
8. Reusable property propositions and theorem statements
9. `def Post_prop ... : Prop := ...`
10. `theorem correctness_thm ... : Pre ... -> Post_prop ... := sorry`
11. Imperative implementation using `Id.run do`
12. Imperative examples/tests
13. Equivalence theorem between functional and imperative implementations
14. `end <TaskName>`

For Float examples, prefer executable Boolean tests with `==` when `native_decide`
cannot prove propositional equality.

## VeriBench format example

The following is a style/format example only. Do not copy its task semantics.

### Example Python

BEGIN_EXAMPLE_PYTHON
{example_py_code}
END_EXAMPLE_PYTHON

### Example Lean 4

BEGIN_EXAMPLE_LEAN
{example_lean_code}
END_EXAMPLE_LEAN

## Target Python source

Translate only this target Python source:

```python
{target_py_code}
```

## Output Requirements

Return only the complete Lean 4 file for the target Python source."""


@dataclass(frozen=True)
class Task:
    task_name: str
    split: str
    task_id: str | None
    py_path: str
    py_code: str


@dataclass
class GenerationRecord:
    task_name: str
    split: str
    task_id: str | None
    py_path: str
    output_lean_path: str
    raw_output_path: str
    prompt_path: str
    model: str
    prompt_chars: int
    raw_generated_chars: int
    generated_chars: int
    elapsed_sec: float
    raw_output_text: str
    output_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Goedel-Prover-V2-8B on all VeriBench py_src tasks with SGLang."
    )
    parser.add_argument("--veribench-root", type=pathlib.Path, default=DEFAULT_VERIBENCH_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gold-example-py", type=pathlib.Path, default=DEFAULT_GOLD_EXAMPLE_PY)
    parser.add_argument("--gold-example-lean", type=pathlib.Path, default=DEFAULT_GOLD_EXAMPLE_LEAN)
    parser.add_argument(
        "--split-jsonl",
        action="append",
        type=pathlib.Path,
        default=None,
        help="Split JSONL to run. Can be repeated. Uses py_code only; lean_text is ignored.",
    )
    parser.add_argument("--gpus", default="0,1,2", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--data-parallel-size", type=int, default=3)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-strip-markdown", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true", help="Do not apply tokenizer chat template.")
    parser.add_argument("--dry-run", action="store_true", help="Validate task discovery/prompt shape only.")
    return parser.parse_args()


def paired_lean_path(py_file: pathlib.Path, py_src: pathlib.Path, lean_src: pathlib.Path) -> pathlib.Path:
    rel = py_file.relative_to(py_src)
    return lean_src / rel.with_suffix(".lean")


def load_tasks(veribench_root: pathlib.Path) -> list[Task]:
    py_src = veribench_root / "veribench_dataset" / "py_src"
    lean_src = veribench_root / "veribench_dataset" / "lean_src" / "veribench"
    if not py_src.exists():
        raise FileNotFoundError(f"Missing Python dataset directory: {py_src}")
    if not lean_src.exists():
        raise FileNotFoundError(f"Missing Lean gold dataset directory: {lean_src}")
    gold_count = sum(1 for _ in lean_src.rglob("*.lean"))
    if gold_count != 896:
        raise RuntimeError(f"Expected 896 Lean gold files under {lean_src}, found {gold_count}")

    tasks: list[Task] = []
    for py_file in sorted(py_src.rglob("*.py")):
        rel = py_file.relative_to(py_src)
        tasks.append(
            Task(
                task_name=rel.with_suffix("").as_posix(),
                split="all",
                task_id=rel.stem,
                py_path=str(py_file),
                py_code=py_file.read_text(encoding="utf-8"),
            )
        )
    return tasks


def load_split_tasks(split_jsonls: list[pathlib.Path], out_dir: pathlib.Path) -> list[Task]:
    tasks: list[Task] = []
    skipped: list[dict[str, Any]] = []
    for split_path in split_jsonls:
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Split rows embed gold `lean_text` for training portability.
                # Drop it immediately; generation prompts must use Python only.
                row.pop("lean_text", None)
                py_code = row.get("py_code")
                task_name = row.get("task_name") or row.get("variant_id") or row.get("task_id")
                if not py_code:
                    skipped.append(
                        {
                            "split": row.get("split"),
                            "task_name": task_name,
                            "task_id": row.get("task_id"),
                            "reason": "missing py_code; not prompted to avoid gold leakage",
                        }
                    )
                    continue
                tasks.append(
                    Task(
                        task_name=str(task_name),
                        split=str(row.get("split") or split_path.stem),
                        task_id=row.get("task_id"),
                        py_path=str(row.get("rel_py_path") or ""),
                        py_code=py_code,
                    )
                )
    if skipped:
        write_json(out_dir / "skipped_missing_py_code.json", skipped)
    return tasks


def read_text(path: pathlib.Path) -> str:
    return path.expanduser().read_text(encoding="utf-8").strip()


def build_prompt(task: Task, example_py_code: str, example_lean_code: str) -> str:
    # Deliberately include only target Python source. Per-task gold Lean is never
    # interpolated; the model must predict/recover the target formalization.
    return PROMPT_TEMPLATE.format(
        example_py_code=example_py_code,
        example_lean_code=example_lean_code,
        target_py_code=task.py_code.strip(),
    )


def maybe_apply_chat_template(tokenizer: Any, prompt: str, raw_prompt: bool) -> str:
    if raw_prompt:
        return prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Lean 4.22 and VeriBench code-translation expert. "
                "Return only a complete Lean 4 source file. Do not use markdown "
                "fences, prose, theorem-prover snippets, or single-theorem answers."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:lean4|lean)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        stripped = match.group(1).strip()
    else:
        stripped = re.sub(r"^\s*```(?:lean4|lean)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped).strip()
    if stripped.startswith("4\n"):
        stripped = stripped[2:].lstrip()
    return stripped


def namespace_for_task(task_name: str) -> str:
    base = pathlib.PurePosixPath(task_name).name
    chunks = [chunk for chunk in re.split(r"[^A-Za-z0-9]+", base) if chunk]
    namespace = "".join(chunk[:1].upper() + chunk[1:] for chunk in chunks)
    if not namespace:
        namespace = "GeneratedTask"
    if namespace[0].isdigit():
        namespace = f"Task{namespace}"
    return namespace


def lean_file_prefix(task: Task) -> str:
    namespace = namespace_for_task(task.task_name)
    title = pathlib.PurePosixPath(task.task_name).name
    return (
        "import Mathlib\n"
        "set_option linter.unusedVariables false\n\n"
        "set_option linter.unreachableTactic false\n"
        "set_option linter.unusedTactic false\n\n"
        "/-!\n"
        f"VeriBench translation for `{title}`.\n"
        "-/\n\n"
        f"namespace {namespace}\n\n"
    )


def extract_lean_candidate(text: str) -> str:
    text = strip_markdown_fence(text)
    for marker in ("### Complete Lean 4 Proof", "## Complete Lean 4 Proof", "Complete Lean 4 Proof"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def fix_comma_binders(line: str) -> str:
    pattern = re.compile(
        r"\(\s*([A-Za-z_][A-Za-z0-9_']*)\s*:\s*([^,()]+?)\s*,\s*"
        r"([A-Za-z_][A-Za-z0-9_']*)\s*:\s*([^)]+?)\s*\)"
    )
    previous = None
    while previous != line:
        previous = line
        line = pattern.sub(r"(\1 : \2) (\3 : \4)", line)
    return line


def normalize_lean_continuation(text: str) -> str:
    text = extract_lean_candidate(text)
    lines: list[str] = []
    seen_decls: dict[str, int] = {}
    skip_proof_body = False
    for line in text.splitlines():
        stripped = line.strip()
        if skip_proof_body:
            if line.startswith((" ", "\t")) or stripped.startswith(("·", "<;>")):
                continue
            skip_proof_body = False
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            continue
        if stripped in {"/-!", "/-", "-/"}:
            continue
        if stripped.startswith("BEGIN_") or stripped.startswith("END_"):
            continue
        if re.match(r"^import\s+", stripped):
            continue
        if stripped.startswith("set_option "):
            continue
        if re.match(r"^namespace\s+", stripped):
            continue
        if re.match(r"^end(?:\s+|$)", stripped):
            continue
        line = fix_comma_binders(line)
        decl_match = re.match(r"^(\s*)(def|theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)\b", line)
        if decl_match:
            name = decl_match.group(3)
            count = seen_decls.get(name, 0)
            seen_decls[name] = count + 1
            if count:
                replacement = f"{name}_dup{count}"
                start, end = decl_match.span(3)
                line = line[:start] + replacement + line[end:]
                stripped = line.strip()
        if stripped.startswith("example ") and ":= by" in line:
            lines.append(line.split(":= by", 1)[0].rstrip() + " := by sorry")
            skip_proof_body = True
            continue
        if stripped.startswith(("theorem ", "lemma ")) and ":= by" in line:
            lines.append(line.split(":= by", 1)[0].rstrip() + " := by sorry")
            skip_proof_body = True
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def build_lean_output(task: Task, raw_text: str, no_strip_markdown: bool) -> str:
    if no_strip_markdown:
        return raw_text.strip()
    namespace = namespace_for_task(task.task_name)
    continuation = normalize_lean_continuation(raw_text)
    prefix = lean_file_prefix(task)
    if not continuation:
        continuation = "/- Empty model continuation. -/"
    return f"{prefix}{continuation.rstrip()}\n\nend {namespace}"


def output_path_for_task(out_dir: pathlib.Path, task_name: str) -> pathlib.Path:
    return out_dir / "lean_outputs" / f"{task_name}.lean"


def raw_output_path_for_task(out_dir: pathlib.Path, task_name: str) -> pathlib.Path:
    return out_dir / "raw_model_outputs" / f"{task_name}.txt"


def prompt_path_for_task(out_dir: pathlib.Path, task_name: str) -> pathlib.Path:
    return out_dir / "prompts" / f"{task_name}.txt"


def read_completed(jsonl_path: pathlib.Path) -> set[str]:
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_name = row.get("task_name")
            if isinstance(task_name, str):
                completed.add(task_name)
    return completed


def extract_text(response: Any) -> str:
    if isinstance(response, dict):
        for key in ("text", "output_text", "generated_text"):
            value = response.get(key)
            if isinstance(value, str):
                return value
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if isinstance(choice, dict):
                return str(choice.get("text") or choice.get("message", {}).get("content") or "")
    return str(response)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    # SGLang reads CUDA visibility during import/engine startup. Set this before
    # importing sglang or torch.
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from huggingface_hub import snapshot_download
    from sglang import Engine

    split_jsonls = args.split_jsonl
    if split_jsonls is None:
        split_jsonls = [
            DEFAULT_SPLIT_DIR / "train.jsonl",
            DEFAULT_SPLIT_DIR / "val.jsonl",
            DEFAULT_SPLIT_DIR / "test.jsonl",
        ]

    if split_jsonls:
        tasks = load_split_tasks(split_jsonls, args.out_dir)
    else:
        tasks = load_tasks(args.veribench_root)
        if len(tasks) != 896:
            raise RuntimeError(f"Expected 896 VeriBench Python/Lean pairs, found {len(tasks)}")
    tasks = tasks[args.start_index :]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "generations.jsonl"
    metadata_path = args.out_dir / "run_metadata.json"

    completed = set() if args.overwrite else read_completed(jsonl_path)
    pending = [task for task in tasks if task.task_name not in completed]
    example_py_code = read_text(args.gold_example_py)
    example_lean_code = read_text(args.gold_example_lean)

    if args.dry_run:
        first_prompt = build_prompt(tasks[0], example_py_code, example_lean_code) if tasks else ""
        print(f"Dry run OK: discovered {len(tasks)} selected promptable tasks")
        print(f"First task: {tasks[0].task_name if tasks else '<none>'}")
        print(f"First split: {tasks[0].split if tasks else '<none>'}")
        print(f"First prompt chars: {len(first_prompt)}")
        print("Per-task gold Lean contents are not loaded or interpolated into prompts.")
        return 0

    metadata = {
        "model": args.model,
        "model_snapshot": snapshot_download(args.model),
        "veribench_root": str(args.veribench_root),
        "split_jsonl": [str(path) for path in split_jsonls],
        "task_count_total": len(tasks),
        "task_count_selected": len(tasks),
        "task_count_pending": len(pending),
        "prompt_template": PROMPT_TEMPLATE,
        "gold_example_py": str(args.gold_example_py),
        "gold_example_lean": str(args.gold_example_lean),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        "raw_prompt": args.raw_prompt,
    }
    write_json(metadata_path, metadata)

    print(f"Loaded {len(tasks)} selected tasks from {args.veribench_root}")
    print(f"Skipping {len(completed)} completed tasks; pending {len(pending)}")
    print(f"Outputs: {args.out_dir}")
    if not pending:
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = Engine(
        model_path=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        dp_size=args.data_parallel_size,
        tp_size=args.tensor_parallel_size,
        load_balance_method="round_robin",
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k >= 0:
        sampling_params["top_k"] = args.top_k

    try:
        with jsonl_path.open("a", encoding="utf-8") as jsonl:
            for offset in tqdm(range(0, len(pending), args.batch_size), desc="Generating"):
                batch = pending[offset : offset + args.batch_size]
                prompts = [
                    maybe_apply_chat_template(
                        tokenizer,
                        build_prompt(task, example_py_code, example_lean_code),
                        args.raw_prompt,
                    )
                    for task in batch
                ]
                started = time.time()
                responses = engine.generate(prompt=prompts, sampling_params=sampling_params)
                elapsed = time.time() - started
                if isinstance(responses, dict):
                    responses = [responses]
                per_item_elapsed = elapsed / max(len(batch), 1)

                for task, prompt, response in zip(batch, prompts, responses):
                    text = extract_text(response)
                    lean_text = build_lean_output(task, text, args.no_strip_markdown)
                    out_path = output_path_for_task(args.out_dir, task.task_name)
                    raw_path = raw_output_path_for_task(args.out_dir, task.task_name)
                    prompt_path = prompt_path_for_task(args.out_dir, task.task_name)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(lean_text.rstrip() + "\n", encoding="utf-8")
                    raw_path.write_text(text, encoding="utf-8")
                    prompt_path.write_text(prompt, encoding="utf-8")

                    record = GenerationRecord(
                        task_name=task.task_name,
                        split=task.split,
                        task_id=task.task_id,
                        py_path=task.py_path,
                        output_lean_path=str(out_path),
                        raw_output_path=str(raw_path),
                        prompt_path=str(prompt_path),
                        model=args.model,
                        prompt_chars=len(prompt),
                        raw_generated_chars=len(text),
                        generated_chars=len(lean_text),
                        elapsed_sec=per_item_elapsed,
                        raw_output_text=text,
                        output_text=lean_text,
                    )
                    jsonl.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                    jsonl.flush()
    finally:
        engine.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
