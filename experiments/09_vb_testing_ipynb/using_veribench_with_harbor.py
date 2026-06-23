#!/usr/bin/env python3
"""Run a local Qwen baseline on VeriBench tasks exposed through Harbor.

This script does three things:

1. Verifies the public Harbor registry entry for ``veribench@1.1``.
2. Optionally downloads the Harbor task bundle with the Harbor CLI.
3. Optionally pulls Qwen3-8B from Hugging Face and generates Lean outputs for a
   bounded set of VeriBench tasks.

Harbor task verification itself requires a running Docker daemon. The generation
path does not.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


REGISTRY_URL = "https://raw.githubusercontent.com/brando90/harbor-datasets/main/harbor_registry.json"
DATASET_NAME = "veribench"
DATASET_VERSION = "1.1"
DATASET = f"{DATASET_NAME}@{DATASET_VERSION}"
EXPECTED_TASKS = 884
MODEL_ID = "Qwen/Qwen3-8B"

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_HARBOR_DIR = HERE / "harbor_download"
DEFAULT_RESULTS_DIR = HERE / "results"
DEFAULT_SPLIT = REPO_ROOT / "experiments" / "08_vb_train_val_test" / "splits" / "smoke.jsonl"

LEAN4_COMPILE_FIRST_PREPROMPT = """\
You are generating Lean 4.22.0 code for VeriBench.

Hard requirements:
- Return only Lean code. Do not use markdown fences.
- Prefer `import Std` only. Do not import Mathlib unless it is absolutely
  necessary. Never use Lean 3 imports such as `import data.bool`, `import tactic`,
  or capitalized pseudo-imports such as `import Data.Bool`.
- Use Lean 4 names: `Bool`, `true`, `false`, `Nat`, `Int`, `List`, `String`.
  Do not use Lean 3 names such as `bool`, `tt`, or `ff`.
- Keep the implementation simple and total. Model Python integers as `Nat` when
  the precondition requires nonnegative values; otherwise use `Int`.
- Include examples with `by native_decide` where possible.
- It is acceptable to use `sorry` for theorem proofs if that helps the file
  compile. The primary score is Lean compilation.
- Produce a complete file with balanced namespace/end commands.

Required file shape:
1. `import Std`
2. `set_option linter.unusedVariables false`
3. namespace named after the task in CamelCase
4. functional implementation
5. unit examples
6. `def Pre`
7. property propositions and theorem statements
8. `def Post_prop`
9. correctness theorem
10. imperative implementation using `Id.run do`
11. equivalence theorem
12. `end <namespace>`
"""


@dataclass
class GenerationRecord:
    task_name: str
    task_id: str | None
    source_kind: str | None
    prompt_chars: int
    generated_chars: int
    elapsed_sec: float
    output_text: str


def fetch_registry(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def find_dataset(registry: list[dict[str, Any]], name: str, version: str) -> dict[str, Any]:
    matches = [
        item
        for item in registry
        if item.get("name") == name and str(item.get("version")) == version
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one {name}@{version}; found {len(matches)}")
    return matches[0]


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: list[str], cwd: pathlib.Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def verify_registry() -> dict[str, Any]:
    registry = fetch_registry(REGISTRY_URL)
    release = find_dataset(registry, DATASET_NAME, DATASET_VERSION)
    tasks = release.get("tasks", [])

    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print(f"OK: found {DATASET} with {len(tasks)} tasks")
    print(f"Description: {release.get('description', '')}")
    print(f"Metrics: {release.get('metrics', [])}")
    if len(tasks) != EXPECTED_TASKS:
        raise AssertionError(f"Expected {EXPECTED_TASKS} tasks; found {len(tasks)}")
    return release


def download_harbor_dataset(output_dir: pathlib.Path) -> None:
    harbor = shutil.which("harbor")
    if not harbor:
        raise RuntimeError(
            "Harbor CLI not found on PATH. Install it in Python 3.12, for example: "
            "uv venv --python 3.12 .venv-harbor && "
            "UV_PROJECT_ENVIRONMENT=.venv-harbor uv pip install --python .venv-harbor/bin/python harbor"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_cmd(
        [
            harbor,
            "download",
            DATASET,
            "--registry-url",
            REGISTRY_URL,
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def harbor_docker_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "docker_cli": command_exists("docker"),
        "harbor_cli": command_exists("harbor"),
        "docker_daemon": False,
    }
    if status["docker_cli"]:
        result = run_cmd(["docker", "info"])
        status["docker_daemon"] = result.returncode == 0
        if result.returncode != 0:
            status["docker_error"] = (result.stderr or result.stdout).strip().splitlines()[-1:]
    return status


def load_jsonl(path: pathlib.Path, max_tasks: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_tasks is not None and len(rows) >= max_tasks:
                break
    return rows


def harbor_instruction_path(harbor_root: pathlib.Path, task_name: str) -> pathlib.Path | None:
    candidate = harbor_root / DATASET_NAME / task_name.replace("/", "__") / "instruction.md"
    if candidate.exists():
        return candidate
    return None


def build_prompt(
    task: dict[str, Any],
    harbor_root: pathlib.Path | None,
    use_harbor_instruction: bool,
    compile_first: bool,
) -> str:
    task_text: str
    if use_harbor_instruction and harbor_root is not None:
        instruction = harbor_instruction_path(harbor_root, task["task_name"])
        if instruction is not None:
            task_text = instruction.read_text(encoding="utf-8")
        else:
            task_text = ""
    else:
        task_text = ""

    if not task_text:
        task_text = f"""# Task: Translate Python to Lean 4

Translate the following Python program into a Lean 4 formalization.

## Python Source Code

```python
{task["py_code"]}
```

## Output Requirements

Return only the Lean 4 code. Include implementation, examples, pre/post
conditions, correctness theorem, imperative implementation, and equivalence
theorem in the standard VeriBench format.
"""

    if not compile_first:
        return task_text

    return f"""{LEAN4_COMPILE_FIRST_PREPROMPT}

## VeriBench task

{task_text}
"""


def pull_model(model_id: str) -> pathlib.Path:
    from huggingface_hub import snapshot_download

    print(f"Pulling {model_id} from Hugging Face...", flush=True)
    path = snapshot_download(repo_id=model_id)
    print(f"Model snapshot: {path}")
    return pathlib.Path(path)


def load_model(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_one(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Lean 4.22 expert. Return only complete Lean code, "
                "with no markdown fences. Optimize for compilation first."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    do_sample = temperature > 0

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_jsonl(args.split, args.max_tasks)
    if not tasks:
        raise RuntimeError(f"No tasks loaded from {args.split}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.results_dir / f"{args.output_prefix}.jsonl"
    out_summary = args.results_dir / f"{args.output_prefix}_summary.json"

    if args.pull_model:
        pull_model(args.model_id)

    tokenizer, model = load_model(args.model_id)
    records: list[GenerationRecord] = []

    with out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, task in enumerate(tasks, start=1):
            prompt = build_prompt(
                task=task,
                harbor_root=args.harbor_dir if args.harbor_dir.exists() else None,
                use_harbor_instruction=args.use_harbor_instruction,
                compile_first=args.compile_first_prompt,
            )
            print(f"[{idx}/{len(tasks)}] generating {task['task_name']}", flush=True)
            start = time.monotonic()
            output = generate_one(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            elapsed = time.monotonic() - start
            record = GenerationRecord(
                task_name=task["task_name"],
                task_id=task.get("task_id"),
                source_kind=task.get("source_kind"),
                prompt_chars=len(prompt),
                generated_chars=len(output),
                elapsed_sec=round(elapsed, 3),
                output_text=output,
            )
            records.append(record)
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()

    summary = {
        "dataset": DATASET,
        "model_id": args.model_id,
        "split": str(args.split),
        "num_tasks": len(records),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "results_jsonl": str(out_jsonl),
        "avg_elapsed_sec": round(sum(r.elapsed_sec for r in records) / len(records), 3),
        "avg_generated_chars": round(sum(r.generated_chars for r in records) / len(records), 1),
        "harbor_status": harbor_docker_status(),
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--split", type=pathlib.Path, default=DEFAULT_SPLIT)
    parser.add_argument("--harbor-dir", type=pathlib.Path, default=DEFAULT_HARBOR_DIR)
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-prefix", default="qwen3_8b_smoke")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-harbor-instruction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-first-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-harbor", action="store_true")
    parser.add_argument("--pull-model", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    verify_registry()
    print(f"Docker CLI: {command_exists('docker')}")
    print(f"Harbor CLI: {command_exists('harbor')}")

    if args.download_harbor:
        download_harbor_dataset(args.harbor_dir)

    if args.run_baseline:
        run_baseline(args)


if __name__ == "__main__":
    main()
