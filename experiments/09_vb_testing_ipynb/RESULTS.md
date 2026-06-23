# Results

This experiment runs VeriBench `smoke.jsonl` generation with
`using_veribench_with_harbor.py`, then compiles the extracted Lean files with
`compile_qwen_outputs.py`. Later runs use the compile-first prompt; the original
Qwen3-8B smoke run used the earlier Harbor-backed prompt path.

Common setup:

- Dataset: `veribench@1.1`
- Split: `experiments/08_vb_train_val_test/splits/smoke.jsonl`
- Tasks: `20`
- Temperature: `0.0`
- Compile environment: Lean `4.22.0` via the local VeriBench-DT Lean project
- Harbor/Docker verifier: not used; Docker daemon access is denied for this user

| Model / run | Max tokens | Prompt mode | Compile passed | Compile failed | Pass rate | Avg generation time | Avg generated chars |
|---|---:|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-8B` original smoke | 512 | Harbor-backed prompt | 0 | 20 | 0.00 | 15.603s/task | 1,657.5 |
| `Qwen/Qwen3-8B` compile-first | 4096 | compile-first + Harbor task text | 0 | 20 | 0.00 | 60.781s/task | 14,485.0 |
| `Qwen/Qwen3-8B` compile-first no-Harbor | 2048 | compile-first, no Harbor wrapper | 0 | 20 | 0.00 | 33.427s/task | 5,626.9 |
| `Goedel-LM/Goedel-Prover-V2-8B` | 1024 | compile-first, no Harbor wrapper | 0 | 20 | 0.00 | 20.579s/task | 1,728.7 |
| `Goedel-LM/Goedel-Prover-V2-32B` | 1024 | compile-first, no Harbor wrapper | 1 | 19 | 0.05 | 268.476s/task | 1,713.3 |

## Qwen3-8B

Three full 20-task Qwen3-8B smoke runs were saved.

### Original 512-token smoke

- Model: `Qwen/Qwen3-8B`
- Generation output: `results/qwen3_8b_smoke_20task.jsonl`
- Generation summary: `results/qwen3_8b_smoke_20task_summary.json`
- Compile summary: `results/qwen3_8b_smoke_20task_compile_summary.json`
- Extracted Lean files: `compiled_outputs/qwen3_8b_smoke_20task/`
- Compile pass result: `0/20`

### Compile-first, 4096-token smoke

- Model: `Qwen/Qwen3-8B`
- Generation output: `results/qwen3_8b_smoke_20task_compilefirst_4096.jsonl`
- Generation summary: `results/qwen3_8b_smoke_20task_compilefirst_4096_summary.json`
- Compile summary: `results/qwen3_8b_smoke_20task_compilefirst_4096_compile_summary.json`
- Extracted Lean files: `compiled_outputs/qwen3_8b_smoke_20task_compilefirst_4096/`
- Compile pass result: `0/20`

This run used the stricter compile-first preprompt, but still included Harbor's
original task instruction text, which conflicted with the `import Std` guidance
by asking for `import Mathlib`.

### Compile-first no-Harbor, 2048-token smoke

- Model: `Qwen/Qwen3-8B`
- Generation output: `results/qwen3_8b_smoke_20task_compilefirst_noharbor_2048.jsonl`
- Generation summary: `results/qwen3_8b_smoke_20task_compilefirst_noharbor_2048_summary.json`
- Compile summary: `results/qwen3_8b_smoke_20task_compilefirst_noharbor_2048_compile_summary.json`
- Extracted Lean files: `compiled_outputs/qwen3_8b_smoke_20task_compilefirst_noharbor_2048/`
- Compile pass result: `0/20`

Observed Qwen failure pattern:

- The simple core implementations were often directionally faithful, especially
  for Boolean gates and conditional stores.
- Files failed at the Lean formalization layer: Lean 3 proof syntax, invalid
  `do` notation, `Bool`/`Prop` confusion, unsafe list indexing, and undeclared
  constants.
- Larger token budgets reduced truncation but did not produce compileable Lean
  on this smoke set.

## Goedel-Prover-V2-8B

- Model: `Goedel-LM/Goedel-Prover-V2-8B`
- Generation output: `results/goedel_prover_v2_8b_smoke_20task_compilefirst_noharbor_1024.jsonl`
- Generation summary: `results/goedel_prover_v2_8b_smoke_20task_compilefirst_noharbor_1024_summary.json`
- Compile summary: `results/goedel_prover_v2_8b_smoke_20task_compilefirst_noharbor_1024_compile_summary.json`
- Extracted Lean files: `compiled_outputs/goedel_prover_v2_8b_smoke_20task_compilefirst_noharbor_1024/`
- Compile pass result: `0/20`

Observed failure pattern:

- Many outputs began with an extra non-Lean token before `import`, causing immediate parser failures.
- Several files mixed `Bool` and `Prop` incorrectly.
- Some outputs repeated declarations until truncation.
- Several proofs used unavailable or invalid tactics.

## Goedel-Prover-V2-32B

- Model: `Goedel-LM/Goedel-Prover-V2-32B`
- Generation output: `results/goedel_prover_v2_32b_smoke_20task_compilefirst_noharbor_1024.jsonl`
- Generation summary: `results/goedel_prover_v2_32b_smoke_20task_compilefirst_noharbor_1024_summary.json`
- Compile summary: `results/goedel_prover_v2_32b_smoke_20task_compilefirst_noharbor_1024_compile_summary.json`
- Extracted Lean files: `compiled_outputs/goedel_prover_v2_32b_smoke_20task_compilefirst_noharbor_1024/`
- Compile pass result: `1/20`
- Passing task: `aerospace_set_synthetic/MyFcsGainSchedule`

Observed failure pattern:

- The 32B model produced more Lean-like structure than the 8B run, but most files still failed to compile.
- Common failures were invalid `do` notation, unavailable tactics, `Bool`/`Prop` confusion, and unsolved goals.
- The 32B run was much slower than the 8B run on the same 20-task smoke set.

## Notes

No Mistral-32B run is reported here. I could not identify a real public official
`mistralai/*32B*` checkpoint; the literal public search hit
`tomaszki/mistral-32-b` is not a 32B model by config.
