# Results

Baseline smoke runs (20 tasks) are in the earlier sections of the run artifacts and are mostly failures after normalization changes. The full comparison report below is from the full 884-row VeriBench run using `Goedel-Prover-V2` and the shared one-shot VeriBench-style prompt.

## Smoke Runs (20 tasks)

| Model | Prompt | Max tokens | Passed | Total | Pass rate |
|---|---|---:|---:|---:|---:|
| `Qwen/Qwen3-8B` (`using_veribench_with_harbor.py`, Harbor prompt) | first harbor script | 512 | 0 | 20 | 0.00% |
| `Qwen/Qwen3-8B` (compile-first, Harbor task text) | compile-first | 4096 | 0 | 20 | 0.00% |
| `Qwen/Qwen3-8B` (compile-first, no Harbor) | compile-first | 2048 | 0 | 20 | 0.00% |
| `Goedel-LM/Goedel-Prover-V2-8B` (no Harbor) | compile-first | 1024 | 0 | 20 | 0.00% |
| `Goedel-LM/Goedel-Prover-V2-32B` (no Harbor) | compile-first | 1024 | 1 | 20 | 5.00% |

## Full run: `Goedel-Prover-V2-8B` on 884 rows (train/val/test)

- Run directory: `results/goedel_prover_v2_8b_sglang_tvt_prompt_v2/`
- Model: `Goedel-LM/Goedel-Prover-V2-8B`
- Prompt: standard VeriBench format, no per-task gold in prompt, no tensor parallelism (`tp=1`), data parallelism only (`dp=3`), sglang on 3 GPUs
- Tokens: `1024`
- Rows prompted: `884` (skipped 7 rows without `py_code`)

| Split | Passed | Total | Pass rate |
|---|---:|---:|---:|
| train | 66 | 702 | 9.40% |
| val | 6 | 86 | 6.98% |
| test | 9 | 96 | 9.38% |
| all | 81 | 884 | 9.16% |

## Pass rate by dataset division

| Division | Passed | Total | Pass rate |
|---|---:|---:|---:|
| `aerospace_set_real` | 0 | 3 | 0.00% |
| `aerospace_set_synthetic` | 1 | 17 | 5.88% |
| `compilers_set_real` | 0 | 5 | 0.00% |
| `compilers_set_synthetic` | 1 | 15 | 6.67% |
| `concurrent_set_real` | 0 | 5 | 0.00% |
| `concurrent_set_synthetic` | 0 | 15 | 0.00% |
| `critical_infra_set_real` | 0 | 6 | 0.00% |
| `critical_infra_set_synthetic` | 5 | 14 | 35.71% |
| `crypto_set_real` | 1 | 16 | 6.25% |
| `crypto_set_synthetic` | 0 | 4 | 0.00% |
| `cs_set` | 0 | 13 | 0.00% |
| `easy_set` | 13 | 41 | 31.71% |
| `finance_set_real` | 5 | 8 | 62.50% |
| `finance_set_synthetic` | 3 | 12 | 25.00% |
| `gov_cert_set_real` | 0 | 4 | 0.00% |
| `gov_cert_set_synthetic` | 3 | 16 | 18.75% |
| `hardware_set_real` | 1 | 5 | 20.00% |
| `hardware_set_synthetic` | 3 | 16 | 18.75% |
| `humaneval_set` | 3 | 56 | 5.36% |
| `hypervisor_set_real` | 1 | 5 | 20.00% |
| `hypervisor_set_synthetic` | 0 | 16 | 0.00% |
| `medical_oss_set_real` | 2 | 9 | 22.22% |
| `medical_oss_set_synthetic` | 2 | 11 | 18.18% |
| `memory_bugs_set_real` | 1 | 5 | 20.00% |
| `memory_bugs_set_synthetic` | 4 | 15 | 26.67% |
| `os_embedded_set_real` | 1 | 6 | 16.67% |
| `os_embedded_set_synthetic` | 4 | 14 | 28.57% |
| `realcode_set` | 1 | 32 | 3.12% |
| `regulated_set_real` | 2 | 5 | 40.00% |
| `regulated_set_synthetic` | 1 | 15 | 6.67% |
| `rfc_set_real` | 4 | 19 | 21.05% |
| `rfc_set_synthetic` | 1 | 1 | 100.00% |
| `security_set` | 18 | 460 | 3.91% |

## Artifacts for review

- Raw outputs: `results/goedel_prover_v2_8b_sglang_tvt_prompt_v2/raw_model_outputs/`
- Lean outputs: `results/goedel_prover_v2_8b_sglang_tvt_prompt_v2/lean_outputs/`
- Compile summary: `results/goedel_prover_v2_8b_sglang_tvt_prompt_v2/compile_summary.json`
- Prompt templates: `results/goedel_prover_v2_8b_sglang_tvt_prompt_v2/prompts/`
