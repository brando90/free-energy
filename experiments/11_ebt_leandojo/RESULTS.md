# Results

## LeanWorkbook Plus Baseline

You were right. I corrected the metric.

What changed:

- The LeanWorkbook Plus task is now:
  - input: `natural_language_statement` + theorem skeleton from `formal_statement`
- Success now means:
  - Lean accepts the completed theorem
  - no errors
  - no `sorry`

I kept the saved 500-sample subset.

Baseline result:

- `25 / 500` compile successfully
- `5.0%` compile-pass rate

Artifacts:

- Summary: [results/goedel_prover_v2_8b_sglang_leanworkbook_plus_val500_compile/summary.json](results/goedel_prover_v2_8b_sglang_leanworkbook_plus_val500_compile/summary.json)
- Per-sample outputs: [results/goedel_prover_v2_8b_sglang_leanworkbook_plus_val500_compile/generated.json](results/goedel_prover_v2_8b_sglang_leanworkbook_plus_val500_compile/generated.json)
