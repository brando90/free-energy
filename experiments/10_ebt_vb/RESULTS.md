# Results

## Goedel-Prover-V2-8B evaluation (parallel compile, 32 workers)

- Run: `evaluate_veribench_outputs_from_safetensors.py`
- Run date: `2026-06-24`
- Workers: `32`
- Input manifest: `experiments/10_ebt_vb/data/context_gold/manifest.jsonl`
- Source outputs: `experiments/09_vb_testing_ipynb/results/goedel_prover_v2_8b_sglang_896_hidden_states_gpus0_5_full_884_bs16`
- Output directory: `experiments/10_ebt_vb/results/veribench_output_eval_workers32`

The input manifest currently contains **689** rows (train/val/test split counts are 550/65/74).

| Split | Rows | IC1 | IC2 | TE1 | D1 | D2 | S_tilde |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 550 | 0.120000 | 0.005455 | 0.000000 | 0.970909 | 0.926571 | 0.000000 |
| val | 65 | 0.076923 | 0.000000 | 0.000000 | 0.984615 | 0.936190 | 0.000000 |
| test | 74 | 0.094595 | 0.000000 | 0.000000 | 0.986486 | 0.921686 | 0.000000 |
| all | 689 | 0.113208 | 0.004354 | 0.000000 | 0.973875 | 0.926954 | 0.000000 |

## Output files

- `experiments/10_ebt_vb/results/veribench_output_eval_workers32/veribench_output_scores.csv`
- `experiments/10_ebt_vb/results/veribench_output_eval_workers32/veribench_output_aggregate.csv`
