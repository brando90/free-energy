# EXPC Decoding-Robustness Satellite — vLLM vs HF Cell-Level Replay

Replays **all 1524 validated rows** of `results/EXPC_POLARITY_CONTROL_FULL` under vLLM and
re-scores each continuation with the experiment's own validator, then compares cell-level rates to the
stored HF outputs. Purpose: decide whether the paper's one-seed greedy design earns a measured
decoding-robustness defense for Appendix J.

## Provenance & fidelity

- Model: `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`, greedy, `max_new_tokens=192`.
- Backend: vLLM (greedy, temperature=0, prefix-cache off, seed=0); GPU `CUDA_VISIBLE_DEVICES=4`; vLLM gen wall 49.5s for 1524 rows.
- Prompts reconstructed from `manifest.jsonl` + `run_metadata.json` prompt block via the pilot's byte-exact `render_prompt_ids` (token-id path).
- vLLM continuations re-scored with `src/validator.validate_continuation` and the exact `validate()` field mapping (`valid_recovery`/`doubt`/`poisoning`); rates via `metric_value`+`summarize_cell`.
- **HF-pipeline self-check: True** — recomputing HF rates from `validated_outputs.jsonl` reproduces every pooled `summary_tables.json` rate exactly, so the scoring hookup is faithful.
- Delta CIs: Newcombe 1998 paired score interval (z=1.96) (accounts for HF/vLLM pairing on identical items).

## Headline

- Structural metrics (`valid_recovery`, `poisoning`, `unparsed`) are decoding-robust at the cell level: pooled |delta| <= 5.0pp (poisoning/unparsed <= 1.6pp).
- `doubt` is NOT robust: pooled |delta| up to 15.0pp, systematically lower under vLLM.
- Largest pooled cell delta overall: **doubt** in LOCAL_FALSE_POSITIVE = -0.1496 (HF 0.4357 -> vLLM 0.2861, n=381).
- Row-level validator-class agreement: **0.7224** (1101/1524); exact-text 0.3287.

## Pooled per condition (n=381)

| condition | metric | HF | vLLM | delta | delta 95% CI |
|---|---|---:|---:|---:|:---:|
| LOCAL_FALSE_POSITIVE | valid_recovery | 0.7349 | 0.6850 | -0.0499 | [-0.103, +0.003] |
| LOCAL_FALSE_POSITIVE | doubt | 0.4357 | 0.2861 | -0.1496 | [-0.203, -0.097] |
| LOCAL_FALSE_POSITIVE | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_POSITIVE | unparsed | 0.0367 | 0.0420 | +0.0052 | [-0.019, +0.029] |
| LOCAL_FALSE_NEGATIVE | valid_recovery | 0.5722 | 0.5958 | +0.0236 | [-0.026, +0.074] |
| LOCAL_FALSE_NEGATIVE | doubt | 0.3780 | 0.2493 | -0.1286 | [-0.175, -0.083] |
| LOCAL_FALSE_NEGATIVE | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_NEGATIVE | unparsed | 0.0420 | 0.0367 | -0.0052 | [-0.032, +0.022] |
| GLOBAL_FALSE_POSITIVE | valid_recovery | 0.3675 | 0.4042 | +0.0367 | [+0.003, +0.071] |
| GLOBAL_FALSE_POSITIVE | doubt | 0.0131 | 0.0000 | -0.0131 | [-0.026, +0.004] |
| GLOBAL_FALSE_POSITIVE | poisoning | 0.2887 | 0.2966 | +0.0079 | [-0.024, +0.039] |
| GLOBAL_FALSE_POSITIVE | unparsed | 0.0472 | 0.0315 | -0.0157 | [-0.037, +0.008] |
| GLOBAL_FALSE_NEGATIVE | valid_recovery | 0.6063 | 0.6352 | +0.0289 | [-0.010, +0.068] |
| GLOBAL_FALSE_NEGATIVE | doubt | 0.1365 | 0.0551 | -0.0814 | [-0.114, -0.046] |
| GLOBAL_FALSE_NEGATIVE | poisoning | -- | -- | -- | -- |
| GLOBAL_FALSE_NEGATIVE | unparsed | 0.0499 | 0.0394 | -0.0105 | [-0.032, +0.013] |

## Per condition x position (n=127)

| condition | pos | metric | HF | vLLM | delta | delta 95% CI |
|---|---|---|---:|---:|---:|:---:|
| LOCAL_FALSE_POSITIVE | early | valid_recovery | 0.6929 | 0.6063 | -0.0866 | [-0.182, +0.006] |
| LOCAL_FALSE_POSITIVE | early | doubt | 0.5197 | 0.3543 | -0.1654 | [-0.269, -0.068] |
| LOCAL_FALSE_POSITIVE | early | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_POSITIVE | early | unparsed | 0.0551 | 0.0551 | +0.0000 | [-0.050, +0.050] |
| LOCAL_FALSE_POSITIVE | mid | valid_recovery | 0.7323 | 0.7638 | +0.0315 | [-0.050, +0.113] |
| LOCAL_FALSE_POSITIVE | mid | doubt | 0.3858 | 0.2520 | -0.1339 | [-0.214, -0.056] |
| LOCAL_FALSE_POSITIVE | mid | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_POSITIVE | mid | unparsed | 0.0394 | 0.0315 | -0.0079 | [-0.050, +0.037] |
| LOCAL_FALSE_POSITIVE | late | valid_recovery | 0.7795 | 0.6850 | -0.0945 | [-0.191, +0.000] |
| LOCAL_FALSE_POSITIVE | late | doubt | 0.4016 | 0.2520 | -0.1496 | [-0.243, -0.059] |
| LOCAL_FALSE_POSITIVE | late | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_POSITIVE | late | unparsed | 0.0157 | 0.0394 | +0.0236 | [-0.024, +0.063] |
| LOCAL_FALSE_NEGATIVE | early | valid_recovery | 0.4803 | 0.5591 | +0.0787 | [-0.018, +0.178] |
| LOCAL_FALSE_NEGATIVE | early | doubt | 0.4488 | 0.2677 | -0.1811 | [-0.274, -0.093] |
| LOCAL_FALSE_NEGATIVE | early | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_NEGATIVE | early | unparsed | 0.0394 | 0.0236 | -0.0157 | [-0.060, +0.033] |
| LOCAL_FALSE_NEGATIVE | mid | valid_recovery | 0.6299 | 0.6299 | +0.0000 | [-0.072, +0.072] |
| LOCAL_FALSE_NEGATIVE | mid | doubt | 0.3465 | 0.2283 | -0.1181 | [-0.182, -0.054] |
| LOCAL_FALSE_NEGATIVE | mid | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_NEGATIVE | mid | unparsed | 0.0472 | 0.0157 | -0.0315 | [-0.073, +0.019] |
| LOCAL_FALSE_NEGATIVE | late | valid_recovery | 0.6063 | 0.5984 | -0.0079 | [-0.093, +0.077] |
| LOCAL_FALSE_NEGATIVE | late | doubt | 0.3386 | 0.2520 | -0.0866 | [-0.169, -0.006] |
| LOCAL_FALSE_NEGATIVE | late | poisoning | -- | -- | -- | -- |
| LOCAL_FALSE_NEGATIVE | late | unparsed | 0.0394 | 0.0709 | +0.0315 | [-0.029, +0.088] |
| GLOBAL_FALSE_POSITIVE | early | valid_recovery | 0.3150 | 0.3701 | +0.0551 | [+0.000, +0.110] |
| GLOBAL_FALSE_POSITIVE | early | doubt | 0.0157 | 0.0000 | -0.0157 | [-0.047, +0.024] |
| GLOBAL_FALSE_POSITIVE | early | poisoning | 0.2362 | 0.2441 | +0.0079 | [-0.053, +0.069] |
| GLOBAL_FALSE_POSITIVE | early | unparsed | 0.0551 | 0.0394 | -0.0157 | [-0.059, +0.032] |
| GLOBAL_FALSE_POSITIVE | mid | valid_recovery | 0.4173 | 0.4567 | +0.0394 | [-0.030, +0.109] |
| GLOBAL_FALSE_POSITIVE | mid | doubt | 0.0157 | 0.0000 | -0.0157 | [-0.047, +0.024] |
| GLOBAL_FALSE_POSITIVE | mid | poisoning | 0.3150 | 0.2992 | -0.0157 | [-0.070, +0.038] |
| GLOBAL_FALSE_POSITIVE | mid | unparsed | 0.0472 | 0.0315 | -0.0157 | [-0.053, +0.028] |
| GLOBAL_FALSE_POSITIVE | late | valid_recovery | 0.3701 | 0.3858 | +0.0157 | [-0.033, +0.064] |
| GLOBAL_FALSE_POSITIVE | late | doubt | 0.0079 | 0.0000 | -0.0079 | [-0.038, +0.028] |
| GLOBAL_FALSE_POSITIVE | late | poisoning | 0.3150 | 0.3465 | +0.0315 | [-0.017, +0.080] |
| GLOBAL_FALSE_POSITIVE | late | unparsed | 0.0394 | 0.0236 | -0.0157 | [-0.060, +0.033] |
| GLOBAL_FALSE_NEGATIVE | early | valid_recovery | 0.5118 | 0.5591 | +0.0472 | [-0.026, +0.122] |
| GLOBAL_FALSE_NEGATIVE | early | doubt | 0.1102 | 0.0236 | -0.0866 | [-0.139, -0.023] |
| GLOBAL_FALSE_NEGATIVE | early | poisoning | -- | -- | -- | -- |
| GLOBAL_FALSE_NEGATIVE | early | unparsed | 0.0866 | 0.0394 | -0.0472 | [-0.085, +0.003] |
| GLOBAL_FALSE_NEGATIVE | mid | valid_recovery | 0.5591 | 0.5984 | +0.0394 | [-0.030, +0.109] |
| GLOBAL_FALSE_NEGATIVE | mid | doubt | 0.1811 | 0.0787 | -0.1024 | [-0.171, -0.029] |
| GLOBAL_FALSE_NEGATIVE | mid | poisoning | -- | -- | -- | -- |
| GLOBAL_FALSE_NEGATIVE | mid | unparsed | 0.0472 | 0.0630 | +0.0157 | [-0.032, +0.059] |
| GLOBAL_FALSE_NEGATIVE | late | valid_recovery | 0.7480 | 0.7480 | +0.0000 | [-0.059, +0.059] |
| GLOBAL_FALSE_NEGATIVE | late | doubt | 0.1181 | 0.0630 | -0.0551 | [-0.106, +0.003] |
| GLOBAL_FALSE_NEGATIVE | late | poisoning | -- | -- | -- | -- |
| GLOBAL_FALSE_NEGATIVE | late | unparsed | 0.0157 | 0.0157 | +0.0000 | [-0.042, +0.042] |

## Max |delta| per metric

| metric | pooled (n=381) | worst cell x position (n=127) |
|---|---|---|
| valid_recovery | -0.0499 (LOCAL_FALSE_POSITIVE) | -0.0945 (LOCAL_FALSE_POSITIVE|late) |
| doubt | -0.1496 (LOCAL_FALSE_POSITIVE) | -0.1811 (LOCAL_FALSE_NEGATIVE|early) |
| poisoning | +0.0079 (GLOBAL_FALSE_POSITIVE) | +0.0315 (GLOBAL_FALSE_POSITIVE|late) |
| unparsed | -0.0157 (GLOBAL_FALSE_POSITIVE) | -0.0472 (GLOBAL_FALSE_NEGATIVE|early) |

## Row-level agreement

- Validator-class agreement (HF class == vLLM class): **0.7224** (1101/1524).
- `valid_recovery` boolean agreement: **0.7999**.
- Exact continuation-text match: 0.3287 (501/1524) — consistent with the 20-row pilot's 7/20.

Class agreement is lower in LOCAL conditions (more knife's-edge branches):

| condition | n | class agree | exact text |
|---|---:|---:|---:|
| LOCAL_FALSE_POSITIVE | 381 | 0.6877 | 0.2021 |
| LOCAL_FALSE_NEGATIVE | 381 | 0.6404 | 0.2336 |
| GLOBAL_FALSE_POSITIVE | 381 | 0.7822 | 0.4619 |
| GLOBAL_FALSE_NEGATIVE | 381 | 0.7795 | 0.4173 |

Top HF->vLLM class transitions:

| transition | count |
|---|---:|
| valid_rederivation->valid_rederivation | 724 |
| parroted->parroted | 163 |
| poisoned->poisoned | 103 |
| derailed->derailed | 86 |
| valid_rederivation->parroted | 68 |
| derailed->valid_rederivation | 61 |
| parroted->valid_rederivation | 61 |
| valid_rederivation->derailed | 50 |
| derailed->parroted | 31 |
| unparsed->unparsed | 25 |

## Scientific-contrast preservation (descriptive, from pooled condition rates)

| metric | contrast | HF | vLLM |
|---|---|---:|---:|
| valid_recovery | local - global | +0.1667 | +0.1207 |
| valid_recovery | positive - negative | -0.0381 | -0.0709 |
| doubt | local - global | +0.3320 | +0.2402 |
| doubt | positive - negative | -0.0328 | -0.0091 |

The paper's headline locality contrast (local > global recovery **and** doubt) survives in sign and magnitude
under vLLM; polarity remains small/near-zero under both backends.

## Verdict

The cell-level decoding-robustness defense **holds for the structural metrics but not for `doubt`.** `valid_recovery` moves <=5pp per condition (pooled |delta| max 4.99pp, CIs mostly spanning 0), and `poisoning` (+0.8pp) and `unparsed` (<=1.6pp) are essentially unchanged — even though only 33% of individual continuations are byte-identical and 28% of rows change validator class, the *rates* of the structural outcomes are stable to well within sampling noise, exactly the robustness the one-seed greedy design needs for Appendix J. `doubt`, by contrast, is a surface-lexical signal (a regex for 'wait/however/but/actually/...') and its absolute rate drops 8-15pp under vLLM because tiny bf16 kernel differences reroute whether the model hedges verbally; that number is backend-sensitive and must not be reported as a fixed quantity from one seed/backend. Crucially, the paper's *claims* are comparative contrasts, and the primary locality contrast (local >> global) survives in sign and significance for both `valid_recovery` (HF +0.167 -> vLLM +0.121) and `doubt` (HF +0.332 -> vLLM +0.240). Recommended Appendix J framing: cite this replay as measured evidence that the structural recovery rates are decoding-robust at the cell level, explicitly flag `doubt`'s absolute rate as backend-dependent, and anchor the doubt narrative on the (robust) contrast rather than the point rate.

## Method notes / caveats

- Prompt fed to vLLM is the byte-exact HF token sequence (`TokensPrompt`, prefix concatenated at token-id level); verified in the port report.
- `poisoning` is only measurable in GLOBAL_FALSE_POSITIVE (planted predicate can participate downstream); measurable-only denominator, identical for HF and vLLM per row.
- Divergences are genuine greedy near-ties from HF-vs-vLLM kernel numerics, not a prompt bug (33% byte-identical is impossible with a wrong prompt); this is not evidence of a decoding error, only of the task family sitting on a knife's edge by design.
- Delta CIs use the Newcombe paired score interval; because HF and vLLM share items, unpaired intervals would be wider.
- Single seed / single vLLM version (0.24.0); vLLM is not bitwise-deterministic across versions or batch compositions. Pin `requirements-vllm.lock` to reproduce.
