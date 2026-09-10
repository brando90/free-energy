# Results Lock

Generated: 2026-06-23T19:24:59.100746+00:00
Git commit: `8894a05c4703b8c58745bb8c0f8acba33a54dfee`

## Regeneration Commands

These commands were run during the lock update; paper tables, macros, and figures are regenerated from artifacts rather than edited by hand.

- Paper tables/macros: `/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python src/paper_artifacts.py`
- Figures: `/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python src/make_workshop_figures.py`
- Paper PDF: run `pdflatex main.tex` twice in `paper_latex/papers/latent_recovery`.

## Paper Table/Figure Sources

- Generated LaTeX tables/macros: `../../paper_latex/papers/latent_recovery/paper_results.tex`
- Generated metadata: `../../paper_latex/papers/latent_recovery/paper_results_metadata.json`
- Figures are generated from JSON summaries by `src/make_workshop_figures.py`.
- Regenerated figure outputs: `fig_dose_response.{tex,pdf,png}`, `fig_regime_map.{tex,pdf,png}`, `fig_verbalization_spectrum.{tex,pdf,png}`.

### Source Hashes

- `data/pilot.jsonl`: `f12654d4b49f9e6e4d2f248589d3119bc625640d4b83bb960a25c243fcba0de9`
- `results/EXPA_GLOBAL_EXPANSION/EXPA_REPORT.md`: `ef15ed4a41281c4ac21194bafd00ad0597b7239b22134d6636f1e47233db0eda`
- `results/EXPA_GLOBAL_EXPANSION/eligibility_audit.jsonl`: `99e12707b35db3b9aba6ee64dc4289d55f789ff84a36b1afd89a36446dbff16d`
- `results/EXPA_GLOBAL_EXPANSION/manifest.jsonl`: `1aa70aec1d61d10b1eb14adb0492b5e22616cfa1c93b6400b40ef2b48f9ba97b`
- `results/EXPA_GLOBAL_EXPANSION/raw_generations.jsonl`: `37e5bf045064fb75f5dcdc43b9fef21c92af4342d37f9cd44fc141afdd456aa9`
- `results/EXPA_GLOBAL_EXPANSION/run_metadata.json`: `c851751ebc0628bc8a0eac65ac79b597417943b17d74a5c095fdc5498e1bf4e7`
- `results/EXPA_GLOBAL_EXPANSION/summary_tables.json`: `92367fa813a911283b53ccf33156b072a63cff23798df0f3d936041f31e46ff9`
- `results/EXPA_GLOBAL_EXPANSION/validated_outputs.jsonl`: `83e4a88f1c32be9de9e87fcedff5f59ac76aded42e19e4f1f77abfb8ee186a15`
- `results/EXPB_LOCAL_CERT_FLIP/EXPB_REPORT.md`: `f9c14f44e1b6e28a5b75e8a084ff5d5cc466dc1e1dcba3b5b5421c43920744b0`
- `results/EXPB_LOCAL_CERT_FLIP/manifest.jsonl`: `d960a8d2bc4d2540063ad625759f2fde6509f80b5786ac185ee14a323bee86ea`
- `results/EXPB_LOCAL_CERT_FLIP/raw_generations.jsonl`: `dec633ea965f44449a7b32b9f1ab1481f43e9fd51456220114dd07ed9ecbf20e`
- `results/EXPB_LOCAL_CERT_FLIP/run_metadata.json`: `461d77222383e1072cf70d7ca4384b7aa8eeebcd10324f03df61ba5c96a597df`
- `results/EXPB_LOCAL_CERT_FLIP/summary_tables.json`: `dd8dff8ab31c08436c7388625b097a196f22d05cd343f0c233f6174de1e63ba6`
- `results/EXPB_LOCAL_CERT_FLIP/validated_outputs.jsonl`: `94a33e23019bff91346b983a4f3d7e3c722524d091472ea0021cef2dc47b34c3`
- `results/EXPC_POLARITY_CONTROL/EXPC_REPORT.md`: `af969c6d93ed9176d1d3190bde3d9e790ebf2d5e7412d12f1fabcb18a41164da`
- `results/EXPC_POLARITY_CONTROL/manifest.jsonl`: `393d7350a6b295ee39f1a3f247f6107290c4d47f2479987dcb5693868aa06b72`
- `results/EXPC_POLARITY_CONTROL/raw_generations.jsonl`: `a1afc2a1998e5a05cb90dcf9443282de3b5bf9e7fe2c31e67eb265cb0b1ef053`
- `results/EXPC_POLARITY_CONTROL/run_metadata.json`: `0f0d662e8268b5e654f0394c5a6eabb1c6b2121d3291bac61f824711d51bcd88`
- `results/EXPC_POLARITY_CONTROL/summary_tables.json`: `ac8759d1d042b2b68c4b250b039ef484c499243b3f0f9c8906283f8521fe0d2d`
- `results/EXPC_POLARITY_CONTROL/validated_outputs.jsonl`: `6d63b4cc3aa8c94e740b7b34456178c3f38f1bd62e6bafdb28ac3e8b8b61e8df`
- `results/arith/summary.json`: `eef0e07eb69981c82d92ff4a215e2b3e5d223f23612ba2dc05292b561cf3977a`
- `results/gsm8k/summary.json`: `5b94173e4dcfafcb146499ca3a15188b8a1e79a758e6f77bcee9020364c941a4`
- `results/validated.jsonl`: `0fd58ce65e42ed5c3471f3ac0ac5a63c9efd00e71bdf8b38227acaa9125931d1`
- `results/validated_summary.json`: `b86a1cfa6322dabdc7209048d71436840b7132b7bd46f622392beb35cbf10fb4`
- `results/validated_summary_contradiction.json`: `6644f6ef80a381d159cf1dc89986e5b08576bff704f3a1dad5b44f343d33da0f`
- `results/validated_summary_distractor.json`: `b830ebf3a07ea0582f5a5acac820ed6532ce619efa91e356b8dd4e4d4ab96747`
- `results/validated_summary_falsehood.json`: `91ce97a198773f5149a38ec5af4115e657f846d082d04e3f9f12b4e38dac4989`
- `results/validated_summary_negstep.json`: `a86abe00bfae56e8d0f7f54bc41836b60e4bddbedfaa9154281467e527799609`
- `results/validated_summary_paraphrase.json`: `0011545920aa8304f7b112db62d120d5ef21de51d208cc411785c1e2af9c748e`
- `results_1p5b/validated_summary_falsehood.json`: `66865f181c86f327d789e094e8f21b2411c36a7f60ab839d2339b3e130950c08`
- `results_1p5b/validated_summary_negstep.json`: `9dc4779fa78acd15d723ac83b8d36f6e388103d14eaef8a82dd789e530dd1b06`
- `results_32b/validated_summary_falsehood.json`: `2dbc6d793a7d0adcc80f02ec871b97e65e476266422e514b9a7faa1a89b9c7b9`
- `results_32b/validated_summary_negstep.json`: `1f1031b1e2416c12f743ee31cc0ead4446de7e1adc05dd04fec9b5f45f7a9437`
- `results_verify/validated_summary_falsehood_verify.json`: `1658e4f67bfea1a0668e270f7d1d7651b7f569b9f6e3392772bcaf8b8916265a`
- `results_verify/validated_summary_negstep_verify.json`: `5da5a8e38e8f8921715cabe55344ea3878052e774aa5b6c3396ce9a41f2a4e45`

## Full-Schema Result Directory Audit

- `results/EXPB_LOCAL_CERT_FLIP`: **PASS** (manifest=yes, raw_generations=yes, validator_outputs=yes, summary_tables=yes, readme_or_report=yes; run_complete=yes)
  - row counts: `{"manifest.jsonl": 900, "raw_generations.jsonl": 900, "summary_validated_rows": 900, "validated_outputs.jsonl": 900}`
- `results/EXPC_POLARITY_CONTROL`: **PASS** (manifest=yes, raw_generations=yes, validator_outputs=yes, summary_tables=yes, readme_or_report=yes; run_complete=yes)
  - row counts: `{"eligibility_audit.jsonl": 23539, "manifest.jsonl": 84, "raw_generations.jsonl": 91, "summary_validated_rows": 84, "validated_outputs.jsonl": 84}`
- `results/EXPA_GLOBAL_EXPANSION`: **PASS** (manifest=yes, raw_generations=yes, validator_outputs=yes, summary_tables=yes, readme_or_report=yes; run_complete=yes)
  - row counts: `{"eligibility_audit.jsonl": 22132, "manifest.jsonl": 1800, "raw_generations.jsonl": 4832, "summary_validated_rows": 1800, "validated_outputs.jsonl": 1800}`
- `results/EXPA_GLOBAL_EXPANSION_SMOKE`: **PASS** (manifest=yes, raw_generations=yes, validator_outputs=yes, summary_tables=yes, readme_or_report=yes; run_complete=yes)
  - row counts: `{"eligibility_audit.jsonl": 19116, "manifest.jsonl": 24, "raw_generations.jsonl": 40, "summary_validated_rows": 24, "validated_outputs.jsonl": 24}`

## Result Directory Immutability

- `results/EXPB_LOCAL_CERT_FLIP`: **WRITABLE** (11 writable entries)
- `results/EXPC_POLARITY_CONTROL`: **WRITABLE** (14 writable entries)
- `results/EXPA_GLOBAL_EXPANSION`: **WRITABLE** (13 writable entries)
- `results/EXPA_GLOBAL_EXPANSION_SMOKE`: **READ-ONLY**

Legacy/minimal result directories used by existing figures or robustness checks do not follow the full manifest/raw/validated/summary/report schema; they are locked as legacy artifacts and are not treated as new full-schema experiments:
- `results`
- `results_1p5b`
- `results_32b`
- `results_olmo`
- `results_llama`
- `results_mistral`
- `results_ctx`
- `results_doubt`
- `results_verify`

## Planned Conditions And Null/Failed Runs

- EXPB_LOCAL_CERT_FLIP: all planned conditions reported: GLOBAL_BASELINE, LOCAL_CERTIFICATE, IRRELEVANT_CERTIFICATE_CONTROL. Null/negative result preserved: local certificate did not improve valid re-derivation over irrelevant-certificate control.
  - sample: {'early': 100, 'late': 100, 'mid': 100}; problem clusters: 183.
- EXPC_POLARITY_CONTROL: all planned 2x2 conditions reported. Null/ambiguous result preserved in report.
  - sample: 84 validated rows across 7 fully matched problems.
- EXPA_GLOBAL_EXPANSION: all planned conditions reported and the configured target was reached.
  - target: 150 global-falsehood examples per position; achieved: 150 problems and 1800 validated rows across matched conditions and positions.
  - artifact counts: failed, unavailable, and raw-generation attempts are retained alongside validated rows. Counts: `{"eligibility_audit.jsonl": 22132, "manifest.jsonl": 1800, "raw_generations.jsonl": 4832, "summary_validated_rows": 1800, "validated_outputs.jsonl": 1800}`

## Exclusion And Availability Counts

- EXPB_LOCAL_CERT_FLIP: No post-generation exclusions. Failed and unparsed generations are retained in denominators and logged.
- EXPB_LOCAL_CERT_FLIP sanity checks: `{"certificate_char_length_delta_local_minus_irrelevant": {"max": 1, "mean": -0.023, "min": -1}, "falsehood_false_rate": 1.0, "irrelevant_certificate_locally_falsifies_count": 0, "irrelevant_certificate_true_rate": 1.0, "local_certificate_true_rate": 1.0, "prompt_formatting": true}`
- EXPB_LOCAL_CERT_FLIP integrity checks: `{"all_original_proofs_validated": true, "duplicate_example_keys": 0, "every_injected_statement_audited": true, "failed_generations_logged": true, "required_result_fields_present": true, "unavailable_perturbations_logged": true, "unique_run_ids": true, "unparsed_generations_logged": true}`
- EXPC_POLARITY_CONTROL exclusion rule: `Fully paired examples only: all four conditions must be formally available at early, mid, and late positions before any model generation. This rule is fixed before generation and is applied identically across conditions.`
- EXPC_POLARITY_CONTROL availability counts: `{"duplicate_question": 139, "fully_matched_2x2_available": 127, "no_direct_local_negative_evidence_for_positive_claim": 1289, "no_unentailed_category_without_negative_evidence": 3150, "not_all_four_cells_available_at_all_positions": 1173, "original_proof_not_validated": 6984, "too_few_intermediate_entity_steps": 10677}`
- EXPC_POLARITY_CONTROL integrity checks: `{"all_injected_statements_audited": true, "all_original_proofs_validated": true, "all_planted_claims_audited_false": true, "all_required_result_fields": true, "duplicate_problem_condition_position_seed": [], "duplicate_problem_condition_position_seed_count": 0, "duplicate_run_id_count": 0, "prompt_hashes": ["f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c"], "strict_validator_separate_from_closure": true, "unique_run_ids": true}`
- EXPA_GLOBAL_EXPANSION eligibility counts: `{"candidate_falsehood_available": 5819, "duplicate_question": 139, "gold_not_solved": 926, "gold_not_validator_valid": 1892, "gold_valid_global_falsehood_available": 150, "no_global_falsehood_candidate": 1753, "too_few_dataset_steps": 11389, "too_few_model_intermediate_steps": 64}`
- EXPA_GLOBAL_EXPANSION integrity checks: `{"all_global_truth_false": true, "all_original_proofs_validated": true, "duplicate_count": 0, "duplicate_problem_condition_position_seed": [], "validated_rows": 1800}`

## Lock Assertions

- Every paper table with experimental numbers is generated by `src/paper_artifacts.py` and included from `paper_results.tex`.
- Figure sources read JSON result summaries through `src/make_workshop_figures.py`.
- The paper reports EXPA availability limits and noisy EXPC rather than filtering them out.
- Strict stepwise validation is reported in a dedicated paper table. EXPB has no strict validator, so stepwise recovery is explicitly unavailable there.
- The only manually written table left in `main.tex` is a qualitative example table; all experimental counts, rates, confidence intervals, and sample sizes are in generated macros/tables.
- No direct numeric-looking experimental rates were found inside non-generated LaTeX table/center blocks in `main.tex`.
