# Code Agent Prompt: Fair CDI/EI vs Transformer Comparison on SNAP Cluster

You are a rigorous ML systems and evaluation engineer with access to a SNAP cluster. Build a fair, reproducible comparison between Coherence-Driven Intelligence / Emergent Intelligence (CDI/EI) and Transformer-based models across training, evaluation, inference, energy, interpretability, and tradeoffs.

The goal is not to prove either side wins. The goal is to produce a defensible answer:

> In which settings, if any, does CDI/EI outperform a Transformer baseline, and in which settings is a Transformer clearly better?

## Core Fairness Rule

Separate these two comparison lanes:

1. **Matched training lane**
   - CDI/EI and Transformer models are trained from scratch on the same task, same train/validation/test split, same preprocessing, same hardware class, and comparable hyperparameter-search budgets.
   - This lane answers: "Is CDI/EI more efficient or accurate than a Transformer architecture under matched conditions?"

2. **Pretrained capability lane**
   - Evaluate pretrained Transformer LMs on standard benchmarks such as MMLU-style multiple-choice tasks.
   - Only compare CDI/EI here if there is a real CDI/EI language or text-choice scoring implementation.
   - This lane answers: "Does CDI/EI currently cover the same task surface as pretrained Transformers?"

Do not collapse these lanes into one leaderboard. A CDI/EI classifier should not be compared directly against a pretrained LLM unless the report labels the comparison as task coverage, not matched training efficiency.

## Local Context

Use these local files as context:

- `experiments/coherence_driven_intelligence/transcription.md`
- `experiments/coherence_driven_intelligence/README.md`
- `experiments/coherence_driven_intelligence/code_agent_prompt.md`
- `experiments/coherence_driven_intelligence/screenshots/photo_1.jpg`
- `experiments/coherence_driven_intelligence/screenshots/photo_2.jpg`
- `experiments/coherence_driven_intelligence/screenshots/photo_3.jpg`
- `experiments/coherence_driven_intelligence/screenshots/photo_4.jpg`

Public references to audit before implementing:

- https://nuveia.com/blp/
- https://coherencegeometry.com/index.php/coherence-driven-intelligence/
- https://coherencegeometry.com/index.php/2025/12/22/cgi-000001-v1-0-seed-note-coherence-driven-intelligence-cdi/
- https://coherencegeometry.com/index.php/information-computation/

## Claims Under Test

Evaluate the following claims independently:

- CDI/EI trains more efficiently than Transformers.
- CDI/EI uses less energy for comparable accuracy.
- CDI/EI avoids backpropagation.
- CDI/EI avoids multi-epoch training.
- CDI/EI is less black-box due to visible coherence basins.
- CDI/EI is a better alternative to Transformers.
- CDI/EI has a better tradeoff curve when considering accuracy, energy, latency, memory, sample efficiency, robustness, interpretability, and implementation complexity.

## First Step: Reproducibility Audit

Before writing benchmark code, determine whether CDI/EI is actually runnable.

1. Fetch public artifacts, including linked Zenodo records if available.
2. Record URLs, versions, checksums, licenses, and dates accessed.
3. Identify whether there is:
   - a runnable reference implementation,
   - pseudocode sufficient for faithful implementation,
   - only conceptual/math material,
   - or only screenshots and high-level claims.
4. If there is no enabling implementation, stop and write `reports/reproducibility_audit.md`.
5. If you create a surrogate implementation, call it `cdi_surrogate`, not `cdi_reference`, and report that results do not validate the original method.

## Experiment Repository Layout

Create the benchmark suite under:

`experiments/coherence_driven_intelligence/transformer_eval/`

Required structure:

- `README.md`
- `pyproject.toml` or `requirements.txt`
- `configs/`
- `src/`
- `scripts/`
- `cluster/`
- `results/raw/`
- `results/tables/`
- `results/figures/`
- `reports/`
- `checksums/`

Every run must emit JSONL or Parquet records with:

- run id
- git SHA if available
- hostname
- scheduler job id
- GPU model and count
- CPU model
- memory
- package versions
- dataset version and checksum
- model config
- seed
- optimizer or update rule
- train examples seen
- effective epochs / data passes
- wall-clock train time
- wall-clock eval time
- peak memory
- measured or estimated energy
- final metrics
- checkpoint path
- failure status if any

## SNAP Cluster Requirements

Assume a SLURM-like cluster unless the environment proves otherwise.

Create:

- `cluster/setup_env.sh`
- `cluster/train_matched.sbatch`
- `cluster/eval_matched.sbatch`
- `cluster/eval_pretrained_lm.sbatch`
- `cluster/aggregate_results.sbatch`
- `cluster/README.md`

Cluster prompt constraints:

- Do not hardcode one specific GPU type unless the cluster requires it.
- Detect available GPUs and log `nvidia-smi -L`.
- Put datasets and model caches on scratch or project storage, not home, if available.
- Use resumable checkpoints.
- Avoid interactive-only workflows.
- Make every job re-runnable and idempotent.
- Support array jobs over seeds, datasets, and model sizes.
- Include CPU-only smoke tests for CI/local debugging.
- Log stdout/stderr to `logs/%x_%A_%a.out` and `logs/%x_%A_%a.err`.

If the cluster uses different module names, leave clearly marked variables at the top of the scripts:

- `CONDA_ENV`
- `PROJECT_DIR`
- `SCRATCH_DIR`
- `DATA_DIR`
- `CACHE_DIR`
- `PARTITION`
- `ACCOUNT`
- `GPU_CONSTRAINT`

## Matched Training Lane

### Tasks

Start with tasks where a CDI/EI classifier could plausibly run:

- MNIST
- Fashion-MNIST
- KMNIST or EMNIST
- CIFAR-10
- CIFAR-100 if CIFAR-10 succeeds

Add text classification only if CDI/EI supports text features:

- AG News
- IMDb
- SST-2

Do not use MMLU in the matched training lane unless both systems are trained or finetuned under a clear, matched, multiple-choice classification protocol.

### Transformer Baselines

For vision:

- Tiny ViT
- Small ViT
- Patch transformer with fixed patch size
- Optional hybrid CNN stem plus transformer encoder

For text:

- Small encoder-only transformer trained from scratch.
- Optional pretrained encoder fine-tuning, reported separately from scratch training.

Also include non-Transformer baselines for calibration:

- logistic regression
- k-nearest neighbors
- nearest centroid
- MLP
- LeNet-style CNN
- small ResNet

Reason: if CDI/EI beats a tiny Transformer but loses to kNN or a small CNN, it is not a strong claim against Transformers.

### Training Metrics

Track:

- cross-entropy loss for Transformer and any model with probabilistic logits
- CDI/EI native objective or update functional
- comparable classification loss after readout, if possible
- train accuracy
- validation accuracy
- validation macro F1
- data passes / effective epochs
- examples processed per second
- tokens or pixels processed per second
- gradient steps for Transformer
- CDI/EI update steps or relaxation steps
- wall-clock time to threshold accuracy
- energy to threshold accuracy

Thresholds:

- MNIST-family: 95%, 98%, 99% accuracy where applicable
- CIFAR-10: 60%, 75%, 85%, 90% accuracy where applicable
- Text classification: 70%, 80%, 85%, 90% accuracy where applicable

If CDI/EI does not optimize cross-entropy, do not force it to. Report:

- native CDI/EI objective curve,
- downstream classification metrics,
- and cross-entropy only if calibrated probabilities or logits are available from a fair readout.

### Hyperparameter Search

Use equal search budgets by method family:

- same number of trials per dataset and model-size class
- same validation split
- no test-set tuning
- same seed list for final runs
- final results over at least 5 seeds

Report total search cost separately from best-run training cost.

## Pretrained Capability Lane

Use this lane only for pretrained Transformer LMs and any CDI/EI system that can naturally answer text multiple-choice questions.

Suggested standard evaluations:

- MMLU-style multiple choice
- ARC-Challenge
- HellaSwag
- WinoGrande
- TruthfulQA multiple choice
- GSM8K only if the system can generate or select mathematical answers

Use an established harness, such as `lm-evaluation-harness`, when feasible.

Report:

- zero-shot accuracy
- few-shot accuracy where supported
- exact prompt templates
- tokenizer
- context length
- batch size
- decoding parameters
- total tokens processed
- wall-clock eval time
- peak memory
- energy
- cost estimate if using external APIs, though prefer local models on cluster

Important:

- If CDI/EI cannot process natural language or score answer choices, mark these evaluations as `not applicable`, not as failures.
- If CDI/EI uses frozen text embeddings from a Transformer, report it as a hybrid method and attribute the representation cost to the Transformer component.

## Inference Benchmarking

Benchmark inference separately from training.

Measure:

- single-example latency
- batch throughput
- p50/p90/p99 latency
- memory at batch sizes 1, 8, 32, 128 where applicable
- energy per prediction
- model/state size on disk
- warm-start and cold-start latency
- CPU-only fallback performance
- GPU utilization
- scaling across 1 GPU vs multi-GPU only if the method supports it

For language models:

- prefill throughput
- decode throughput
- tokens per second
- KV cache memory
- latency as context length grows

For CDI/EI:

- relaxation/update steps per inference
- convergence criteria
- sensitivity to stopping tolerance
- basin lookup/readout cost

## Robustness and Generalization

Evaluate:

- Gaussian noise
- blur
- occlusion
- rotations
- translations
- corruptions appropriate to each dataset
- OOD transfer, such as MNIST to Fashion-MNIST only as diagnostic, not as a normal accuracy claim
- adversarial stress tests if practical
- performance at 1%, 5%, 10%, 25%, 50%, and 100% of training data

## Interpretability and Black-Box Claims

Quantify the claim that CDI/EI is less black-box:

- basin stability across seeds
- basin separability by class
- correspondence between basin assignment and prediction
- reproducibility of basin plots across reruns
- ability to intervene on basin/channel/phase parameters to fix known errors
- failure modes where basins overlap or fragment

Compare to Transformer interpretability baselines:

- attention maps where relevant
- representation PCA/UMAP
- probing classifiers
- activation clustering
- saliency or attribution methods for vision/text

Do not treat a nice-looking scatter plot as sufficient evidence. Require quantitative stability and faithfulness.

## Energy and Systems Measurement

Prefer measured energy:

- NVIDIA GPUs: NVML / `nvidia-smi` sampling / CodeCarbon
- CPU: RAPL or pyJoules where available
- cluster-level power tools if SNAP provides them

Always report:

- whether energy is measured or estimated
- sampling interval
- idle baseline subtraction policy
- GPU utilization
- CPU utilization
- memory bandwidth if available

Primary energy metric:

- joules per correct test prediction

Secondary energy metrics:

- joules to threshold accuracy
- joules per epoch/effective data pass
- joules per 1,000 inferences
- joules per 1,000 generated tokens for language models

## Statistical Reporting

Use:

- mean and standard deviation across seeds
- bootstrap confidence intervals
- paired comparisons where seeds/splits align
- effect sizes
- rank stability across datasets

Do not make winner claims based on one seed or one dataset.

## Final Deliverables

Produce:

- `reports/reproducibility_audit.md`
- `reports/matched_training_report.md`
- `reports/pretrained_capability_report.md`
- `reports/inference_report.md`
- `reports/energy_report.md`
- `reports/interpretability_report.md`
- `reports/final_transformer_comparison.md`
- `results/tables/main_results.csv`
- `results/tables/energy_results.csv`
- `results/tables/inference_results.csv`
- `results/figures/tradeoff_frontiers.png`
- `results/figures/time_to_accuracy.png`
- `results/figures/energy_to_accuracy.png`

## Required Final Verdict Table

Use this exact table in `reports/final_transformer_comparison.md`:

| Dimension | Best CDI/EI result | Best Transformer result | Winner | Caveat |
| --- | --- | --- | --- | --- |
| Matched accuracy | TBD | TBD | TBD | TBD |
| Cross-entropy / native loss | TBD | TBD | TBD | CDI/EI may not use CE |
| Time to threshold accuracy | TBD | TBD | TBD | TBD |
| Energy to threshold accuracy | TBD | TBD | TBD | TBD |
| Inference latency | TBD | TBD | TBD | TBD |
| Inference throughput | TBD | TBD | TBD | TBD |
| Peak memory | TBD | TBD | TBD | TBD |
| State/model size | TBD | TBD | TBD | TBD |
| Robustness | TBD | TBD | TBD | TBD |
| Sample efficiency | TBD | TBD | TBD | TBD |
| Interpretability | TBD | TBD | TBD | Requires quantitative faithfulness |
| MMLU-style task coverage | TBD | TBD | TBD | Only fair if CDI/EI supports language QA |
| Implementation maturity | TBD | TBD | TBD | TBD |

Then answer plainly:

1. Is CDI/EI better than Transformers under matched training?
2. Is CDI/EI better than pretrained Transformers on standard language evaluations?
3. Does CDI/EI offer a better energy/accuracy tradeoff?
4. Is the method reproducible enough to validate the claims?
5. What would need to be true for CDI/EI to become a serious Transformer alternative?

## Failure/Stop Conditions

Stop and report rather than overclaim if:

- CDI/EI cannot be run from public materials.
- CDI/EI requires speculative implementation choices that dominate the result.
- MMLU-style evaluation is not applicable to CDI/EI.
- Energy measurement is unavailable.
- Cluster jobs fail repeatedly for infrastructure reasons.

Use verdicts:

- `validated`
- `partially supported`
- `not supported`
- `not applicable`
- `not testable`

Do not use stronger language than the data supports.
