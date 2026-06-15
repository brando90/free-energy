# Code Agent Prompt: Test Coherence-Driven Intelligence Claims

You are a rigorous ML research engineer. Build a reproducible experiment suite to test claims around Coherence-Driven Intelligence / Emergent Intelligence (CDI/EI), especially whether it is a better alternative to transformers, CNNs, ResNets, or other current neural architectures.

## Context

Local materials:

- `experiments/coherence_driven_intelligence/screenshots/photo_1.jpg`
- `experiments/coherence_driven_intelligence/screenshots/photo_2.jpg`
- `experiments/coherence_driven_intelligence/screenshots/photo_3.jpg`
- `experiments/coherence_driven_intelligence/screenshots/photo_4.jpg`
- `experiments/coherence_driven_intelligence/transcription.md`

Relevant public pages:

- https://nuveia.com/blp/
- https://coherencegeometry.com/index.php/coherence-driven-intelligence/
- https://coherencegeometry.com/index.php/2025/12/22/cgi-000001-v1-0-seed-note-coherence-driven-intelligence-cdi/
- https://coherencegeometry.com/index.php/information-computation/

Claims to test:

- CDI/EI is more training-efficient than current state-of-the-art models.
- It can reduce energy consumption.
- It does not need backpropagation.
- It trains without multiple epochs.
- It is less black-box because internal class/response structure appears as visible coherence basins.
- It is fundamentally different infrastructure from standard neural-network training.

Important caveat from the public materials: treat CDI/EI as a proof-of-principle unless you find a fully enabling implementation or paper. Do not assume it is SOTA. Do not claim a fair comparison to transformers or ResNets unless the implementation, task, compute budget, and evaluation protocol are comparable.

## Objective

Answer this question empirically:

> Under matched tasks, matched data splits, matched preprocessing, and measured compute/energy budgets, does CDI/EI provide a better accuracy-efficiency-interpretability tradeoff than standard baselines?

The answer must distinguish:

- `validated`: supported by reproduced experiments with fair baselines.
- `partially supported`: true in a narrow setting, such as MNIST-like classification, but not broader.
- `not supported`: contradicted by fair experiments.
- `not testable`: public materials lack enough procedural detail or code.

## Required Approach

1. Source and reproducibility audit

   - Fetch the linked public materials and any linked Zenodo artifacts.
   - Record all artifact URLs, versions, checksums, and licenses.
   - Determine whether a runnable CDI/EI reference implementation exists.
   - If no enabling implementation exists, produce a reproducibility report and stop before inventing a method that could be unfairly attributed to the author.
   - If only pseudocode or partial math exists, implement a clearly labeled `cdi_surrogate` and separate it from `cdi_reference`.

2. Build a controlled benchmark harness

   - Use a fresh folder such as `experiments/coherence_driven_intelligence/eval/`.
   - Include `README.md`, `requirements.txt` or `pyproject.toml`, `src/`, `configs/`, `scripts/`, `results/`, and `reports/`.
   - Every run must write a machine-readable result file with config, seed, git SHA if available, hardware info, package versions, wall-clock time, memory, and energy measurements when available.
   - Run at least 5 seeds for final comparisons.

3. Datasets

   Start with datasets close to the apparent CDI/EI demonstration:

   - MNIST
   - Fashion-MNIST
   - KMNIST or EMNIST
   - CIFAR-10 only after MNIST-like datasets are working

   Add sequence/text tasks only if the CDI/EI implementation naturally supports sequences:

   - AG News or IMDb for small text classification
   - Do not compare CDI/EI to LLMs on open-ended generation unless CDI/EI has an actual generative language implementation.

4. Baselines

   Include baselines that are fair for small-data classification and for modern neural comparisons:

   - Logistic regression
   - k-nearest neighbors
   - SVM or kernel ridge model, if computationally feasible
   - PCA plus nearest centroid
   - Small MLP
   - LeNet-style CNN
   - Small ResNet, such as ResNet-18 adapted to small images
   - Small Vision Transformer or patch transformer
   - Small text transformer only for text tasks

   Do not compare CDI/EI against huge pretrained LLMs unless the question is explicitly about task coverage rather than training efficiency.

5. Metrics

   Measure:

   - Accuracy, macro F1, per-class accuracy, and confusion matrix
   - Calibration, such as ECE or Brier score, if class probabilities exist
   - Wall-clock train time
   - Wall-clock inference latency and throughput
   - Peak memory
   - Parameter count or state size
   - Effective data passes / epochs
   - Energy estimate using the best available tool for the hardware
   - Energy per correctly classified test example
   - Time-to-threshold accuracy, for example time to 95%, 98%, and 99% MNIST accuracy
   - Robustness under noise, rotations, translations, occlusion, and OOD datasets
   - Sample efficiency at 1%, 5%, 10%, 25%, 50%, and 100% of training data

6. Energy measurement

   Prefer real measurements over estimates.

   - Linux CPU: RAPL or pyJoules when available.
   - NVIDIA GPU: `nvidia-smi`, NVML, or CodeCarbon.
   - macOS: document limits; use powermetrics if permissions allow, otherwise report wall-clock and estimated package energy separately.
   - Always report hardware model and whether energy is measured or estimated.

7. Fairness rules

   - Same train/test splits across all methods.
   - Same preprocessing unless a method requires a documented transform.
   - Equal hyperparameter-search budget by method family.
   - No test-set tuning.
   - Report failures and unstable runs.
   - Include confidence intervals across seeds.
   - Compare against both simple baselines and modern neural baselines. A method that beats a CNN but loses to kNN on MNIST is not a strong general alternative.

8. Interpretability evaluation

   Do not rely on screenshots alone. Quantify interpretability claims where possible:

   - Are class basins stable across random seeds?
   - Are basin assignments faithful to predictions?
   - Can a human-visible intervention improve or repair a class error?
   - Are basin plots reproducible under reruns and data resampling?
   - Does the visualization scale past 10 classes or higher-dimensional inputs?
   - Compare to standard representation visualizations: PCA, UMAP, t-SNE, activation clustering, saliency maps, and prototype methods.

9. Statistical analysis

   - Use bootstrap confidence intervals for accuracy and energy-per-correct-example.
   - Use paired comparisons where seeds/splits are shared.
   - Report effect sizes, not only p-values.
   - Acknowledge when differences are within measurement noise.

10. Final report

   Produce `reports/final_report.md` with:

   - Executive summary
   - Reproducibility status
   - Claim-by-claim verdict table
   - Method descriptions
   - Dataset and preprocessing details
   - Hardware and measurement details
   - Main result tables
   - Robustness and sample-efficiency results
   - Interpretability findings
   - Failure cases
   - Clear answer to: "Is CDI/EI actually a better alternative to transformers, CNNs, or ResNets?"

## Initial Hypotheses

Use these as hypotheses, not conclusions:

- CDI/EI may be promising if it reaches comparable accuracy with fewer data passes, less energy, and reproducible basin-level interpretability on the same tasks.
- CDI/EI is not a demonstrated replacement for transformers unless it handles sequence modeling, attention-like long-range dependencies, scaling, pretraining, transfer, and generation.
- CDI/EI is not a demonstrated replacement for CNNs or ResNets unless it matches them on vision benchmarks beyond MNIST-like toy tasks, including robustness and CIFAR/ImageNet-scale behavior.
- "No backpropagation" is not enough by itself. kNN, SVMs, random forests, nearest-centroid classifiers, and many optimization methods also avoid backpropagation; the relevant question is the full accuracy-efficiency-scalability tradeoff.

## Stop Conditions

Stop and produce a clear report if:

- No executable CDI/EI method can be obtained from public artifacts.
- The method cannot be implemented without speculative choices that dominate performance.
- Energy measurement is not available; in that case, complete accuracy/time/memory comparisons and label energy results as unavailable.

## Expected Bottom Line Format

Use this exact final table format:

| Claim | Test | Result | Verdict |
| --- | --- | --- | --- |
| More efficient training | Matched wall-clock and energy on MNIST-like datasets | TBD | validated / partially supported / not supported / not testable |
| Reduces energy consumption | Joules per correct test prediction | TBD | validated / partially supported / not supported / not testable |
| No backpropagation | Source/code audit | TBD | validated / partially supported / not supported / not testable |
| Trains without multiple epochs | Effective data pass accounting | TBD | validated / partially supported / not supported / not testable |
| Less black-box | Basin stability and intervention tests | TBD | validated / partially supported / not supported / not testable |
| Better alternative to transformers/CNNs/ResNets | Cross-task benchmark comparison | TBD | validated / partially supported / not supported / not testable |
