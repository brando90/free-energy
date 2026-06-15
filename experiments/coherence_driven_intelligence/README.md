# Coherence-Driven Intelligence Evaluation Notes

This folder captures the screenshots and a coding-agent prompt for testing Coherence-Driven Intelligence / Emergent Intelligence claims.

## Files

- `screenshots/photo_1.jpg` through `screenshots/photo_4.jpg`: saved source screenshots.
- `transcription.md`: visible text transcribed from the screenshots.
- `code_agent_prompt.md`: prompt for a coding agent to build a reproducible benchmark suite.
- `transformer_comparison_prompt.md`: focused prompt for a fair CDI/EI-vs-Transformer comparison on a cluster.

## Current Technical Judgment

The available public framing does not justify saying CDI/EI is a better alternative to transformers, CNNs, or ResNets yet.

The strongest fair statement is:

> CDI/EI may be an interesting non-backpropagation, coherence-basin-based classification approach, but it needs reproducible implementation details and matched benchmark evidence before it can be considered better than standard architectures.

Why the stronger claim is not established:

- The visible screenshots and public pages emphasize MNIST-like classification and basin visualizations, not broad language, vision, or multimodal capabilities.
- A method can avoid backpropagation and multiple epochs without being better. kNN, SVMs, nearest-centroid classifiers, and many optimization methods also avoid backpropagation.
- Transformers are primarily strong because of scalable sequence modeling, pretraining, transfer, long-context attention variants, and generation. A classifier demo does not test those abilities.
- CNNs and ResNets are strong image baselines because they scale beyond MNIST-style tasks to harder natural-image datasets. A fair comparison needs CIFAR-10/100 and eventually ImageNet-style evidence.
- The public Coherence Geometry seed note says the disclosure is non-enabling, does not provide full procedural reproducibility, and does not make leaderboard or SOTA claims.
- The public CDI page presents performance optimization, scaling, and comparison to conventional AI systems as future engineering directions.

What would make CDI/EI genuinely compelling:

- A runnable reference implementation.
- Reproduced results across multiple seeds.
- Matched preprocessing, splits, and tuning budgets.
- Comparable accuracy at lower measured energy, wall-clock time, memory, or data passes.
- Robustness beyond MNIST-like datasets.
- Quantified interpretability showing stable, faithful, actionable basin structure.
- Evidence on tasks where transformers or ResNets are actually relevant, not only toy classification.

## Sources Checked

- Nuveia profile for Dr. Barry L. Petersen: https://nuveia.com/blp/
- CDI overview: https://coherencegeometry.com/index.php/coherence-driven-intelligence/
- CDI seed note: https://coherencegeometry.com/index.php/2025/12/22/cgi-000001-v1-0-seed-note-coherence-driven-intelligence-cdi/
- Information and computation overview: https://coherencegeometry.com/index.php/information-computation/
