# Toy EBM Training Plan

This experiment turns Brando's Lean AI Club toy idea into a controlled
finite-support EBM training benchmark. It is not an embedding experiment.

The core question is:

> If we train an energy function with the EBM maximum-likelihood update,
> can small neural architectures recover a structured target distribution
> over task-conditioned proof/code candidates?

The toy deliberately makes the partition function tractable. That lets us
separate "does the EBM update work?" from "can we sample from the model?"
before moving to short-run MCMC or Lean candidate pools.

## Source Notes

Raw meeting-note photos are stored in:

- `assets/toy_ebm_notes_photo_1.jpg`
- `assets/toy_ebm_notes_photo_2.jpg`

The readable transcription is in `TRANSCRIPTION.md`. Ambiguous handwriting is
marked with `unclear`.

## Mathematical Object

For each task/context `c`, define a finite candidate space

```text
X = {0, 1}^L.
```

The EBM is conditional:

```text
p_theta(x | c) = exp(-E_theta(c, x)) / Z_theta(c)
Z_theta(c) = sum_{x in X} exp(-E_theta(c, x)).
```

The exact conditional negative log-likelihood for the data distribution
`p_star(x | c)` is

```text
L(theta; c) = E_{x ~ p_star(. | c)}[E_theta(c, x)] + log Z_theta(c).
```

The gradient is the positive phase minus the model expectation:

```text
grad_theta L
  = E_{p_star}[grad_theta E_theta(c, x)]
    - E_{p_theta}[grad_theta E_theta(c, x)].
```

This is the handwritten update in executable form. In this toy, both
expectations can be evaluated exactly by enumeration.

## Synthetic Data Distribution

Each task is a binary vector `c` of length `L`. A candidate `x` is another
binary vector of the same length. The hidden data energy rewards five features:

- positionwise agreement between `x_i` and `c_i`;
- one-step shifted agreement between `x_i` and `c_{i-1}`;
- local smoothness in `x`;
- a global parity match between `x` and `c`;
- endpoint agreement.

Then

```text
p_star(x | c) proportional_to exp(-E_star(c, x) / temperature).
```

This target is intentionally mixed:

- a linear independent-token model should learn the easy copy signal but miss
  local/global dependencies;
- CNNs and ResNets should capture local structure;
- the MLP and transformer should have access to global parity-like structure.

## Models

Train the following PyTorch energy functions:

1. `linear`: position/token additive energy over `(c_i, x_i)`.
2. `mlp`: two-layer feed-forward network on flattened one-hot token pairs.
3. `cnn`: small 1D convolutional energy model.
4. `resnet`: residual 1D convolutional energy model.
5. `transformer`: compact encoder over token-pair embeddings.

All models return one scalar energy per `(task, candidate)` pair.

## Evaluation

For held-out tasks, enumerate all candidates and compare the learned
distribution `p_theta(. | c)` to `p_star(. | c)`:

- `kl_pstar_model`: `KL(p_star || p_theta)`, lower is better;
- `tv_distance`: total variation distance, lower is better;
- `nll_pstar`: exact expected NLL under `p_star`, lower is better;
- `target_mode_rank`: rank of the true target mode under learned energy;
- `mode_match_rate`: fraction of tasks where the learned mode equals the
  target mode.

The uniform distribution is reported as a baseline.

## Execution Plan

1. Preserve the raw photos under the experiment assets directory.
2. Transcribe the notes with uncertainty markers.
3. Implement a self-contained PyTorch runner:
   - finite support enumeration;
   - synthetic `p_star`;
   - exact conditional EBM objective;
   - five model families;
   - JSON and markdown reports.
4. Add smoke tests:
   - support enumeration and normalization;
   - model forward passes;
   - one short training run that must improve KL over the uniform baseline.
5. Run a smoke experiment.
6. Run a real exact finite-support experiment across all five models.
7. Save the reports in `results/`.

## Commands

Smoke test:

```bash
cd /Users/brandomiranda/free-energy
./experiments/01_toy_ebm_training/run_smoke_test.sh
```

Real exact finite-support run:

```bash
cd /Users/brandomiranda/free-energy
.venv/bin/python experiments/01_toy_ebm_training/run_toy_ebm.py \
  --tag real_exact \
  --models linear mlp cnn resnet transformer \
  --seq-len 9 \
  --num-train-tasks 48 \
  --num-test-tasks 16 \
  --epochs 80 \
  --batch-size 8 \
  --hidden-dim 64 \
  --lr 0.003 \
  --device auto
```

## Success Criteria

The implementation is considered ready when:

- `.venv/bin/python -m pytest experiments/01_toy_ebm_training/test_toy_ebm.py` passes;
- `run_smoke_test.sh` completes and writes a passing JSON report;
- the real experiment writes both JSON and markdown reports;
- at least one nonlinear model beats the uniform baseline on held-out KL.

