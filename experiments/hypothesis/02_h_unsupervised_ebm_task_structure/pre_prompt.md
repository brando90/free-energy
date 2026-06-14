# Pre-Prompt - Hypothesis 02: Transformers May Already Solve The Input Issue

**TLDR:** Test whether transformers and attention already solve the alleged
unbounded-length/full-input conditioning issue, leaving little special
advantage for EBMs on this axis.

## Compact Hypothesis Summary

- **Goal:** Test whether transformers/attention already solve the alleged
  unbounded-length/full-input conditioning issue, leaving little special
  advantage for EBMs on this axis.
- **Confidence:** High.
- **Importance:** Very high, 9.8/10.
- **Key uncertainty:** The useful comparison may be attention versus recurrence
  and forgetting, not EBM versus AR by itself.

## Source

- Saved source photo: `assets/source_photo.jpg`
- Saved summary update photo: `assets/summary_update_photo.jpg`
- Original upload path: `/tmp/codex-remote-attachments/019ec76d-55f8-7671-aed0-2667ac416b8b/0d7a30d9-1e1b-47e1-9e9f-644e9753f2ff/3-Photo-3.jpg`
- Summary update upload path: `/tmp/codex-remote-attachments/019ec76d-55f8-7671-aed0-2667ac416b8b/ea880cd6-79b2-4fda-ad8e-23ca65ea9c0a/3-Photo-3.jpg`
- Visible formulas resemble unconditional and conditional EBM normalization.
- Transcript confidence: medium-low. Re-read the image before finalizing.

## Rough User Seed

The summary update states:

> Hypothesis: transformers solve the unbounded-length/full-input issue.
> Importance 9.8/10, high confidence.

The notebook seems to ask:

> If the EBM is unsupervised, how do we solve the task issue? Maybe the model
> should learn/transfer energy structure from the inputs, but perhaps
> transformers already do this. Can a dumb chat/task setup use it? Compare to
> LSTM or attention. Is attention the magic, with no free energy advantage?

## Prompt To Future Agent

Do not treat "unsupervised learns structure" as an answer. First test whether
attention/transformers already solve the input-coverage or forgetting issue
that EBMs are being credited with solving.

Plan:

1. Re-transcribe both source photos and separate four claims:
   - transformers/attention solve the full-input or long-context issue
   - unsupervised EBMs learn semantic/task structure
   - learned energy transfers to downstream tasks
   - attention/AR baselines may already explain the gains
2. Define the target claim in measurable terms:
   - task accuracy after frozen energy pretraining
   - sample efficiency after light fine-tuning
   - OOD robustness or calibration
   - search/reranking improvement using energy
3. Build a small benchmark with controlled labels:
   - synthetic sequence task with known latent factors
   - proof-step validity or arithmetic trace validity
   - text classification with concept perturbations
4. Compare models:
   - unconditional EBM over inputs
   - conditional EBM over `(x, y)` or `(context, completion)`
   - AR transformer log-prob baseline
   - LSTM baseline
   - contrastive or masked-model baseline if cheap
5. Control for capacity, data, and inference compute.
6. Run transfer tests:
   - train energy unsupervised, freeze or lightly tune, then evaluate task
   - use energy as reranker for candidate completions
   - test whether improvements remain when labels are permuted or latent
     factors are ablated
7. State kill criteria. If the EBM matches weaker baselines but not the
   compute-matched attention/AR baseline, the unsupervised-task-structure claim
   is not supported.

Deliverable: a falsifiable experiment plan plus a minimal prototype showing
whether unsupervised energy learning gives task structure beyond standard
sequence-modeling baselines.
