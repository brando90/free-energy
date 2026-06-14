# Pre-Prompt - Hypothesis 00: Boltzmann Exponential as a Hardware Liability

**TLDR:** Test whether the Boltzmann exponential `exp(-E)` is a bad
hardware/inference primitive, and whether a different monotone or normalized
score can preserve the useful ordering without paying exponential and
partition-function costs.

## Compact Hypothesis Summary

- **Goal:** Test whether the Boltzmann exponential `exp(-E)` is a bad
  hardware/inference primitive, and whether another monotone or normalized
  score can keep the useful ordering without paying exponential/partition
  costs.
- **Confidence:** ~75%.
- **Importance:** Very high, roughly 9.5/10.
- **Key uncertainty:** The handwritten note flags that "maybe any function is
  fine" if the model can learn enough from data; make this the main alternative
  hypothesis.

## Source

- Saved source photo: `assets/source_photo.jpg`
- Saved summary update photo: `assets/summary_update_photo.jpg`
- Original upload path: `/tmp/codex-remote-attachments/019ec76d-55f8-7671-aed0-2667ac416b8b/0d7a30d9-1e1b-47e1-9e9f-644e9753f2ff/1-Photo-1.jpg`
- Summary update upload path: `/tmp/codex-remote-attachments/019ec76d-55f8-7671-aed0-2667ac416b8b/ea880cd6-79b2-4fda-ad8e-23ca65ea9c0a/1-Photo-1.jpg`
- Visible formula: `p_theta(y | x) = exp(-E_theta(x, y)) / Z_theta(x)`
- Transcript confidence: low. Re-read the image before finalizing the exact
  research question.

## Rough User Seed

The summary update states:

> Hypothesis: `e^x` is a bad idea for hardware reasons.

The notebook seems to ask:

> Is there a Boltzmann/surjection-like object here? Where does it come from?
> Is it a set? Does it cause numerical issues like the partition function `Z`
> does? Can we define it from first principles using familiar objects such as
> matrices, graphs, or spaces? Is there a hypothesis about when the "not set" or
> many-to-one weighting is useful, maybe for proof cost or other structured
> problems?

## Prompt To Future Agent

Do not start by saying "this is just the partition function." First determine
whether the exponential Boltzmann form is a real hardware/inference liability,
then decide whether the older surjection/many-to-one reading adds anything.

Plan:

1. Re-transcribe both source photos and write the cleanest possible version of
   the one-sentence hypothesis, confidence, and importance.
2. Formalize the standard object:
   - conditional EBM: `p_theta(y | x) = exp(-E_theta(x, y)) / Z_theta(x)`
   - partition function: `Z_theta(x) = sum_y exp(-E_theta(x, y))`
   - log-probability: `log p_theta(y | x) = -E_theta(x, y) - log Z_theta(x)`
3. Interpret the "surjection-like" phrase as a possible many-to-one map from
   latent microstates to observed macrostates. Ask whether `Z` is implicitly
   counting or integrating over preimages.
4. Compare candidate mathematical readings:
   - Boltzmann distribution over a finite set
   - quotient or pushforward distribution under a many-to-one map
   - latent-variable marginalization
   - energy over proofs/programs/states where multiple derivations map to the
     same answer
5. Identify what numerical trouble is actually caused by `Z`:
   - overflow/underflow of `exp`
   - intractable exact summation or integration
   - high-variance Monte Carlo estimates
   - gradient bias from approximate negative samples
6. Build a toy finite example where many latent derivations map to one observed
   output, then show how energy, degeneracy, and normalization interact.
7. State falsification criteria. If the idea reduces to standard Boltzmann
   normalization with no new modeling or measurement consequence, say so.

Deliverable: a short note plus a toy example that decides whether this is a
useful new hypothesis or just terminology for known EBM normalization.
