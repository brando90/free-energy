# Softplus Activation Hardware Question
**TLDR:** This packet preserves two source photos and sharpens the question: if hardware efficiency matters so much, why do modern ML systems still use smooth or gated activations such as softplus, GELU, GEGLU, and SwiGLU? The follow-up experiment should separate activation math cost from fused-kernel reality and model-quality gains.

## Compact Question Summary

- **Goal:** Test why smooth or gated activations such as softplus, GELU,
  GEGLU, and SwiGLU are used if hardware efficiency is so important, and when
  their training or quality benefits outweigh their hardware cost.
- **Confidence:** Handwritten question transcription is high. Interpreting
  "Noam S." as Noam Shazeer and his GLU variants is medium.
- **Importance:** High, roughly 8/10.
- **Key uncertainty:** The question may be about textbook softplus/sigmoid
  math, transformer feed-forward activations, or hardware-aware activation
  design more broadly. Treat the experiment as a way to separate these.

## Source Artifacts

- `assets/photo_1.jpg`
- `assets/photo_2.jpg`
- Original upload paths:
  - `/tmp/codex-remote-attachments/019ee07e-f55d-73b1-9095-0311a118709c/5ffdeeb6-670b-452a-a33c-8c4ccf2f6b35/1-Photo-1.jpg`
  - `/tmp/codex-remote-attachments/019ee07e-f55d-73b1-9095-0311a118709c/5ffdeeb6-670b-452a-a33c-8c4ccf2f6b35/2-Photo-2.jpg`

## Files

- `transcription.md` - image transcription and uncertainty notes.
- `pre_prompt.md` - sharpened research framing for a future agent.
- `PROTOCOL.md` - locked experiment protocol before any expensive run.
- `coding_agent_prompt.md` - paste-ready prompt for implementing `expt_v1/`.
- `issue.md` - GitHub issue body.

## GitHub Issue

- [#55: Q00: Test why smooth activations survive hardware pressure](https://github.com/brando90/free-energy/issues/55)
