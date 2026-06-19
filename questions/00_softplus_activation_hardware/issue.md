# Issue - Test Why Smooth Activations Survive Hardware Pressure
**TLDR:** The screenshot asks why hardware-expensive activations are used if hardware matters. This issue turns that question into a reproducible benchmark comparing ReLU-like, smooth, and gated activations on cost and quality.

**Source artifacts.** Notebook/book photos saved at:

- `questions/00_softplus_activation_hardware/assets/photo_1.jpg`
- `questions/00_softplus_activation_hardware/assets/photo_2.jpg`

**Question packet.**

- `questions/00_softplus_activation_hardware/transcription.md`
- `questions/00_softplus_activation_hardware/pre_prompt.md`
- `questions/00_softplus_activation_hardware/PROTOCOL.md`
- `questions/00_softplus_activation_hardware/coding_agent_prompt.md`

**Compact question.**

- **Goal:** Test why smooth or gated activations such as softplus, GELU,
  GEGLU, and SwiGLU are used if hardware efficiency is so important.
- **Confidence:** Handwritten question high; `Noam S.` as Noam Shazeer medium.
- **Importance:** High, roughly 8/10.
- **Key uncertainty:** The source page is about softplus/sigmoid identities,
  but the handwritten question also points toward modern transformer
  activations.

**Question.** If hardware efficiency is important, why do successful ML systems
still use smooth or gated activations such as softplus, GELU, GEGLU, and
SwiGLU?

**Tasks**

- [ ] Re-read the source photos and refine the transcription if needed.
- [ ] Write a short note distinguishing softplus, GELU, SiLU/Swish, ReGLU,
  GEGLU, and SwiGLU.
- [ ] Build `expt_v1/` following `PROTOCOL.md` and `coding_agent_prompt.md`.
- [ ] Benchmark activation primitives and feed-forward blocks on CPU.
- [ ] If a GPU is available, run exactly one-GPU smoke benchmarks.
- [ ] Compare matched-parameter and unmatched GLU-family feed-forward blocks.
- [ ] Run one tiny controlled training task and report speed-quality tradeoffs.
- [ ] Save raw results, plots/tables, and `expt_v1/results/verdict.md`.
- [ ] Decide whether the hardware objection is strengthened, weakened, or
  ambiguous.

**Deliverable.** A reproducible `expt_v1/` benchmark and a one-paragraph verdict
on whether these activations earn their hardware cost.
