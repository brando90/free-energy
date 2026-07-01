# Dynamic Weighted CD for Continuous EBMs
**TLDR:** This packet turns the MCMC/Langevin handwritten notes into a concrete EBM research task: test whether a time-weighted mixture of short Langevin trajectory states can improve over CD-1 on a toy continuous density while preserving the correct theory about partition functions, MCMC, and score-based alternatives.

## Compact Question Summary

- **Goal:** Investigate whether explicit partition-function computation and standard long-run MCMC can be avoided for practical EBM training, and test a dynamic weighting scheme for contrastive divergence using Langevin trajectories.
- **Confidence:** The printed textbook content is clear; most handwritten questions are readable, but some margin notes are uncertain.
- **Importance:** High, roughly 9/10, because this is directly tied to whether EBMs can be scaled with parallel short-run dynamics.
- **Key uncertainty:** The proposed "later samples become positives" framing conflicts with the usual EBM likelihood-gradient sign. The prototype treats all trajectory samples as negative-phase samples and uses later samples as higher-confidence approximations to the model distribution.

## Source Artifacts

- `assets/photo_1.jpg`
- `assets/photo_2.jpg`
- `assets/photo_3.jpg`
- `assets/photo_4.jpg`
- Original upload paths:
  - `/tmp/codex-remote-attachments/019f0f6f-20b5-7122-af0d-38e0678c8ffb/343e750c-6b5e-4607-8a63-be93d13e0be5/1-Photo-1.jpg`
  - `/tmp/codex-remote-attachments/019f0f6f-20b5-7122-af0d-38e0678c8ffb/343e750c-6b5e-4607-8a63-be93d13e0be5/2-Photo-2.jpg`
  - `/tmp/codex-remote-attachments/019f0f6f-20b5-7122-af0d-38e0678c8ffb/343e750c-6b5e-4607-8a63-be93d13e0be5/3-Photo-3.jpg`
  - `/tmp/codex-remote-attachments/019f0f6f-20b5-7122-af0d-38e0678c8ffb/343e750c-6b5e-4607-8a63-be93d13e0be5/4-Photo-4.jpg`

## Files

- `transcription.md` - image transcription and uncertainty notes.
- `pre_prompt.md` - sharpened research framing.
- `research_brief.md` - theory, literature check, and verdict on the hypothesis.
- `ROADMAP.md` - corrected follow-up plan based on the Claude response.
- `RESULTS.md` - local and SNAP result summary.
- `PROTOCOL.md` - locked toy experiment protocol.
- `coding_agent_prompt.md` - paste-ready future-agent prompt.
- `issue.md` - concise GitHub issue body.
- `expt_v1/` - runnable PyTorch prototype for CD-1 vs dynamic weighted CD.

## Current Prototype

Run:

```bash
uv run python questions/02_ebm_dynamic_cd_langevin/expt_v1/src/dynamic_weighted_cd.py --steps 800 --out-dir questions/02_ebm_dynamic_cd_langevin/expt_v1/results
```

Smoke-tested command:

```bash
uv run python questions/02_ebm_dynamic_cd_langevin/expt_v1/src/dynamic_weighted_cd.py --steps 80 --batch-size 128 --grid-size 80 --eval-samples 512 --out-dir questions/02_ebm_dynamic_cd_langevin/expt_v1/results/smoke
```

The script writes `report.json`, `energy_surfaces.png`, `samples.png`, and `verdict.md`.

SNAP 5-seed result:

```bash
cd questions/02_ebm_dynamic_cd_langevin/expt_v1
./run_on_snap.sh snap_5seed
```

The aggregate is in `expt_v1/results/snap_5seed/aggregate.md`. Dynamic weighted
CD improved mean nearest-mode distance from `0.902 +/- 0.010` to
`0.783 +/- 0.022` while preserving full mode coverage, at about `2.25x` runtime
for this implementation.
