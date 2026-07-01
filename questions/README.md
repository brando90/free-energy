# Research Questions From Screenshots
**TLDR:** This folder stores numbered research-question packets extracted from user screenshots. Each packet preserves the source images, a transcription, a sharpened pre-prompt, a locked protocol, a coding-agent prompt, and a GitHub-issue body.

Use this folder when the user sends screenshots with the trigger phrase `Q go`.
Number folders sequentially as `00_<short_slug>`, `01_<short_slug>`, and so on.

## Q Go Workflow

When a user sends screenshots plus `Q go`:

1. Save the source images under `questions/<NN>_<slug>/assets/`.
2. Create or update these files:
   - `README.md`
   - `transcription.md`
   - `pre_prompt.md`
   - `PROTOCOL.md`
   - `coding_agent_prompt.md`
   - `issue.md`
3. Transcribe user handwriting directly. For copyrighted printed material,
   preserve equations and brief bibliographic context, but summarize long prose
   instead of copying full passages.
4. Follow the experiment-folder style from `experiments/hypothesis/`: compact
   goal, confidence, importance, source artifact paths, future-agent prompt,
   issue body, and a clear deliverable.
5. Open a GitHub issue from `issue.md`, then link that issue in this index and
   the question packet.
6. Run QA, commit only the relevant question/config files, push to `main`, and
   verify `origin/main` contains the commit.

## Index

| ID | Folder | Prompt | Source images | GitHub issue | Goal | Confidence | Importance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | `00_softplus_activation_hardware` | `coding_agent_prompt.md` | `assets/photo_1.jpg`, `assets/photo_2.jpg` | [#55](https://github.com/brando90/free-energy/issues/55) | Test why smooth or gated activations such as softplus, GELU, GEGLU, and SwiGLU are used if hardware efficiency is so important. | Handwriting high; Noam S. interpretation medium | High, 8/10 |
| 02 | `02_ebm_dynamic_cd_langevin` | `coding_agent_prompt.md` | `assets/photo_1.jpg`, `assets/photo_2.jpg`, `assets/photo_3.jpg`, `assets/photo_4.jpg` | Not opened | Test whether dynamic weighting over short Langevin CD trajectories improves continuous EBM training while keeping the partition-function theory correct. | Printed text high; handwriting medium | High, 9/10 |
