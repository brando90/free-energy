# TASK BLOCKED: Project path unresolved — codebase not present on this machine

## Status
The orchestrator passed the literal string `undefined` for every project path ("Local copy of the project is at undefined", "undefined/exp", "undefined/paper", "undefined/PROFESSOR_FEEDBACK.md"). A variable substitution failure upstream. The latent-recovery project source tree (paper LaTeX, `exp/` code, results directories) **does not exist anywhere on this machine**. The requested inventory cannot be built from local files; do not treat the partial data below as the "full menu" — it is paper-derived fallback only.

## Search evidence (exhaustive)
Searched with `mdfind` (Spotlight) and `find` across `/Users/elyas` (Desktop, Documents, Downloads, home, dotfiles), `/Volumes/Data` (contains only `AppleInternal`), for all named artifacts. Zero hits for any of:
- `PROFESSOR_FEEDBACK.md`, `RESULTS_LOCK.md`, `UPDATE.md` (project), `PAPER_CHANGELOG.md`, `exp/PLAN.md`
- `EXPA_REPORT.md` / `EXPB_REPORT.md` / `EXPC_REPORT.md`, `EXPA_GLOBAL_EXPANSION`, `EXPB_LOCAL_CERT_FLIP`, `EXPC_POLARITY_CONTROL`
- `validated_summary*.json`, `matched_rho.json`, `functional_rho.json`, `rho_stability.json`, `cluster_stats.json`, `shuffle_control.json`, `geometry.json`, `doubt_judge.json`
- `results_1p5b`, `results_32b`, `results_llama`, `results_mistral`, `results_olmo`, `results_ctx`, `results_doubt`, `results_verify`, `anchor_knockout`, `anchor_qa`, `arith`, `gsm8k`, `r1`, `probes`, `null*`, `matched_pert.py`
- any file/dir matching `*prontoqa*` or `*latent*` (except items below)

## What DOES exist locally (compiled PDFs only, no source)
- `/Users/elyas/Desktop/main.pdf` — **newest compiled draft, Jun 23 2026**, title "Nearby Evidence Shapes How Transformers Respond to Planted Errors in Their Own Reasoning"
- `/Users/elyas/Desktop/latent_recovery_paper_latest.pdf` (Jun 16), `latent_recovery_paper.pdf` (Jun 16), `latent_recovery_paper_v4.pdf` (Jun 15)
- `/Users/elyas/Desktop/latent_recovery_old_versions/{latent_recovery_workshop_paper.pdf (Jun 10), _v2.pdf (Jun 11), _v3.pdf (Jun 11)}`
- `/Users/elyas/Desktop/Literature Briefing on Latent Recovery in Autoregressive Transformers.pdf` (+ copies in Downloads); `/Users/elyas/Downloads/main (2).pdf`, `main (3).pdf`

## Fallback: experiments visible in newest draft (extracted from main.pdf) — PAPER-LEVEL ONLY, not the codebase inventory
Cannot verify lock status, N per cell beyond what the paper prints, or anything the paper omits.
- **Main Qwen2.5-7B PrOntoQA sweep** (Table 3): expanded audited cells n=150; global falsehoods closure-valid 0.360–0.447 across injection positions, injection-dependence 0.233–0.420; one-hop falsehoods 0.633–0.740 closure-valid with much higher doubt; distractor/contradiction rows retain original auxiliary sweep (smaller n).
- **Distance-k dose-response** ("negate a fact derived k hops downstream" = the `perturbed_neghop2..5` line): doubt and closure-valid decline with k then plateau; 32B curve "shifts upward and compresses". Exists for 7B and 32B per paper.
- **Verification prompting** (`results_verify`): middle position, global falsehood inj-dep 0.371, closure-valid 0.343, doubt 0.029 (n=35); one-hop under same prompt doubt 0.423 (n=130). Single "available verification run".
- **EXPB certificate flip** (local certificate control): "strictly paired, artifact-locked"; certificates increase doubt (Δ=+0.370 doubt line) and reduce injection-dependence but do not guarantee full repair.
- **EXPC lexical-polarity control**: 2×2 locality-by-polarity; strict local-false-positive cell infeasible in grammar — bottlenecked to 7 problems / 84 validated rows; underpowered, "too noisy for formal inference"; stores a strict replay validator.
- **Task-structure comparison** (Table 7): PrOntoQA closure-valid 0.360 / inj-dep 0.373, n=150; **GSM8K** 0.088 / 0.877, n=57; **chained arithmetic** 0.020 / 0.990, n=99.
- **Goal-anchor ablations** (`anchor_qa` + `anchor_knockout`): boolean-query reframing = closure-valid "attenuated but persists" in "pilot data" (pilot → likely small n, unlocked); attention-mask goal knockout = collapses both arms, declared a negative/task-breaking result.
- **Hidden-state probes** (`probes`, ρfunc): strongly negative — JS divergence spikes post-injection but tracks surface style (benign paraphrase gives largest early divergence), not semantics.
- **Scale sweep**: Qwen2.5-1.5B / 7B (primary) / 32B (Table 8, ranges over early/middle/late positions, regenerated from "matched cross-model summary artifacts" — i.e., `matched_pert.py`/`matched_rho.json` pipeline exists).
- **Cross-lineage**: OLMo-2-7B-Instruct and Llama-3.1-8B-Instruct show the behavioral gradient (Llama repairs with little verbalized doubt — doubt/recovery dissociation); DeepSeek-R1-Distill-Qwen-7B isolates reasoning-specific training. **Mistral appears nowhere in this draft** despite `results_mistral` being named in the task spec — that sweep, plus `results_ctx`, `results_doubt`, `doubt_judge.json`, null controls, `shuffle_control.json`, `geometry.json`, and all lock statuses, are codebase-only and unrecoverable here.

## Recommendation to orchestrator
Re-dispatch this task with the project path variable actually resolved. The source tree is not under `/Users/elyas`; per the user's memory notes the likely location is the Stanford SNAP cluster (`skampere2`, requires `kinit eobbad@CS.STANFORD.EDU` + Stanford VPN — not reachable non-interactively from this session). Alternatively the tree may live on another machine that produced `main.pdf` on Jun 23 2026. `PROFESSOR_FEEDBACK.md` is likewise absent locally, so the "read it first" precondition also cannot be satisfied.