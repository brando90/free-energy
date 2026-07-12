# VeriBench Cross-Entropy Baselines — B1: LM-SFT, B2: EBT (execution prompt)

**TLDR:** SFT a small LM on the VB-train split, tune only on val, report completion-only cross-entropy (+ ppl) on test; then attempt the same protocol with the official EBT codebase. Data comes from `experiments/08_vb_train_val_test/splits/`. This file is the runnable version of the prompt drafted in the 2026-06-09 meeting, with placeholders filled from the actual split.

**Provenance:** drafted from the [2026-06-09 meeting](https://fathom.video/share/GsBj2jyUo3Xo_pNaqWdvoZ4Ud4SXgdfv) via [this claude.ai chat](https://claude.ai/chat/95a8bcf4-5535-4d09-a414-d6ab99a4fd01); placeholders filled + numbers corrected against the real split on 2026-06-09 (this repo, branch `vb-ce-baselines-elyas`).

---

You are Claude Code on the SNAP cluster (iterate on **skampere2**; only use skampere3 for large runs). Working repo: **brando90/free-energy** — commit all configs, scripts, and results under `experiments/09_vb_ce_baselines/`. Goal: SFT on the VeriBench **train** split, tune only on **val**, report cross-entropy on **test**. Just the loss — no task eval needed yet.

## Filled-in parameters (were `<FILL>` in the draft)
- `VERIBENCH_SPLIT` = `experiments/08_vb_train_val_test/splits/{train,val,test}.jsonl` (this repo; self-contained JSONL with `py_code` + `lean_text` embedded — see that experiment's README for schema). Sizes: **train 707** (702 py-paired) / **val 87** (86) / **test 97** (96). The draft said "~200 examples" — that was wrong.
- `MODEL` = `Qwen/Qwen2.5-0.5B` (≤1.5B class)
- `MAX_SEQ_LEN` = **4096** (draft said 2048, but est. token length of py+lean pairs: median ≈2046, p95 ≈3986, max ≈10120 → 2048 truncates ~50% of examples, 4096 only ~4.3%). Report truncation counts.

## Protocol (applies to both parts)
- **Metric:** mean per-token NLL in nats (cross-entropy) + perplexity = exp(CE). The data has prompt→target structure (`py_code` → `lean_text`), so compute **completion-only CE** (mask prompt tokens) as the primary number and full-sequence CE as secondary. Skip the 7 `paired == false` rows (Lean-only CLRS tasks) for conditional CE.
- **Hygiene:** never touch test until the final measurement. Fix the tokenizer across all runs. Seed everything; run the best config with **3 seeds**, report mean ± std.
- **References to report alongside:** zero-shot CE of the *un*-finetuned `MODEL` on test (so the SFT delta is visible).
- **Logging:** per-run config (model, params, LR, epochs, batch, tokens seen), train/val CE curves, approximate training FLOPs (6·N·D). These logs get reused by the fair-comparison project — don't skip them.

## Part A — B1: LM + SFT (run this first)
1. Sanity check: overfit a single batch (CE → ~0) before any real run.
2. Full fine-tune `MODEL` on VB-train, bf16, grad accum as needed. Sweep LR ∈ {1e-5, 2e-5, 5e-5}, up to ~10 epochs with early stopping on **val** CE.
3. Final: best config × 3 seeds → **test CE + perplexity** (completion-only primary).
4. Deliverable: `results_b1.md` with the table {model, params, LR, epochs, tokens, val CE, test CE, ppl, zero-shot test CE} + exact repro commands.

## Part B — B2: EBT (after Part A works)
1. Clone the official EBT codebase ([Gladstone et al., arXiv:2507.02092](https://arxiv.org/abs/2507.02092)). Use the LM-style variant — next-token energies normalize at O(V), so test CE is computable.
2. Attempt SFT-style training on VB-train with the same split/tokenizer/protocol. **If the codebase doesn't support SFT-style training, fall back to the original paper's recipe and say so explicitly in the report** (per the meeting).
3. Match Part A's budget as closely as possible (params, tokens seen); log the differences you can't match.
4. Deliverable: `results_b2.md`, same table format, + one paragraph on any EBT-specific training instabilities (divergence/NaNs).
   - Note: a deleted toy EBT experiment exists in git history at `experiments/05_energy_based_transformer_baseline/` (commit `84f7441`) — context only; the official codebase is the source of truth for B2.

## Stop conditions
If the split is missing, a download fails, or EBT training diverges across all reasonable settings — stop, write up what happened rather than improvising the protocol.
