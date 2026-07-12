# B2 results — official EBT on VeriBench train/val/test (paper-recipe fallback)

**TLDR:** Official EBT (`alexiglad/EBT` @ `19420cb`, `ebt-xxs` + Qwen vocab = 131M params) trained **from scratch** on VB-train (the repo ships no pretrained LM checkpoints, so SFT-style training is not runnable as-is — this is the prompt's explicit fallback): best-val checkpoint (epoch 14 of 32) reaches **test completion-CE 2.7794 nats (ppl 16.11)**; full-seq 2.7960. Reference points from B1: pretrained Qwen2.5-0.5B zero-shot 1.1485, SFT 0.3124 ± 0.0008. The B1-vs-B2 gap is dominated by **pretrained-vs-from-scratch**, not architecture — do not read it as an EBT-vs-transformer comparison.

## Why the fallback (stated per protocol)

The official codebase has an `--execution_mode finetune`, but it requires `--finetuning_model_ckpt` and **no pretrained EBT LM checkpoints are released** (README's inference flow assumes you trained your own). SFT atop a pretrained EBT is therefore not runnable as-is. Per `PROMPT.md` Part B step 2, we fell back to the paper's pretrain recipe (`job_scripts/nlp/pretrain/ebt_s1.sh` hparams) on VB-train, with the same data formatting, tokenizer, and eval masking as B1.

## Result

| model | params | recipe | LR | eff. batch | steps (epochs) | selected ckpt | tokens seen @ sel. | val CE | **test CE** | test ppl | full-seq test CE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EBT-xxs (time_embed, mcmc 2) | 131M | from scratch | 1.2e-3 | 16 seqs | 1400 (31.9) | epoch 14 (val top-1) | ~19.4M | 2.8258 | **2.7794** | 16.11 | 2.7960 |

- `last.ckpt` (step 1400) scores test CE 2.7795 — identical to the val-selected ckpt; val plateaued from epoch 14 (mild overfitting regime, no divergence).
- Eval = `eval_b2.py`: token-weighted NLL from the final MCMC step's normalized next-token distribution, same prompt-masking as B1.

## Training stability (per protocol, the instabilities paragraph)

No divergence and no NaN/Inf events across 1400 steps (the EBT forward raises on NaN/Inf MCMC gradients — never triggered; learnable step-size α logged in `EBT/logs/console.log`). The only failure modes encountered were infrastructural: (i) **memory** — EBT's MCMC keeps `[B, S, |V|]` prob-dist tensors with a double-backprop graph; at context 4096 even bs=1 OOM'd a 143 GB H200, hence context 2048 + bs 1 × accum 16 (~30 GB steady); (ii) a **data bug in our own loader** (`prompt[-0:]` negative-zero slice keeping the whole prompt when a target is exactly max_len) — exactly 1 train doc triggered it and tripped EBT's RoPE shape assert; fixed in `8d9b7cc`, audited: 0 B1 items affected.

## Budget match vs B1 (differences we could not match, logged per protocol)

| dimension | B1 | B2 | note |
|---|---|---|---|
| init | pretrained Qwen2.5-0.5B | from scratch | the dominant difference |
| params | 494M | 131M (vocab-dominated: 2×151936×384 embeddings) | their canonical xxs size |
| tokenizer | Qwen2.5 | Qwen2.5 (same) | CE in nats/token directly comparable |
| context | 4096 (20/8 prompt/target-trunc of 702) | 2048 (165/174) | MCMC memory bound |
| tokens seen @ selection | 6.11M | ~19.4M (41.2M full run) | from-scratch needs more passes |
| FLOPs est. 6·N·D @ sel. | 1.81e16 | ~1.52e16 | comparable compute |
| seeds | 3 (std 0.0008) | 1 (their default 33) | rerun if a std is needed (~2 h/run) |
| eos target in CE | included | excluded (their pad==eos `ignore_index` convention) | ~1 token per ~2k-token doc |
| LR sched | cosine, wu 3% | their recipe (peak 1.2e-3, min_lr_scale 10, wu 100) | per ebt_s1.sh |

## Repro

```bash
# train (patches a pinned EBT clone, then runs their train_model.py on VB):
bash run_b2.sh 0 1400          # [GPU_ID] [MAX_STEPS]; ~2 h on 1x H200
# eval best-val ckpt with B1-comparable masking:
python eval_b2.py --ckpt <EBT>/logs/checkpoints/<run>/ --split test
```
Raw eval JSONs: `results/b2_eval_{test_best,val_best,test_last}.json` (includes n_params, token counts).
