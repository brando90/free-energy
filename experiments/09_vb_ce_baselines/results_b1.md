# B1 results — LM + SFT on VeriBench train/val/test

**TLDR:** Qwen/Qwen2.5-0.5B full-SFT on VB-train (707 tasks): test completion-CE 0.3124 ± 0.0008 nats (ppl 1.37) over 3 seeds vs zero-shot 1.1485 nats (ppl 3.2). 

Completion-only CE (prompt tokens masked) in nats; ppl = exp(CE). Full protocol: `PROMPT.md`.

## LR sweep (seed 0, early stop on val CE)

| LR | epochs ran | best epoch | val CE | test CE* |
|---|---|---|---|---|
| 1e-05 | 5 | 2 | 0.3429 | 0.3164 |
| 2e-05 | 4 | 1 | 0.3362 | 0.3125 |
| 5e-05 | 4 | 1 | 0.3403 | 0.3100 |

*test CE shown for completeness; best LR was selected on val only.

## Final: best LR = 2e-05 × 3 seeds

| model | params | LR | sched | eff. batch | max epochs | best epoch (per seed) | tokens seen (mean) | val CE (mean±std) | **test CE (mean±std)** | test ppl | full-seq test CE | zero-shot test CE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | 494M | 2e-05 | cosine, wu 3% | 16 | 10 | 1,1,1 | 6.11e+06 | 0.3353±0.0006 | **0.3124±0.0008** | 1.37 | 0.6695 | 1.1485 |

FLOPs estimate (6·N·D): 1.81e+16 per final run. Truncation at max_seq_len 4096: 20 prompt-truncated, 8 target-truncated of 702 train examples; 5/1/1 unpaired rows skipped (train/val/test).

## Repro
```bash
python train_b1.py --mode zeroshot
python train_b1.py --mode overfit --lr 5e-5
for lr in 1e-5 2e-5 5e-5; do python train_b1.py --mode train --lr $lr --seed 0; done
for s in 0 1 2; do python train_b1.py --mode train --lr 2e-05 --seed $s; done
python train_b1.py --mode aggregate
```
