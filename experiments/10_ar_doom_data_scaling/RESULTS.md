# Results (filled as runs finish). Rate = held-out propagation (planted computed value carried into line 12), 100 programs per depth, greedy.

Arms: `d*` = correction data (planted error + re-check line, plus clean control half); `clean_*` = general-data control (correct traces only). Qwen2.5-7B-Instruct, LoRA r16, paper Table 4 hyperparameters, seed 0.

| arm | N | d1 | d2 | d3 | d4 | d5 | d6* | d8* | flag@d1 | flag@d3 | flag@d5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (untuned) | - | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0 |
| correction, depth 1 only | 100 | 0.39 | 0.65 | 0.89 | 0.97 | 1.00 | 1.00 | 1.00 | 0.31 | 0.07 | 0 |
| correction, depth 1 only | 300 | 0.00 | 0.12 | 0.75 | 0.91 | 0.97 | 1.00 | 0.99 | 0 | 0.02 | 0.01 |
| correction, depths 1-3 | 100 | 0.72 | 0.63 | 0.64 | 0.80 | 0.90 | 0.93 | 1.00 | 0.34 | 0.40 | 0.15 |
| correction, depths 1-3 | 300 | 0.13 | 0.38 | 0.63 | 0.69 | 0.91 | 0.95 | 0.95 | 0 | 0.14 | 0.08 |
| correction, depths 1-3 | 1000 | 0.00 | 0.02 | 0.13 | 0.24 | 0.38 | 0.40 | 0.47 | 0 | 0.05 | 0.30 |
| correction, depth 1 only | 1000 | 0.00 | 0.13 | 0.80 | 0.87 | 0.96 | 0.99 | 0.98 | 0 | 0 | 0 |
| correction, depths 1-5 | 300 | 0.78 | 0.45 | 0.38 | 0.45 | 0.78 | 0.87 | 0.98 | 0.11 | 0.54 | 0.21 |
| correction, depths 1-5 | 1000 | pending (GPU 6 queue) |
| clean (general data), depths 1-5 | 300 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0 |
| clean (general data), depths 1-5 | 1000 | queued after the N=300 clean run |

\* depths 6 and 8 are never trained on. flag@dK = false-flag rate on correct traces (re-check line emitted when nothing is wrong).
Raw JSON: /lfs/skampere1/0/brando9/at1445/res/ (cluster, not committed).

Reading so far (one seed, 100 per cell, about +-10 points noise):
- More correction data fixes the depths it covers: depth-1-only goes from 0.39 to 0.00 propagation at d1 and 0.65 to 0.12 at d2 between N=100 and N=300.
- It does not fix depths it never covers: d6 and d8 stay at 0.93 to 1.00 propagation in every run, d4-d5 stay at 0.69 or above.
- Adding depths 2 and 3 widens coverage (d4 0.69, d6 0.95) but at N=300 costs d1 (0.13 vs 0.00) and raises false flags on clean traces (0.14 at d3).

## Self-correction arm vs general data (AT-1499, Oct 26 19:40)
- Matched pair (depths 1-5, N=300, same example count): clean data gives propagation 1.00 at every depth, identical to the untuned base, 0 false flags, training loss 0.0000. Correction data gives 0.78 / 0.45 / 0.38 / 0.45 / 0.78 / 0.87 at depths 1-6 (0.98 at d8), with false flags 0.11 / 0.54 / 0.21 at d1 / d3 / d5.
- Stronger result with no clean counterpart yet: correction depths 1-3 at N=1000 gives 0.00 / 0.02 / 0.13 at d1-d3 and 0.24 / 0.38 / 0.40 / 0.47 at d4 / d5 / d6 / d8 (d4, d6, d8 never trained), false flags 0.05 at d3 and 0.30 at d5. Clean N=1000 is still running (update 290/375 at 19:35).
- Reading, with limits: at N=300 general data does not move the wall and correction data does, at the cost of false flags on correct traces. The "general data does nothing" claim rests on one clean run, one seed, until clean N=1000 finishes. One seed, 100 per cell, about +-10 points. Consistent with the conjecture at the top of PLAN.md (re-verification data is the lever).
