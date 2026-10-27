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
| correction, depths 1-5 | 1000 | 0.01 | 0.01 | 0.20 | 0.23 | 0.25 | 0.29 | 0.45 | 0 | 0.03 | 0.37 |
| clean (general data), depths 1-5 | 300 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0 |
| clean (general data), depths 1-5 | 1000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0 |
| correction, depth 1 only, seed 1 | 300 | 0.00 | 0.12 | 0.80 | 0.92 | 0.97 | 1.00 | 1.00 | 0 | 0 | 0 |
| correction, depths 1-3, seed 1 | 300 | 0.10 | 0.03 | 0.03 | 0.02 | 0.15 | 0.16 | 0.49 | 0.83 | 0.99 | 0.89 |

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

## Final round (Oct 26, 21:10; all queues finished, GPUs 5, 6, 3 released)
- Clean (general-data) control is now matched at both sizes: depths 1-5 at N=300 and N=1000 give propagation 1.00 at every depth, identical to the untuned base, 0 false flags. Two sizes, one seed each; going from 300 to 1000 clean programs moves nothing.
- Correction depths 1-5 at N=1000: 0.01 / 0.01 / 0.20 / 0.23 / 0.25 / 0.29 at d1-d6 and 0.45 at d8 (d6 and d8 never trained), versus 0.78 / 0.45 / 0.38 / 0.45 / 0.78 / 0.87 / 0.98 at N=300. So with re-verification data, quantity (300 to 1000) does move the unseen depths. Cost: false flags 0.37 at d5 (0.00 at d1, 0.03 at d3).
- Seed variance is large for mixed-depth data. Depths 1-3, N=300: seed 0 gives 0.13 / 0.38 / 0.63 at d1-d3 with false flags 0.00 / 0.14 / 0.08; seed 1 gives 0.10 / 0.03 / 0.03 with false flags 0.83 / 0.99 / 0.89. Seed 1 low propagation is mostly the model emitting a re-check on nearly every trace, correct or not, so it is not a real detector. Read propagation only together with flag rate. Depth-1-only is stable across seeds (d1 0.00, d3 0.75 vs 0.80).
- Reading, with limits: (1) general data does not move the wall (0 of 14 cells differ from base at N=300 and N=1000); (2) correction data fixes covered depths, and at N=1000 reaches depths 4-8 that were never trained, but d8 stays at 0.45 and the d5 false-flag rate is 0.37; (3) the cheapest depth-1 recipe does not extrapolate (d6 and d8 at 0.99-1.00 even at N=1000); (4) one seed per cell except two N=300 pairs, 100 per cell, about +-10 points; the seed-1 flip means mixed-depth N=300 numbers are unreliable. Next test if wanted: second seed at N=1000 depths 1-5, and a trace-level check separating real detection (flags only on perturbed traces) from blanket re-checking.
