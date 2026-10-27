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

## Second seeds at N=1000 and trace-level real-detection test (AT-1523 / AT-1493, Oct 27)
Seeds 1-3 at N=1000 were trained and evaluated under AT-1493 (skampere1 GPUs 3 and 6, Oct 26 23:30 to Oct 27 04:16); AT-1523 adds the trace-level test. 100 programs per cell, about +-10 points.

Perturbed items corrected at depth 1 / 2 / 3 / 4 / 5 (original eval.py, greedy), and clean control_d5 flagged:

| arm | seed | d1 | d2 | d3 | d4 | d5 | flagged on clean d5 |
|---|---|---|---|---|---|---|---|
| depths 1-3, N=1000 | 0 | 1.00 | 0.98 | 0.87 | 0.76 | 0.62 | 0.30 |
| depths 1-3, N=1000 | 1 | 1.00 | 0.99 | 0.87 | 0.80 | 0.66 | 0.34 |
| depths 1-3, N=1000 | 2 | 1.00 | 0.99 | 0.82 | 0.76 | 0.67 | 0.43 |
| depths 1-3, N=1000 | 3 | 0.99 | 0.98 | 0.80 | 0.80 | 0.68 | 0.24 |
| depths 1-5, N=1000 | 0 | 0.99 | 0.99 | 0.80 | 0.77 | 0.75 | 0.37 |
| depths 1-5, N=1000 | 1 | 1.00 | 0.98 | 0.79 | 0.75 | 0.78 | 0.38 |
| depths 1-5, N=1000 | 2 | 1.00 | 0.98 | 0.79 | 0.71 | 0.75 | 0.32 |
| depths 1-5, N=1000 | 3 | 0.98 | 0.86 | 0.48 | 0.36 | 0.31 | 0.10 |

Seeds 0-2 agree for both depth sets; depths 1-5 seed 3 diverges (corrects less, flags clean input least). So the earlier "d12345 N=1000 corrects 0.25 at d5" (propagation 0.25, seed 0) holds for seeds 1-2 and not for seed 3.

### Trace-level test (`eval_trace.py`, `analyze.py`, `trace_results/`)
Question: is the model detecting the planted error, or re-checking every trace? Each of the 100 programs per depth (depths 1-5) is run twice from the same prompt, once with the planted wrong line 10 (perturbed) and once with the true line 10 (control). TPR = re-check on the perturbed trace, FPR = re-check on its own control, J = TPR - FPR (0 means a blanket re-checker or no detector, 1 means perfect). "Discriminative" = re-check on the perturbed trace and not on its control. Per-item outputs are in `trace_results/trace_per_item_all_runs.json`; per-run tables in `trace_results/analysis.txt`. Greedy decoding with vLLM LoRA batching can differ by 1-2 items from the original eval (e.g. d123 s0 clean d5 flagged 0.31 here, 0.30 there).

| arm (pooled over seeds) | depth | TPR | FPR | J | discriminative | control-only flags |
|---|---|---|---|---|---|---|
| depths 1-3, N=1000, seeds 0-3 | 1 | 1.00 | 0.00 | +1.00 | 399/400 | 0 |
| | 2 | 0.99 | 0.02 | +0.97 | 387/400 | 0 |
| | 3 | 0.84 | 0.05 | +0.79 | 319/400 | 3 |
| | 4 | 0.77 | 0.24 | +0.53 | 232/400 | 19 |
| | 5 | 0.65 | 0.33 | +0.32 | 174/400 | 46 |
| depths 1-5, N=1000, seeds 0-2 | 1 | 1.00 | 0.00 | +1.00 | 299/300 | 0 |
| | 2 | 0.99 | 0.03 | +0.96 | 288/300 | 0 |
| | 3 | 0.79 | 0.05 | +0.75 | 227/300 | 3 |
| | 4 | 0.74 | 0.24 | +0.51 | 171/300 | 19 |
| | 5 | 0.75 | 0.36 | +0.39 | 145/300 | 27 |
| depths 1-5, N=1000, seed 3 | 1 | 0.98 | 0.00 | +0.98 | 98/100 | 0 |
| | 3 | 0.49 | 0.04 | +0.45 | 46/100 | 1 |
| | 5 | 0.31 | 0.11 | +0.20 | 23/100 | 3 |
| depth 1 only, N=1000, seed 0 | 1 | 1.00 | 0.00 | +1.00 | 100/100 | 0 |
| | 3 | 0.20 | 0.00 | +0.20 | 20/100 | 0 |
| | 5 | 0.04 | 0.00 | +0.04 | 4/100 | 0 |
| base (untuned) and both clean-data twins (depths 1-3 and 1-5, N=1000) | 1-5 | 0.00 | 0.00 | 0.00 | 0 | 0 |

Reading, labeled a read: (1) Real detection exists. Through depth 3 the re-check is almost entirely discriminative (J 0.75-1.00, FPR at most 0.05), so at d1-d3 the corrections in the main table are real detections, not blanket re-checking. (2) It degrades with depth. At d4-d5 the FPR is 0.24-0.36 and J falls to 0.32-0.53 for mixed-depth data; at d5 depths 1-3 seeds 0-3 flag 46 of 400 controls without flagging the perturbed twin, so part of the d5 flag rate is noise-like, not detection. Depth 1-5 training keeps more detection at d5 (TPR 0.75, J 0.39) than depths 1-3 (TPR 0.65, J 0.32) but with the same FPR. (3) The blanket-re-check failure is not in these N=1000 runs; it was the N=300 seed-1 run (flag 0.83-0.99 on clean). (4) Seed 3 of depths 1-5 is the most selective (FPR 0.04-0.11) and also the weakest detector at depth (TPR 0.31 at d5). (5) Depth-1-only data has FPR 0 and J up to 1.00 at d1 but detection vanishes by d3. Limits: 100 programs per cell, one training seed per row (pooled over 3-4 seeds where shown), regex-based re-check flag from `grade`, one model (Qwen2.5-7B-Instruct).
