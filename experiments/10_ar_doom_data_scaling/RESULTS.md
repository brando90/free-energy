# Results (filled as runs finish). Rate = held-out propagation (planted computed value carried into line 12), 100 programs per depth, greedy.

Arms: `d*` = correction data (planted error + re-check line, plus clean control half); `clean_*` = general-data control (correct traces only). Qwen2.5-7B-Instruct, LoRA r16, paper Table 4 hyperparameters, seed 0.

| arm | N | d1 | d2 | d3 | d4 | d5 | d6* | d8* | flag@d1 | flag@d3 | flag@d5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (untuned) | - | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0 |
| correction, depth 1 only | 100 | 0.39 | 0.65 | 0.89 | 0.97 | 1.00 | 1.00 | 1.00 | 0.31 | 0.07 | 0 |
| correction, depth 1 only | 300 | 0.00 | 0.12 | 0.75 | 0.91 | 0.97 | 1.00 | 0.99 | 0 | 0.02 | 0.01 |
| correction, depths 1-3 | 100 | 0.72 | 0.63 | 0.64 | 0.80 | 0.90 | 0.93 | 1.00 | 0.34 | 0.40 | 0.15 |
| correction, depths 1-3 | 300 | 0.13 | 0.38 | 0.63 | 0.69 | 0.91 | 0.95 | 0.95 | 0 | 0.14 | 0.08 |
| correction, depths 1-5 | 300 | pending (skampere1 GPU 5 queue) |
| correction, depths 1-5 | 1000 | pending (GPU 6 queue) |
| clean (general data), depths 1-5 | 300 | running on GPU 3 since 18:40 |
| clean (general data), depths 1-5 | 1000 | queued after the N=300 clean run |

\* depths 6 and 8 are never trained on. flag@dK = false-flag rate on correct traces (re-check line emitted when nothing is wrong).
Raw JSON: /lfs/skampere1/0/brando9/at1445/res/ (cluster, not committed).
