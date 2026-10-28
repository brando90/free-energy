# Cluster-bootstrap reanalysis of recompute-wall headline cells

B=2000, seed=20260728, cluster=program_id, percentile CI. Metric = absorbed.
Greedy-only arms (widen/bridge/claude_pooled): 1 rollout per program, so
cluster bootstrap == row bootstrap there; it differs from Wilson only in method.

| cell | n_roll | n_prog | rate | Wilson95 | clusterBoot95 | greedy | sampled | prog@0% | prog@100% | prog mid |
|---|---|---|---|---|---|---|---|---|---|---|
| bridge/gpt-3.5-turbo-instruct/adjacent_contradiction | 68 | 68 | 0.1471 | [0.0819, 0.2500] | [0.0735, 0.2353] | 0.1471 | - | 58 | 10 | 0 |
| bridge/gpt-3.5-turbo-instruct/deep_kc5 | 59 | 59 | 1.0000 | [0.9389, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 59 | 0 |
| bridge/gpt-3.5-turbo-instruct/onehop_kc1 | 58 | 58 | 1.0000 | [0.9379, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 58 | 0 |
| bridge/gpt-4.1/adjacent_contradiction | 70 | 70 | 0.0143 | [0.0025, 0.0766] | [0.0000, 0.0429] | 0.0143 | - | 69 | 1 | 0 |
| bridge/gpt-4.1/deep_kc5 | 60 | 60 | 1.0000 | [0.9398, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 60 | 0 |
| bridge/gpt-4.1/onehop_kc1 | 60 | 60 | 1.0000 | [0.9398, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 60 | 0 |
| bridge/gpt-4o/adjacent_contradiction | 70 | 70 | 0.0143 | [0.0025, 0.0766] | [0.0000, 0.0429] | 0.0143 | - | 69 | 1 | 0 |
| bridge/gpt-4o/deep_kc5 | 60 | 60 | 0.9500 | [0.8630, 0.9829] | [0.8833, 1.0000] | 0.9500 | - | 3 | 57 | 0 |
| bridge/gpt-4o/onehop_kc1 | 60 | 60 | 1.0000 | [0.9398, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 60 | 0 |
| bridge/gpt-5.1/adjacent_contradiction | 70 | 70 | 0.0857 | [0.0399, 0.1747] | [0.0286, 0.1571] | 0.0857 | - | 64 | 6 | 0 |
| bridge/gpt-5.1/deep_kc5 | 60 | 60 | 1.0000 | [0.9398, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 60 | 0 |
| bridge/gpt-5.1/onehop_kc1 | 60 | 60 | 1.0000 | [0.9398, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 60 | 0 |
| claude_pooled/claude-haiku-4-5/adjacent_contradiction | 190 | 190 | 0.0000 | [0.0000, 0.0198] | [0.0000, 0.0000] | 0.0000 | - | 190 | 0 | 0 |
| claude_pooled/claude-haiku-4-5/deep_kc5 | 180 | 180 | 1.0000 | [0.9791, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 180 | 0 |
| claude_pooled/claude-haiku-4-5/onehop_kc1 | 180 | 180 | 1.0000 | [0.9791, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 180 | 0 |
| claude_pooled/claude-opus-4-8/adjacent_contradiction | 190 | 190 | 0.0000 | [0.0000, 0.0198] | [0.0000, 0.0000] | 0.0000 | - | 190 | 0 | 0 |
| claude_pooled/claude-opus-4-8/deep_kc5 | 180 | 180 | 0.9889 | [0.9604, 0.9969] | [0.9722, 1.0000] | 0.9889 | - | 2 | 178 | 0 |
| claude_pooled/claude-opus-4-8/onehop_kc1 | 180 | 180 | 1.0000 | [0.9791, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 180 | 0 |
| claude_pooled/claude-sonnet-4-5/adjacent_contradiction | 190 | 190 | 0.0000 | [0.0000, 0.0198] | [0.0000, 0.0000] | 0.0000 | - | 190 | 0 | 0 |
| claude_pooled/claude-sonnet-4-5/deep_kc5 | 180 | 180 | 0.9889 | [0.9604, 0.9969] | [0.9722, 1.0000] | 0.9889 | - | 2 | 178 | 0 |
| claude_pooled/claude-sonnet-4-5/onehop_kc1 | 180 | 180 | 0.9722 | [0.9366, 0.9881] | [0.9444, 0.9944] | 0.9722 | - | 5 | 175 | 0 |
| claude_pooled/claude-sonnet-5/adjacent_contradiction | 190 | 190 | 0.0105 | [0.0029, 0.0376] | [0.0000, 0.0263] | 0.0105 | - | 188 | 2 | 0 |
| claude_pooled/claude-sonnet-5/deep_kc5 | 180 | 180 | 1.0000 | [0.9791, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 180 | 0 |
| claude_pooled/claude-sonnet-5/onehop_kc1 | 178 | 178 | 1.0000 | [0.9789, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 178 | 0 |
| e11/haiku/k0_bare | 240 | 60 | 0.0125 | [0.0043, 0.0361] | [0.0000, 0.0292] | 0.0000 | 0.0167 | 57 | 0 | 3 |
| e11/haiku/k1_bare | 240 | 60 | 1.0000 | [0.9842, 1.0000] | [1.0000, 1.0000] | 1.0000 | 1.0000 | 0 | 60 | 0 |
| e11/llama8b/k0_bare | 1350 | 150 | 0.5963 | [0.5699, 0.6222] | [0.5467, 0.6430] | 0.6200 | 0.5933 | 6 | 20 | 124 |
| e11/llama8b/k1_bare | 1350 | 150 | 0.9859 | [0.9781, 0.9910] | [0.9719, 0.9956] | 0.9933 | 0.9850 | 0 | 140 | 10 |
| e11/qwen7b/k0_bare | 1350 | 150 | 0.7126 | [0.6879, 0.7361] | [0.6674, 0.7541] | 0.8333 | 0.6975 | 4 | 38 | 108 |
| e11/qwen7b/k1_bare | 1350 | 150 | 0.7844 | [0.7617, 0.8056] | [0.7348, 0.8326] | 0.8467 | 0.7767 | 6 | 75 | 69 |
| widen/llama8b/adjacent_contradiction | 150 | 150 | 0.5800 | [0.5000, 0.6560] | [0.5067, 0.6533] | 0.5800 | - | 63 | 87 | 0 |
| widen/llama8b/deep_kc5 | 64 | 64 | 1.0000 | [0.9434, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 64 | 0 |
| widen/llama8b/onehop_kc1 | 150 | 150 | 1.0000 | [0.9750, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 150 | 0 |
| widen/olmo7b/adjacent_contradiction | 150 | 150 | 0.6533 | [0.5742, 0.7248] | [0.5733, 0.7267] | 0.6533 | - | 52 | 98 | 0 |
| widen/olmo7b/deep_kc5 | 15 | 15 | 0.8667 | [0.6212, 0.9626] | [0.6667, 1.0000] | 0.8667 | - | 2 | 13 | 0 |
| widen/olmo7b/onehop_kc1 | 66 | 66 | 0.7273 | [0.6096, 0.8200] | [0.6212, 0.8333] | 0.7273 | - | 18 | 48 | 0 |
| widen/qwen1p5b/adjacent_contradiction | 61 | 61 | 0.5410 | [0.4172, 0.6599] | [0.4262, 0.6721] | 0.5410 | - | 28 | 33 | 0 |
| widen/qwen1p5b/deep_kc5 | 1 | 1 | 1.0000 | [0.2065, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 1 | 0 |
| widen/qwen1p5b/onehop_kc1 | 30 | 30 | 0.5000 | [0.3315, 0.6685] | [0.3333, 0.6667] | 0.5000 | - | 15 | 15 | 0 |
| widen/qwen32b/adjacent_contradiction | 150 | 150 | 0.0600 | [0.0319, 0.1101] | [0.0267, 0.1000] | 0.0600 | - | 141 | 9 | 0 |
| widen/qwen32b/deep_kc5 | 150 | 150 | 1.0000 | [0.9750, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 150 | 0 |
| widen/qwen32b/onehop_kc1 | 150 | 150 | 1.0000 | [0.9750, 1.0000] | [1.0000, 1.0000] | 1.0000 | - | 0 | 150 | 0 |
| widen/qwen7b/adjacent_contradiction | 150 | 150 | 0.7267 | [0.6504, 0.7917] | [0.6533, 0.7933] | 0.7267 | - | 41 | 109 | 0 |
| widen/qwen7b/deep_kc5 | 105 | 105 | 0.9905 | [0.9480, 0.9983] | [0.9714, 1.0000] | 0.9905 | - | 1 | 104 | 0 |
| widen/qwen7b/onehop_kc1 | 150 | 150 | 0.8800 | [0.8183, 0.9227] | [0.8267, 0.9333] | 0.8800 | - | 18 | 132 | 0 |

## Cross-checks vs stored paper summaries
- widen/qwen1p5b/adjacent_contradiction: recomputed k/n/rate=[33, 61, 0.541] paper=[33, 61, 0.541] match=True
- widen/qwen1p5b/onehop_kc1: recomputed k/n/rate=[15, 30, 0.5] paper=[15, 30, 0.5] match=True
- widen/qwen1p5b/deep_kc5: recomputed k/n/rate=[1, 1, 1.0] paper=[1, 1, 1.0] match=True
- widen/qwen7b/adjacent_contradiction: recomputed k/n/rate=[109, 150, 0.7267] paper=[109, 150, 0.7267] match=True
- widen/qwen7b/onehop_kc1: recomputed k/n/rate=[132, 150, 0.88] paper=[132, 150, 0.88] match=True
- widen/qwen7b/deep_kc5: recomputed k/n/rate=[104, 105, 0.9905] paper=[104, 105, 0.9905] match=True
- widen/qwen32b/adjacent_contradiction: recomputed k/n/rate=[9, 150, 0.06] paper=[9, 150, 0.06] match=True
- widen/qwen32b/onehop_kc1: recomputed k/n/rate=[150, 150, 1.0] paper=[150, 150, 1.0] match=True
- widen/qwen32b/deep_kc5: recomputed k/n/rate=[150, 150, 1.0] paper=[150, 150, 1.0] match=True
- widen/olmo7b/adjacent_contradiction: recomputed k/n/rate=[98, 150, 0.6533] paper=[98, 150, 0.6533] match=True
- widen/olmo7b/onehop_kc1: recomputed k/n/rate=[48, 66, 0.7273] paper=[48, 66, 0.7273] match=True
- widen/olmo7b/deep_kc5: recomputed k/n/rate=[13, 15, 0.8667] paper=[13, 15, 0.8667] match=True
- widen/llama8b/adjacent_contradiction: recomputed k/n/rate=[87, 150, 0.58] paper=[87, 150, 0.58] match=True
- widen/llama8b/onehop_kc1: recomputed k/n/rate=[150, 150, 1.0] paper=[150, 150, 1.0] match=True
- widen/llama8b/deep_kc5: recomputed k/n/rate=[64, 64, 1.0] paper=[64, 64, 1.0] match=True
- bridge/gpt-3.5-turbo-instruct/adjacent_contradiction: recomputed k/n/rate=[10, 68, 0.1471] paper={'k': 10, 'n': 68, 'rate': 0.1471, 'wilson95': [0.0819, 0.25]} match=None
- bridge/gpt-3.5-turbo-instruct/onehop_kc1: recomputed k/n/rate=[58, 58, 1.0] paper={'k': 58, 'n': 58, 'rate': 1.0, 'wilson95': [0.9379, 1.0]} match=None
- bridge/gpt-3.5-turbo-instruct/deep_kc5: recomputed k/n/rate=[59, 59, 1.0] paper={'k': 59, 'n': 59, 'rate': 1.0, 'wilson95': [0.9389, 1.0]} match=None
- bridge/gpt-4o/adjacent_contradiction: recomputed k/n/rate=[1, 70, 0.0143] paper={'k': 1, 'n': 70, 'rate': 0.0143, 'wilson95': [0.0025, 0.0766]} match=None
- bridge/gpt-4o/onehop_kc1: recomputed k/n/rate=[60, 60, 1.0] paper={'k': 60, 'n': 60, 'rate': 1.0, 'wilson95': [0.9398, 1.0]} match=None
- bridge/gpt-4o/deep_kc5: recomputed k/n/rate=[57, 60, 0.95] paper={'k': 57, 'n': 60, 'rate': 0.95, 'wilson95': [0.863, 0.9829]} match=None
- bridge/gpt-4.1/adjacent_contradiction: recomputed k/n/rate=[1, 70, 0.0143] paper={'k': 1, 'n': 70, 'rate': 0.0143, 'wilson95': [0.0025, 0.0766]} match=None
- bridge/gpt-4.1/onehop_kc1: recomputed k/n/rate=[60, 60, 1.0] paper={'k': 60, 'n': 60, 'rate': 1.0, 'wilson95': [0.9398, 1.0]} match=None
- bridge/gpt-4.1/deep_kc5: recomputed k/n/rate=[60, 60, 1.0] paper={'k': 60, 'n': 60, 'rate': 1.0, 'wilson95': [0.9398, 1.0]} match=None
- bridge/gpt-5.1/adjacent_contradiction: recomputed k/n/rate=[6, 70, 0.0857] paper={'k': 6, 'n': 70, 'rate': 0.0857, 'wilson95': [0.0399, 0.1747]} match=None
- bridge/gpt-5.1/onehop_kc1: recomputed k/n/rate=[60, 60, 1.0] paper={'k': 60, 'n': 60, 'rate': 1.0, 'wilson95': [0.9398, 1.0]} match=None
- bridge/gpt-5.1/deep_kc5: recomputed k/n/rate=[60, 60, 1.0] paper={'k': 60, 'n': 60, 'rate': 1.0, 'wilson95': [0.9398, 1.0]} match=None
- claude_pooled/claude-opus-4-8/adjacent_contradiction: recomputed k/n/rate=[0, 190, 0.0] paper=[0, 190, 0.0] match=True
- claude_pooled/claude-opus-4-8/onehop_kc1: recomputed k/n/rate=[180, 180, 1.0] paper=[180, 180, 1.0] match=True
- claude_pooled/claude-opus-4-8/deep_kc5: recomputed k/n/rate=[178, 180, 0.9889] paper=[178, 180, 0.9889] match=True
- claude_pooled/claude-sonnet-5/adjacent_contradiction: recomputed k/n/rate=[2, 190, 0.0105] paper=[2, 190, 0.0105] match=True
- claude_pooled/claude-sonnet-5/onehop_kc1: recomputed k/n/rate=[178, 178, 1.0] paper=[178, 178, 1.0] match=True
- claude_pooled/claude-sonnet-5/deep_kc5: recomputed k/n/rate=[180, 180, 1.0] paper=[180, 180, 1.0] match=True
- claude_pooled/claude-haiku-4-5/adjacent_contradiction: recomputed k/n/rate=[0, 190, 0.0] paper=[0, 190, 0.0] match=True
- claude_pooled/claude-haiku-4-5/onehop_kc1: recomputed k/n/rate=[180, 180, 1.0] paper=[180, 180, 1.0] match=True
- claude_pooled/claude-haiku-4-5/deep_kc5: recomputed k/n/rate=[180, 180, 1.0] paper=[180, 180, 1.0] match=True
- claude_pooled/claude-sonnet-4-5/adjacent_contradiction: recomputed k/n/rate=[0, 190, 0.0] paper=[0, 190, 0.0] match=True
- claude_pooled/claude-sonnet-4-5/onehop_kc1: recomputed k/n/rate=[175, 180, 0.9722] paper=[175, 180, 0.9722] match=True
- claude_pooled/claude-sonnet-4-5/deep_kc5: recomputed k/n/rate=[178, 180, 0.9889] paper=[178, 180, 0.9889] match=True
- e11/haiku/k0_bare: recomputed k/n/rate=[3, 240, 0.0125] paper=[3, 240, 0.0125] match=True
- e11/haiku/k1_bare: recomputed k/n/rate=[240, 240, 1.0] paper=[240, 240, 1.0] match=True
- e11/llama8b/k0_bare: recomputed k/n/rate=[805, 1350, 0.5963] paper=[805, 1350, 0.5963] match=True
- e11/llama8b/k1_bare: recomputed k/n/rate=[1331, 1350, 0.9859] paper=[1331, 1350, 0.9859] match=True
- e11/qwen7b/k0_bare: recomputed k/n/rate=[962, 1350, 0.7126] paper=[962, 1350, 0.7126] match=True
- e11/qwen7b/k1_bare: recomputed k/n/rate=[1059, 1350, 0.7844] paper=[1059, 1350, 0.7844] match=True
