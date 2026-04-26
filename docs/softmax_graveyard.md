# The softmax-replacement graveyard

A synthesis of why softmax attention has survived nearly every proposed
replacement, written as the prerequisite for any architectural proposal
in this repo. Per the README's Definition of Done #5, every candidate
substrate must declare which of these failure modes it inherits, partially
inherits, or avoids — with an experiment to detect each.

The goal of this document is not to be exhaustive. It's to make the
question "why won't your thing fail like X did?" answerable in one
sentence per prior X.

## A. The graveyard

Each row names: the architecture, the replacement mechanism, the **specific**
failure mode established in the literature (not "underperforms"), the
workload that exposes it, and the canonical citation.

| Architecture | Replacement mechanism | Named failure mode | Workload exposing it | Citation |
| --- | --- | --- | --- | --- |
| Linear attention | Kernel feature map `φ(Q)·(φ(K)ᵀV)`; drops normalizer | Loss of "spikiness" — bounded feature maps cannot produce low-entropy attention from bounded scores | Long-context retrieval / associative recall | Katharopoulos et al. 2020 (arXiv:2006.16236) |
| Performer | Random Fourier features approximating `exp(qᵀk)` | Variance of random-feature estimator grows with sequence length; recall degrades | Long-context QA, copying | Choromanski et al. 2020 (arXiv:2009.14794) |
| Linformer | Low-rank projection of K, V along sequence | Information bottleneck independent of input — fixed `k` collapses on high-resolution / long sequence | High-res vision; long-doc LM | Wang et al. 2020 (arXiv:2006.04768) |
| Reformer | LSH-bucketed sparse attention | Reordering / hashing overhead + accuracy gap from missed cross-bucket dependencies | Anything where the right key sits in the wrong bucket | Kitaev et al. 2020 (arXiv:2001.04451) |
| Hedgehog (linear-attn diagnosis) | Learned feature maps mimicking softmax via spikiness + dot-product monotonicity | *Diagnoses* prior linear-attn failures (low-entropy + monotonicity missing); recovers ~99% of softmax quality but only by **adding back what softmax already had** | Causal LM perplexity, GLUE | Zhang et al. ICLR 2024 (arXiv:2402.04347) |
| Mamba / Mamba-2 | Selective state-space recurrence | Fixed-size hidden state cannot copy strings of arbitrary length — fundamentally bounded by state dimension | Copy-from-context, in-context retrieval | Gu & Dao 2023 (arXiv:2312.00752); Dao & Gu 2024 (arXiv:2405.21060) |
| Hyena | Long implicit convolution + gating | Associative-recall gap vs attention; convolutional bias can't represent content-addressed lookup | Synthetic associative recall, long-context QA | Poli et al. 2023 (arXiv:2302.10866) |
| RWKV | Linear-time RNN with attention-style channel mixing | Information loss with depth; in-context learning gap vs same-FLOPs transformer | Few-shot eval, long-context recall | Peng et al. 2023 (arXiv:2305.13048) |
| RetNet | Retention = linear attn with exponential decay | Decay erases distant tokens; explicit accuracy/recall trade-off vs softmax | Long-context tasks beyond decay horizon | Sun et al. 2023 (arXiv:2307.08621) |
| Repeat After Me (post-mortem) | Theoretical + empirical on GSSMs vs transformers | Two-layer transformer can copy strings of exponential length; any fixed-state SSM cannot | Synthetic copy + retrieval | Jelassi et al. 2024 (arXiv:2402.01032) |
| Were RNNs All We Needed? | minGRU / minLSTM (parallel-trainable simplified RNNs) | Competitive on small / decision-making benchmarks but does not close the in-context-learning / copy gap to attention | Same as the SSM gap above | Feng et al. 2024 (arXiv:2410.01201) |
| Bridging the Divide / InLine | Adds injectivity + local modeling to linear attention | **Proves linear attention is not injective** — distinct queries can map to identical weight rows; recovers softmax-level quality only after restoring competition + locality | Vision classification + segmentation at high res | Han et al. NeurIPS 2024 (arXiv:2412.06590) |
| On the Expressiveness of Softmax Attention | Theoretical — derives softmax attention as a recurrent operation | Linear attention is the *first-order* approximation of the softmax numerator; higher-order terms (the part softmax keeps) are what's expressive | Ablation of softmax components | Mongaras & Larson 2025 (arXiv:2507.23632) |
| Softmax Linear Attention (SLA) | Linear attention per-token + softmax across heads | Demonstrates the load-bearing softmax property is **competition** — moved from token level to head level, the rest can be linear | LM perplexity, long-context retrieval | (arXiv:2602.01744, 2026) |
| DeepSeek-V4 (production-scale) | Hybrid Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA) interleaved | Pure softmax too expensive at 1M-token context; pure linear / SSM doesn't ship — production must **keep softmax in the loop** at multiple compression levels | 1M-token agentic LM | DeepSeek-AI 2026 (V4-Pro release) |

### Reading the table

The pattern across rows is the same: a replacement removes one or both
of (a) the global normalizer that produces competition or (b) the
injective non-saturating mapping from scores to weights, and the
failure shows up wherever the workload requires *content-addressed
retrieval over arbitrarily long context*. The most successful "post-softmax"
proposals (Hedgehog, InLine, SLA, V4) work by **putting back what was
removed** at some scale — token level, head level, or in a sparse hybrid.

## B. Why softmax wins — five properties nothing else replicates simultaneously

Each property is named, briefly motivated, and tied to the paper that
established it as load-bearing.

1. **Global competition / winner-take-all.** Softmax's `exp` + normalizer
   forces attention weights into a probability simplex; raising β makes
   one key dominate. No bounded feature map of Q, K reproduces this from
   the same bounded inputs. *Established in:* Hedgehog (Zhang 2024); SLA
   (2026) — the latter shows just adding head-level softmax on top of
   linear token attention recovers the bulk of the gap.

2. **Injective query → weight-vector map.** Distinct query vectors yield
   distinct attention distributions over keys. Linear attention is
   provably non-injective: distinct queries can map to identical weight
   rows after row-normalization, causing semantic confusion. *Established
   in:* Bridging the Divide (Han et al. 2024).

3. **Spiky / low-entropy attention from bounded scores.** Softmax can
   produce near-degenerate distributions purely by raising β; linear /
   feature-map attention's entropy is lower-bounded by feature-map
   geometry. Hedgehog identifies this as one of the two missing
   properties of prior linear attentions. *Established in:* Hedgehog
   (Zhang 2024).

4. **Streaming-stable numerical structure (online softmax).** The
   `log-sum-exp` formulation with a running max correction (online
   softmax) lets softmax attention fuse into a single streaming kernel
   without ever materializing the `N×N` matrix — this is the entire basis
   of FlashAttention. Most exp-free proposals lose this and trade quadratic
   memory for quadratic compute or vice versa. *Established in:* Milakov &
   Gimelshein 2018 (arXiv:1805.02867); Dao et al. 2022 — FlashAttention
   (arXiv:2205.14135).

5. **Hopfield-energy interpretation with exponential capacity.** Softmax
   attention **is** the gradient step on a specific continuous Hopfield
   energy with exponential storage capacity in pattern dimension. The
   Hopfield reframe (next section) is what makes "EBMs vs softmax" the
   wrong frame: softmax is already an EBM update on a particular energy.
   *Established in:* Ramsauer et al. 2020 (arXiv:2008.02217); see
   `docs/hopfield_equivalence.md`.

A useful sixth diagnostic — not a "win" property but a failure-mode
detector — is that pure-attention stacks without skips/MLPs lose rank
doubly-exponentially in depth (Dong, Cordonnier, Loukas 2021,
arXiv:2103.03404). Any replacement must pass this signal-propagation
test, regardless of how it scores on properties 1–5.

## C. Hopfield reframe

Ramsauer et al. (2020) proved that softmax attention is the gradient
step on the continuous modern Hopfield network energy

    E(ξ) = -lse(β · X ξ) + ½ ξᵀ ξ + const.

where `X` plays the role of stored patterns (think keys), `ξ` is the
query state, and one update step `ξ ← X · softmax(β X ξ)` is exactly
softmax attention with `Q = ξ`, `K = V = X`. The capacity of this energy
is exponential in pattern dimension.

The consequence for this project is sharper than "we want EBMs instead
of softmax":

> Softmax attention is already an EBM update. The actual research
> question is **(a)** is there a different energy whose gradient is more
> useful, **(b)** can we fit the same energy without paying for `Z` at
> every forward pass, or **(c)** can we run multiple gradient steps
> (energy descent) at inference and beat fixed-compute training? If
> none of these is yes, the project is a different parameterization of
> the same thing.

This reframe also explains why the post-mortem papers (Repeat After Me,
Bridging the Divide, On the Expressiveness…, SLA) keep arriving at
"softmax-with-X" as the answer: the energy `E` *is* doing work, and
removing `lse` or normalization removes properties 1–3 above directly.

See `docs/hopfield_equivalence.md` for the careful statement, including
where this is a literal equivalence vs. where it becomes metaphor.

## D. What this means for our project

For each of the five properties from §B, one sentence on whether an
EBM-style / partition-function-free attention plausibly preserves it,
breaks it, or is unclear. Default to "unclear" if unsure. Per Definition
of Done #5, every later proposal must refine these from "unclear" to a
specific claim with a detection experiment.

1. **Global competition.** *Likely breaks* by default — most proposals
   that remove the normalizer also remove competition. SLA's
   "softmax-at-the-head-level" trick suggests competition can be
   recovered cheaply at *some* level of granularity; whether that's
   enough for the workloads we care about is **unclear**.

2. **Injective query → weights.** *Likely breaks* unless we explicitly
   design for injectivity (cf. InLine). Detection: run the per-query
   weight-row distinctness probe from Han et al. on candidate substrate.

3. **Spiky / low-entropy attention.** *Likely breaks* with kernel
   feature maps; *unclear* for energy-descent inference (multiple steps
   may sharpen). Detection: attention-entropy distribution vs depth, vs
   softmax baseline at matched accuracy.

4. **Streaming-stable kernel.** *Unclear.* Some exp-free substrates
   (sigmoid, softplus) are themselves streaming-stable; energy-descent
   inference is *not* obviously fusable into a single FlashAttention-style
   kernel. This is the load-bearing question for issue #4 (hardware
   co-design). Detection: write the streaming kernel; see Pillar 3.

5. **Hopfield-energy interpretation.** *Preserved by construction* if
   the substrate is defined as gradient flow on a stated energy. The
   research surface is then "which energy" — see candidate-energy catalog
   in `docs/hopfield_equivalence.md`.

The honest summary: **default expectation is that any substrate
designed without explicit awareness of properties 1–4 will inherit 1–3
graveyard failure modes simultaneously.** The bar to clear is to name
which property is being preserved by what mechanism, before running any
benchmark.

## What's intentionally not in this document

- Per-paper detailed reading notes — these belong in `papers/notes/`
  (see issue #10).
- The Lean side quest's formalization target — see issue #9.
- Hardware analysis — see issue #4 and forthcoming `HARDWARE.md`.
- Falsification criteria for the four thesis claims — see `MOTIVATION.md`
  (forthcoming, issue #17).
