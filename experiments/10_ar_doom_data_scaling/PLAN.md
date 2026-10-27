# Does data fix "LMs do not re-verify computed values"? (data-scaling fine-tuning test)

Context: Elyas's paper ("Language Models Do Not Re-Verify Computed Values in Their Own Reasoning")
shows that 14 LMs continue from a planted wrong computed value in 88-100% of continuations. His only
data intervention (Sec. 5.7 / App. B) trains Qwen2.5-32B on 300 programs (600 examples): depth-1-only
data removes depth-1 propagation (94% -> 0%) but depth 3 is 30-48% and depth 5 is 97-98%; mixed
depth 1-3 data reaches further, even depth 5. Open question for "is AR doomed?": is that a
data-quantity problem, a data-coverage problem, or a wall?

## Questions
1. Quantity: at fixed depth coverage, how does propagation at held-out depths fall with N training programs (100 -> 1000)?
2. Coverage: depth-1 only vs depths 1-3 vs depths 1-5, at the same N. Does extrapolation to unseen depths 6 and 8 ever appear?
3. Cost: false-flag rate on correct traces (does the model start "re-checking" when nothing is wrong?) and whole-trace accuracy.
4. Variance: second seed at N=300.

## Setup (all recorded here so no setting is "unknown")
- Model: Qwen/Qwen2.5-7B-Instruct (bf16), LoRA r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down (40.4M params, 0.53%).
- Training = paper Table 4: AdamW lr 1e-4 (5 warmup updates, cosine to 0), 2 x 8 accumulation = 16 ex/update, 3 epochs, clip 1.0, grad checkpointing, max len 1536. Loss on target tokens + <|im_end|> only. PyTorch 2.10.0, Transformers 4.57.6, PEFT 0.21.2, vLLM 0.17.1. Single seed unless named.
- Data: `common.py` re-implements the App. B protocol (20-line programs, planted line 10, +/-10 or +/-20 plant, result-only trace lines, target = one "re-check" line then true trace; control example = true line 10, no check). Depth k = k arithmetic ops (k+1 operands on the k+1 lines before line 10). Training seeds >= 1,000,000; eval seeds 0..99 (disjoint). One 1-op worked example in the prompt. This is a re-implementation, not Elyas's exact generator (his training script is not in the repo).
- Eval: greedy decoding, max 500 new tokens, continuation from the trace prefix ending at the planted (or true) line 10; 100 held-out programs per depth for depths 1,2,3,4,5,6,8; controls at depths 1,3,5. Labels: `propagated` (line 12 uses the planted value, no re-check), `corrected` (re-check line or true value used), `other`.
- Hardware: skampere1 A100-80GB GPUs 5 and 6 (empty at launch), free; no paid API.
- Runs: depth sets {1}, {1,2,3}, {1..5} x N programs in {100, 300, 1000}, plus base model and a second seed at N=300.

## Files
common.py (generator/format/grader), build_data.py, train.py, eval.py, run_queue.sh; results in results/*.json; RESULTS.md.
