# vLLM Generation Backend — Port Report

Status: COMPLETE — all measurements taken on skampere2, 2026-07-02.
Headline: prompt rendering byte-exact; greedy replay 7/20 exact (9/20
validator-class flips — kernel numerics, see §4 and caveat 1); throughput
2073.6 gen tok/s on one H200 (~35-69x over the HF batch-1 baseline).
Location: `improvement_plan/tooling/` (this directory). Nothing outside this
directory was modified; `src/`, `results/`, `data/` were read-only inputs.

## 1. What this is

`vllm_gen.py` is an offline vLLM wrapper that reproduces the exact generation
semantics of the HF-transformers harness (`src/common.py` `make_prompt_ids` +
`greedy`, as used by the EXPC generate stage), but batched:

- chat template applied to one user message
  (`INSTR + FEWSHOT + "Q: {question} Prove: {target}\nA:"`),
  `add_generation_prompt=True`;
- assistant turn pre-filled with
  `" " + " ".join(prefix_steps + [injected_statement])`, tokenized with
  `add_special_tokens=False` and concatenated at the TOKEN-ID level
  (identical to `common.make_prompt_ids`); vLLM receives `prompt_token_ids`
  (`TokensPrompt`), so the prompt is byte-for-byte the HF prompt by
  construction;
- greedy (`temperature=0`), `max_tokens=192` (== HF `max_new_tokens=192`),
  stop on the union of tokenizer eos + `generation_config` eos ids
  (mirrors HF `model.generate` default stopping), decode with
  `skip_special_tokens=True`.

## 2. Install (pinned)

- venv: `improvement_plan/tooling/.venv-vllm` (python 3.11.15, created from
  the main free-energy venv's interpreter via `python -m venv`; the main venv
  itself was NOT touched)
- vllm: **0.24.0** (plain `pip install vllm`, no index overrides)
- torch: **2.11.0+cu130**, triton 3.6.0, flashinfer-python 0.6.12
- transformers: **5.12.1**, tokenizers 0.22.2 (main venv has transformers
  5.10.2 — chat template comes from the model repo, rendering verified
  identical; see §3)
- Driver 580.82.07 / CUDA 13.0 on 8x H200 (sm90) — cu130 wheels load and run
  with no driver issues.
- Full `pip freeze` in `requirements-vllm.lock` (187 packages).

## 3. Prompt-rendering fidelity (byte-for-byte)

Verified on EXPC run `c2db53046d9712fd837c73bb`
(`--verify-prompt`): token-id path (HF `make_prompt_ids` replica) vs
string path (`apply_chat_template(tokenize=False, add_generation_prompt=True)`
+ manual prefix concat):

- 465 tokens both ways, `decoded_ids == string` True,
  `re-encoded string == ids` True, first divergence: none.

The generation path always uses `prompt_token_ids`, so even if a future
tokenizer merged differently at the assistant-prefill boundary, the prompt
fed to vLLM would still be the exact HF token sequence.

## 4. Replay fidelity vs stored EXPC generations

Setup: first 20 replayable rows of
`results/EXPC_POLARITY_CONTROL_FULL/raw_generations.jsonl` (model-generated,
non-failed; `ORIGINAL_PROOF` rows are stored gold text, not generations, and
are skipped), prompts reconstructed from `manifest.jsonl`
(question/target/prefix_steps/injected_statement) + the prompt block of
`run_metadata.json`, same model + pinned revision
(`Qwen/Qwen2.5-7B-Instruct @ a09a3545`), greedy, `max_new_tokens=192`.
Full per-row details in `replay_report.json` (batched) /
`replay_report_seq.json` (batch-1).

Cross-checks that the harness itself is faithful:

- validator agreement: re-running `src/validator.validate_continuation` on the
  STORED continuations reproduces the official `validated_outputs.jsonl`
  closure class **20/20** — the read-only validator hookup is exact.
- prompt identity: byte-for-byte (§3), and 7/20 vLLM generations are
  character-identical to the stored HF generations, which is impossible with a
  wrong prompt.

Batched submission (all 20 prompts in one `llm.generate` call):

- exact-match: **7/20 (35%)**
- first-divergence token position (13 diverging rows):
  0, 0, 1, 3, 3, 6, 8, 16, 17, 23, 26, 54, 96 (median 8)
- validator class flips: **9/20** (e.g. derailed -> valid_rederivation,
  parroted -> poisoned; both directions occur)

Divergences are genuine greedy near-ties, not prompt bugs: in every inspected
case both stored and regenerated text are coherent continuations of the same
prompt that pick a different branch (this task family puts the model on a
knife's edge by design — a planted false step — so tiny bf16 logit
differences between HF batch-1 kernels and vLLM paged/batched kernels flip
argmax, and one flipped token reroutes the rest of the proof).

Batch-1 sequential submission (`--sequential`, one prompt per
`llm.generate` call) gives **the same 7/20 exact / 9 class flips** with nearly
identical divergence positions (0, 0, 1, 3, 6, 8, 16, 17, 23, 26, 28, 54, 90)
— so the mismatch is HF-vs-vLLM kernel numerics, not vLLM's continuous
batching. Batching costs nothing extra in fidelity.

Implication for the experiment suites: vLLM is safe as the generation backend
for NEW runs (EXPD/EXPE/EXPG), where all conditions/cells are generated under
the same backend and compared within-backend. It is NOT safe to mix backends
inside one experiment, to append vLLM rows to an existing HF-generated results
directory, or to treat a vLLM rerun as a row-level reproduction of the locked
EXPA/EXPB/EXPC results. Condition-level rates should be re-estimated, not
assumed transferable row-by-row.

## 5. Throughput

Benchmark: first 100 manifest prompts (EXPC-style, ~485 prompt tokens each),
greedy, `max_new_tokens=192`, one H200, engine warm (startup excluded; cold
engine start adds ~60-90 s once per process):

- **2073.6 generated tokens/sec** (8397 tokens across 100 prompts in 4.05 s;
  24.7 prompts/sec; 48,515 prompt tokens prefilled in the same window)
- HF batch-1 baseline: ~30-60 tok/s => **~35-69x speedup** (vs the measured
  EXPC_FULL wall clock of ~33 s/generation, it is >200x per wall-clock row)

Scale check: EXPC_FULL's ~1,900 model generations, which took ~10 h under HF
batch-1, fit in ~2-3 min of vLLM generation time. The MASTER_PLAN §7 estimate
("single-digit hours after the vLLM port" for all suites) holds with large
margin; EXPD/EXPE/EXPG become engine-startup- and validation-bound, not
generation-bound.

## 6. Usage for EXPD / EXPE / EXPG runners

```bash
cd /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/tooling
source env.sh                 # HF_HOME + all caches on /lfs, 1 GPU, no __pycache__
CUDA_VISIBLE_DEVICES=<free gpu> .venv-vllm/bin/python your_runner.py
```

From Python:

```python
from vllm_gen import generate_continuations, render_prompt_ids, get_tokenizer

tok = get_tokenizer(MODEL, revision=REV)
rows = [{"prompt_token_ids": render_prompt_ids(tok, question, target,
                                               answer_prefix=" " + " ".join(steps))}
        for (question, target, steps) in batch]
outs = generate_continuations(MODEL, rows, max_new_tokens=192, revision=REV)
# outs[i]["text"] is the continuation (skip_special_tokens=True), input order preserved
```

Rows may alternatively carry `{"prompt": "<fully rendered string>"}` or
`{"messages": [...], "assistant_prefill": "..."}`. The engine is cached per
(model, revision) within a process — submit large batches, not one call per row.

CLI (fidelity replay / benchmark):

```bash
.venv-vllm/bin/python vllm_gen.py --replay ../../results/EXPC_POLARITY_CONTROL_FULL/raw_generations.jsonl \
  --manifest ../../results/EXPC_POLARITY_CONTROL_FULL/manifest.jsonl \
  --run-metadata ../../results/EXPC_POLARITY_CONTROL_FULL/run_metadata.json \
  --n 20 --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --validator-src ../../src --out replay_report.json
.venv-vllm/bin/python vllm_gen.py --benchmark --n 100 --model ... --manifest ... --run-metadata ...
```

## 7. Caveats

1. **Greedy != reproducible across backends.** 35% exact-match / 9 class
   flips out of 20 on this task family (see §4). Never mix HF- and
   vLLM-generated rows within one experiment or compare against locked
   results row-by-row. Re-run whole suites under one backend.
2. **`~/.cache` on AFS is full**: flashinfer ignores `XDG_CACHE_HOME` and
   crashes the engine with `Disk quota exceeded` unless
   `FLASHINFER_WORKSPACE_BASE` is set (env.sh handles this, plus
   HF/triton/vllm/inductor caches). ALWAYS `source env.sh` first.
3. **transformers 5.x API**: `apply_chat_template` returns a BatchEncoding;
   `vllm_gen.py` handles both v4 (list) and v5 (dict) forms. The vllm venv
   has transformers 5.12.1 vs 5.10.2 in the main venv; prompt rendering was
   verified identical for the Qwen2.5 template.
4. **Stop handling**: HF `generate` stops on `generation_config.eos_token_id`
   (for Qwen2.5: `<|im_end|>` 151645 and `<|endoftext|>` 151643);
   `vllm_gen.py` passes that union as `stop_token_ids` explicitly rather than
   relying on vLLM's generation-config auto-detection, and strips the trailing
   stop token before decoding (matching HF `skip_special_tokens=True`).
5. **Determinism within vLLM**: `enable_prefix_caching=False` and `seed=0`
   are set; repeated vLLM runs of the same batch were stable in this session,
   but vLLM does not guarantee bitwise determinism across its own versions or
   batch compositions either — pin `requirements-vllm.lock` for any suite you
   may need to re-run.
6. **One GPU**: the wrapper never sets device counts; control placement with
   `CUDA_VISIBLE_DEVICES` (env.sh defaults to 0 only if unset — pick a free
   GPU via `nvidia-smi` first). ~19 GB VRAM at
   `gpu_memory_utilization=0.85` on a 144 GB H200 for the 7B model.
7. `ORIGINAL_PROOF` rows in `raw_generations.jsonl` are stored gold text
   (`generated_by_model: false`), not generations — the replay CLI skips
   them automatically.
