Stress-test runs (2026-07-31, author-directed, results discussed in chat before paper use).
Protocol: identical to e10_widen queue (expg_progtrace.py, cells/cap/oversample/decoding/seed
byte-matched to e10_config.sh); worlds copied from EXPG_PROGTRACE_qwen32b (identical substrate,
fingerprint 64583365b3ad48dc46b168be) so pairs share exact programs. Judge stage skipped
(flag channel = frozen doubt_lex only, as in E11). EXPG_LLM_MAX_MODEL_LEN=8192.
- EXPG_PROGTRACE_qwq32b:    Qwen/QwQ-32B (reasoning-RL on the roster's Qwen2.5-32B base).
  Prefill continuation mode = forced-direct arm (thinking suppressed by prefill); a
  think-channel arm may follow. Matched-base pair: compare to roster EXPG_PROGTRACE_qwen32b.
- EXPG_PROGTRACE_olmo7b_sft / _dpo: allenai/OLMo-2-1124-7B-{SFT,DPO} — training-stage ladder;
  compare to roster EXPG_PROGTRACE_olmo7b (final Instruct). Base checkpoint pending (no chat
  template; needs raw-prompt adapter).
