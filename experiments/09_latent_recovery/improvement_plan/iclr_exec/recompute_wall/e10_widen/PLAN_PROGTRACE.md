# PLAN_PROGTRACE (E10 WIDEN, open-weights half) — timestamped scope authorization

This file is the timestamped plan that authorizes running the EXPG program-trace
instrument BEYOND the pilot scope (n/cell > 35), which the locked
`expg_progtrace.py:guard_scope` gates behind `--confirm-timestamped-plan-progtrace`.

- Governing registration: `../E10_WIDEN.md` (LIGHT registration, pre-run).
- Scope granted: the 7 PILOT_CELLS at cap=150, oversample=240 (4x pilot budget),
  on the open-weights roster {qwen1p5b, qwen7b (re-anchor), olmo7b, llama8b, qwen32b}.
- Machinery: EXPG generator/validator reused UNCHANGED; this tree edits nothing in expg/.
- Shared world substrate fingerprint (5 models identical): e4981b1f679c182a.
- Honest exclusion (E10 §5) applies at analysis; --ignore-gate-a lets every model
  produce continuations so the per-cell/<25%-cohort rules can be applied.
- Prepared + CPU-validated: .
