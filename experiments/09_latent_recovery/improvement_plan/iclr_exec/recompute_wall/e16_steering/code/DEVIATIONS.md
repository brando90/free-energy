# E16 deviations from the registered spec (logged pre-run)

1. **World partition instead of spec sample sizes.** The spec set extraction
   n=60/side and confirm n=150; each E11 family has its own ~150 gold-passed
   programs (families were gated independently and do NOT share programs), and
   extraction/evaluation disjointness takes precedence. Partition is
   PER-FAMILY (by sorted program_id, deterministic): last 60 = extraction,
   first 90 = evaluation, first 40 = screen. Confirm program-n is therefore 90
   per family, not 150 (greedy + R=4 gives 450 rollouts/cell). Extraction
   contrasts that mix families (V2) are unpaired, which diff-in-means
   permits.
2. **V1 representation.** The spec said planted-line-span and final-token
   representations are recorded separately; for the probe-framing side the
   planted line sits inside a chat-templated assistant turn where span
   indexing is not comparable, so V1 is built from final-token
   representations on both sides. V2 uses planted-line-span means on both
   sides. Both recorded in extract_manifest.json.
3. **V3 escalation swapped** from LinEAS to the critique-vector-style
   contrast (per the positioning amendment in E16_STEERING.md, committed
   pre-run after the lit check).
