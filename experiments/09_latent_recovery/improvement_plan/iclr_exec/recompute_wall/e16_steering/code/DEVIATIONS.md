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
4. **Norm-matched screen extension (post-screen, pre-confirm-adjudication).**
   Reading arXiv:2603.16331 in full showed their steering convention adds the
   UNNORMALIZED diff-in-means vector with alpha <= 1.0, i.e. doses up to the
   full natural magnitude of the contrast (our raw norms: 12.7-101.6). The
   registered grid (unit vector, alpha <= 16) therefore reached only
   ~16-30% of that dose at the mid/late layers. Twelve conditions are added:
   alpha = ||v|| x {0.5, 1.0} per vector x layer. Disclosed extension, same
   worlds, same breach rule; motivated by dose comparability with the
   published positive result, not by the screen outcome.
