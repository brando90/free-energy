# H00 Verdict

**TLDR:** The hypothesis is partially supported: normalization is not free, but
in this pure-Python benchmark `logsumexp` is a modest fraction of a small energy
network rather than the dominant cost.

On the 64K-candidate, batch-8 primitive sweep, raw argmin took
36.980 ms, `logsumexp` took 112.473
ms, and full softmax took 166.394 ms per batch. In the
synthetic energy-network split at 64K candidates, normalization was
2.7% of combined energy-network plus `logsumexp` time. The
ranking-only path preserves top-1 decisions and avoids the normalization pass;
softmax adds normalized probabilities but pays the exponential pass. Those
probabilities are not automatically calibrated: in the task benchmark softmax at
T=1 had ECE 0.229 (rank-margin heuristic:
0.360). This weakens the strongest version of "exp is the
main bottleneck" for small learned energies, but supports treating normalization
as a measurable cost at large candidate counts.

Caveat: the toy energy network costs about `dims` = 32 multiply-sin steps per
candidate versus one `exp` for `logsumexp`, so the normalization fraction is
close to 1/(dims + 1) by construction. The dtype sweep, peak memory and
numerical-failure counts asked for in the prompt were not measured.
