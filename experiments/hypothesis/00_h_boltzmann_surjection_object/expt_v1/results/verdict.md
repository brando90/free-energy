# H00 Verdict

**TLDR:** The hypothesis is partially supported: normalization is not free, but
in this pure-Python benchmark `logsumexp` is a modest fraction of a small energy
network rather than the dominant cost.

On the 64K-candidate, batch-8 primitive sweep, raw argmin took
36.980 ms, `logsumexp` took 112.473
ms, and full softmax took 166.394 ms per batch. In the
synthetic energy-network split at 64K candidates, normalization was
2.7% of combined energy-network plus `logsumexp` time. The
ranking-only path preserves top-1 decisions and avoids probability calibration;
softmax adds calibrated probabilities but pays the exponential pass. This
weakens the strongest version of "exp is the main bottleneck" for small learned
energies, but supports treating normalization as a measurable cost at large
candidate counts.
