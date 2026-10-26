# H02 Verdict

**TLDR:** The benchmark supports the hypothesis: full-context attention solves
the synthetic input-coverage task, while recurrence fails from forgetting and
energy scoring without full input access does not fix it.

Across lengths 16-512 and five seeds, transformer-style full-context lookup
achieved 1.000 mean accuracy, and the
full-context EBM reranker achieved 1.000. The
last-16 recurrent baseline achieved 0.353,
falling as contexts exceed its memory window, and the recent-only EBM reranker
achieved 0.076. This indicates that the
critical ingredient is full input access/attention, not energy scoring by
itself. The EBM reranker only succeeds when it uses the same full-context lookup
that the transformer baseline uses.

Caveat: the "transformer" baseline is an exact key scan and the full-context EBM
reranker calls that same scan, so both reach 1.000 by construction. The run
illustrates the input-access argument; it is not a trained-model comparison.
The full-context reranker's 0/1 energies also give it poor calibration (ECE
0.919).
