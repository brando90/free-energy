# H01 Verdict

**TLDR:** The benchmark supports a narrow version of the hypothesis: a heavily
compressed scalar energy is bad for this position-sensitive task, while a
structured token-position energy recovers both accuracy and localization.

Across three seeds, the compressed scalar EBM reached
0.583 accuracy and
0.000 localization F1. The decomposed
EBM reached 1.000 accuracy and
0.867 localization F1, with a
positive-intervention probability drop of
0.975. This does not
prove that any scalar-valued neural energy is intrinsically too lossy; it shows
that if the scalar energy is not structured enough to expose token/position
contributions, the model can lose the task-relevant credit assignment signal.
