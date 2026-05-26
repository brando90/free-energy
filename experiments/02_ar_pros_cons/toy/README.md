# Toy controls for LeCun's `(1-e)^T` hypothesis

The toy layer is not meant to prove anything about Lean. It is meant to make the
assumptions visible before we touch VeriBench.

## Question

When does a locally good autoregressive generator actually behave like:

```text
P(success at length T) = (1 - e)^T
```

and when does feedback break that curve?

## Current toy

`toy_error_process.py` simulates three processes:

1. **Blind AR rollout.** Each step has independent unrecoverable error
   probability `e`. This is LeCun's premise, so the geometric model should fit.
2. **Verifier resampling.** Each step can be retried up to `k` times. The raw
   proposal error can stay high, but the surviving error is lower.
3. **Recoverable state process.** The rollout can leave the valid manifold and
   return with recovery probability `r`. This tests the key assumption that
   errors are absorbing.

The output is a JSON summary plus a plot comparing empirical success curves
against the geometric prediction.

## Elyas input requested

@eobbad: please suggest one toy task that is closer to VeriBench than this pure
error-process control, ideally with both:

- a visible AR con: left-to-right commitment makes a local choice that can poison
  a later proof/program constraint;
- a visible AR pro: local factorization/teacher forcing gives a clean training or
  sampling advantage over a global scorer.

Candidate toy domains:

- a tiny Lean-like tactic language with a checker and a backtracking search loop;
- balanced parentheses with typed holes and delayed global constraints;
- small arithmetic programs where a verifier catches state-changing mistakes.
