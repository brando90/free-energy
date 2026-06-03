# Open questions for the review paper

## Literature

- Which claims about AR/LLM limits are theorem-level versus empirical?
- Which softmax alternatives preserve global competition over memory?
- What is the cleanest statement of the AR factor as a locally normalized EBM?
- Which JEPA results should be treated as mature evidence versus motivation?
- Which diffusion-language-model results are closest to the AR pros/cons frame?
- What is the fairest baseline for AR + verifier/search?

## Experiments

- What toy example should `@eobbad` propose that is closer to VeriBench than
  independent Bernoulli token errors?
- What VeriBench metadata should `@Srivatsava` treat as the main proof-depth
  variable: tactic count, theorem count, Lean file length, generated token count,
  or something else?
- Should the first real-data curve use full VeriBench program/spec/proof files,
  VeriBench-FTP theorem-only examples, or both?
- Does the geometric model fail only under retry/search, or already under
  single-shot model generations?
- How should MNIST be tokenized for the cleanest order-effect result?

## Writing

- Keep Sutton's critique in an appendix or later discussion: it is about goal,
  ground truth, and continual learning, not specifically about AR factorization.
- Keep LeCun's `(1-e)^T` argument in the main text only as a testable hypothesis,
  not as a load-bearing proof.
- Use EBMs as the central alternative only after stating their real cost:
  the global partition function and hard sampling/inference.
