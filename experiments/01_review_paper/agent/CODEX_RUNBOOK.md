# Codex runbook for the review paper

You are editing the Free Energy review-paper project. Work from the repository
root:

```bash
cd /Users/brandomiranda/free-energy
```

## Ground rules

1. Do not duplicate the review paper. The canonical LaTeX source is
   `paper_latex/main.tex`.
2. Treat `experiments/01_review_paper/` as the coordination layer.
3. Treat `experiments/00_ar_pros_cons/` as the first empirical chapter.
4. When you add a blog post, use the canonical TL;DR block:

   ```text
   *Brando Miranda — Month YYYY · ~X min read*

   **TL;DR.** Single paragraph.

   ---
   ```

5. If you touch paper prose that cites a new paper, add or verify the BibTeX entry
   in `paper_latex/refs.bib`.
6. After any `.tex` edit in `paper_latex/`, run:

   ```bash
   cd paper_latex
   make
   ```

   If `make` fails because LaTeX is unavailable, report that explicitly.

## Immediate task queue

### Task 1: finish the LeCun error-compounding real-data probe

- Start from `experiments/00_ar_pros_cons/probes/probe_06_error_compounding.py`.
- Use `experiments/00_ar_pros_cons/data/setup.py` to create the VeriBench split.
- Compute success versus length/depth on the validation/test split.
- Fit three models:
  - geometric `(1-e)^T`;
  - constant pass probability;
  - recoverable Markov process.
- Write:
  - JSON stats;
  - one figure;
  - a paragraph in `experiments/00_ar_pros_cons/FINDINGS.md`;
  - a short update to `paper_latex/04a_experimental_program.tex`.

### Task 2: add MNIST methodology smoke

- Create an MNIST subfolder under `experiments/00_ar_pros_cons/` or a new
  dedicated experiment if it grows too large.
- Keep it small:
  - raster-order AR baseline;
  - shuffled-order AR baseline;
  - masked/iterative baseline;
  - simple EBM/ranker.
- The point is not SOTA. The point is to test whether order and local
  normalization create measurable global-validity effects on an interpretable
  domain.

### Task 3: paper/blog synchronization

- When a result is stable, update all three:
  - `experiments/00_ar_pros_cons/FINDINGS.md`;
  - `paper_latex/`;
  - `experiments/01_review_paper/blog/`.
- The blog post should read as a clean standalone explanation; the paper should
  read as the more formal version.

## What to ask Elyas and Sri

Tag `@eobbad` for:

- a toy example that is closer to VeriBench than independent Bernoulli errors;
- a toy that shows both the claimed AR failure and the real AR advantage;
- sanity checks on interpreting a recovered/non-geometric success curve.

Tag `@Srivatsava` for:

- whether the current task-level split is the right leakage boundary;
- which VeriBench metadata best approximates proof depth;
- whether pass@k should be measured on generated full files, isolated theorem
  proofs, or both.

## PR checklist

- [ ] `git status --short` shows only intended files staged.
- [ ] `make -C paper_latex` succeeds, or failure is explained.
- [ ] Blog draft uses the TL;DR block.
- [ ] PR body tags `@eobbad` and `@Srivatsava`.
- [ ] Assignment is attempted with GitHub CLI/API.
- [ ] If assignment is blocked by pending collaborator status, say so in the PR
      and final response.
