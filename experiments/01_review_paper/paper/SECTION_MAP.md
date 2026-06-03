# Paper section to experiment map

This map prevents the review paper from becoming a literature-only essay.
Every major claim should either point to an experiment, a theorem, or an explicit
future-work marker.

| Paper section | Source/evidence | Experiment owner |
|---|---|---|
| Introduction | `paper_latex/01_introduction.tex`; `docs/foundational_takeaways.md` | review paper |
| What AR LLMs get right | exact chain rule, teacher forcing, exact likelihood, scaling/system evidence | review paper |
| Catalog of objections | `experiments/00_ar_pros_cons/CLAIMS.md`; `PROBE_SPECS.md` | AR pros/cons suite |
| What survives the audit | isolated probes + integrated ablations | `00_ar_pros_cons` |
| Experimental program | `paper_latex/04a_experimental_program.tex`; this experiment folder | `01_review_paper` |
| LeCun `(1-e)^T` test | `00_ar_pros_cons/toy/`; `probe_06_error_compounding.py`; VeriBench | `00_ar_pros_cons` |
| VeriBench/Lean real data | `00_ar_pros_cons/VERIBENCH.md`; local `~/veribench` | `@Srivatsava` review requested |
| Toy examples | `00_ar_pros_cons/toy/README.md` | `@eobbad` review requested |
| MNIST first vision domain | TODO smoke experiment | review paper / AR pros/cons suite |
| EBMs | `paper_latex/05_ebm_motivation.tex`; `06_ebm_training.tex`; `07_ebm_inference.tex` | EBM chapters |
| JEPA/diffusion/SSM/AR+verifier | `paper_latex/08_other_alternatives.tex` | alternatives chapter |
| Open questions | `paper_latex/98_appendix_open_questions.tex`; `notes/OPEN_QUESTIONS.md` | review paper |

## Result promotion rule

A result is ready to move from experiment notes into the paper only when it has:

1. a reproducible command;
2. a saved JSON/stat file;
3. a figure or table;
4. at least one caveat line;
5. a paragraph in `FINDINGS.md`.

Blog posts can discuss earlier-stage results, but the paper should mark them as
preliminary until those five conditions are met.
