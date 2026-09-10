# Recompute-wall paper (ICLR draft)

Draft of **"The Recompute Wall: Language Models Do Not Re-Verify Computed
Values in Their Own Reasoning."** The experiment code and all run artifacts
live in
`experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/`
(which has its own README); the coupling is strictly one-way: artifacts →
registry → prose and figures. Nothing here is imported by the experiments.

## Layout

- `main.tex` — thin shell: ICLR 2025 style, anonymous author block
  (`\iclrfinalcopy` commented out), `\input`s the section files in order, and
  declares all four figure floats itself. **Prose is never edited in
  `main.tex`**; the section files must stay byte-identical to the drafting
  bundle, which is why the floats live here.
- `sections/` — eleven prose files plus the appendix:
  `00_abstract`, `01_intro`, `02_setup` (protocol), `03_wall` (13-model
  asymmetry), `04_depth` (one-operation onset), `05_opacity` (deference not
  inability), `06_fixes` (instruction null), `07_reasoning`, `08_related`,
  `09_discussion`, `10_conclusion`, `A_appendix` (two sections: External
  validity checks; Bookkeeping and provenance).
- `figures/` — `fig1.pdf` (protocol example, Haiku, seed 507786) after Sec 2;
  `fig2.pdf` (13-model readable-vs-computed bars) after Sec 3; `fig3.pdf`
  (depth ladder) after Sec 4; `fig4.pdf` (opacity arms) after Sec 5.
  `make_figs.py` regenerates **fig3 and fig4 only**, from the registry
  (`python3 make_figs.py [--outdir DIR] [--registry PATH]`). fig1/fig2 are
  static Jul-28 artifacts with no build script; fig2's data is in the
  registry's `fig2` block if a script is ever needed. All four captions are
  marked `CAPTION DRAFT` pending author sign-off.
- `main.bib` + vendored style files (`iclr2025_conference.sty/.bst`,
  `natbib.sty`, `fancyhdr.sty`, `math_commands.tex`) — no external deps.
- `CLAUDE.md` — prose style rules (full paragraphs, claim-first, no hype,
  numbers over adjectives, `\todo{verify}` instead of any untraceable number).

## Compile

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

(`pass1-3.log` + `bibtex.log` record this sequence; `bib.log` is dead debris
from a failed early run.)

## The numbers registry

`results/verified_numbers.json` is the single source of truth for every number
in the prose. Blocks: `_sources` (per-block provenance: artifact path, JSON
pointer, CI methodology, citation sites), `fig2`/`fig3`/`fig4` (parallel
arrays consumed by `make_figs.py`), and experiment blocks (`e13`,
`e13_bridge`, `e14_recon`, `e15_nl`, `natural_persistence`, `yee_renumber`).

Registry writes go through one-shot scripts, never hand edits.
`update_registry_e13c.py` is the template: read the run artifact, assert
invariants on every cell, refuse to overwrite an existing block, back up the
registry, write to `.tmp`, round-trip-validate, `os.replace` atomically.

CI conventions (recorded in `_sources.ci_policy_fig3_fig4`): fig3/fig4 use
program-cluster bootstrap 95% intervals (rollouts nest in programs), with
Wilson fallback at degenerate ceiling cells; fig2 uses Wilson (one continuation
per program per cell).

## Editing conventions

- Tex edits are applied by all-or-nothing anchored-replace scripts that take a
  `.bak` backup first; numbers in prose come from the registry.
- Backup suffixes: `.bak_aug5` = pre-edit snapshots of the Aug-5 session
  (registry, `main.bib`, seven section files); `.bak_e13c` = automatic registry
  snapshot taken by `update_registry_e13c.py`. Nothing reads the backups.
- LaTeX wording, figures, and captions are approved by the author in chat
  before being applied — flag, don't fix.

## Nearby directories

`papers/` contains other tracks (`latent_recovery`, `latent_recovery_tmlr` —
the pre-pivot drafts — plus unrelated projects), and `paper_latex/main.tex` at
the top level is a different paper entirely. This directory is the active ICLR
draft.

One fragile provenance note: the registry's `yee_renumber` block sources
`/lfs/skampere2/0/eobbad/scratch/yee_memo/results.txt`, which lives on scratch
outside both the paper and experiment trees.
