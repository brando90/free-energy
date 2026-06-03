# Blog workflow

This repo includes Brando's website as a git submodule because the project has
two connected writing surfaces:

- `free-energy`: experiment plans, claims, probes, findings, and some canonical
  blog drafts that grow directly out of an experiment.
- `website/brandomiranda`: the public website repo where blog drafts are
  rendered and eventually published.

The submodule is:

```text
website/brandomiranda -> https://github.com/brando90/brandomiranda.git
```

Initialize it from a fresh clone with:

```bash
git submodule update --init --recursive website/brandomiranda
```

## Canonical draft locations

Use the location that matches ownership of the content:

- If the post is mainly an experiment report, keep the canonical draft in
  `experiments/<NN_name>/blog/*.md` in `free-energy`.
- If the post is mainly website-native or personal narrative, keep the
  canonical draft in `website/brandomiranda/experiments/<NN_name>/blog/*.md`
  or `website/brandomiranda/_drafts/*.md`.
- If a `free-energy` experiment draft should render on the website, copy the
  draft into `website/brandomiranda/_drafts/`. Do not rely on external
  symlinks for anything that must be visible on GitHub.

The website copy should be a real markdown file on the website repo's `main`
branch. Local symlinks are fine for private editing convenience, but they are
not the published state.

## Required blog header

Every website-visible draft must follow the website repo rule immediately after
frontmatter:

```markdown
*Brando Miranda — Month YYYY · ~X min read*

**TL;DR.** Single paragraph summary.

---
```

Do not add a duplicate top-level `# Title`; Jekyll renders the title from
frontmatter.

## PR flow

When a draft affects the website:

1. Commit the website files inside `website/brandomiranda`.
2. Open a PR against `brando90/brandomiranda:main`.
3. If GitHub allows it, assign `@brando90`, `@eobbad`, and `@Srivatsava`.
   If the website repo does not allow Elyas/Sri as assignees, assign Brando and
   tag them in the PR body.
4. Merge the website PR.
5. Return to the `free-energy` root and commit the updated
   `website/brandomiranda` submodule pointer.
6. Open a `free-energy` PR against `main`, assigned to `@brando90`, `@eobbad`,
   and `@Srivatsava`.
7. Merge the `free-energy` PR.

This keeps both repos discoverable:

- website readers can find drafts directly in `brandomiranda`.
- project collaborators can clone `free-energy`, initialize the website
  submodule, and see the exact website commit associated with the experiments.

## Current examples

- `experiments/00_ar_pros_cons/blog/2026-05-26-ar-error-compounding-real-or-fiction.md`
  is canonical in `free-energy` and mirrored into the website drafts folder.
- `website/brandomiranda/experiments/08_ebm_partition_function_motivation/`
  owns a website-side source-note/blog workspace.
- `website/brandomiranda/experiments/10_autoregressive_llm_pros_cons/` owns
  the website-side consolidated LeCun/autoregressive pros-cons blog workspace.
