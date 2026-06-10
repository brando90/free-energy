# VB Train/Val/Test Split — CE + SCSC ready, for Elyas' baselines (LM, EBT, Diffusion)

**TLDR:** Portable 3-way split of VeriBench gold tasks (891 tasks → 707 train / 87 val / 97 test, task-level, deterministic) with Python source and gold Lean **embedded in the JSONL**, so you can train CE baselines from these files alone and score test generations with the SCSC metric via the stable `task_name` linkage. Split assignment is byte-identical to the 2026-05-26 split in `experiments/00_ar_pros_cons` (verified: 0/1258 mismatches).

## Files (`splits/`)

| File | Rows | What it is |
|---|---:|---|
| `train.jsonl` | 707 (702 py-paired) | Gold tasks for training |
| `val.jsonl` | 87 (86 py-paired) | Gold tasks for tuning / model selection |
| `test.jsonl` | 97 (96 py-paired) | Gold tasks — **touch once, for final reported numbers** |
| `agent_variants.jsonl` | 368 (302/30/36 per split) | Agent-generated Lean candidates per task — e.g. EBT negatives / contrastive samples. Each inherits its task's split: **never mix across splits** |
| `smoke.jsonl` | 20 (14/3/3) | Tiny stratified subset for pipeline smoke tests |
| `summary.json` | — | Counts, fractions, pairing methods, unpaired task list |
| `manifest.jsonl` | 1259 | All rows in one file (gitignored — duplicates the above; regenerate with `make_split.py`) |

## Row schema (one JSON object per line)

`split` (train/val/test) · `task_id` (lean stem; the split key) · `task_name` (`<family>/<stem>` — matches the SCSC metric's `discover_gold_tasks` naming) · `variant_id` · `source_kind` (`gold` | `generated_agent`) · `family` · `rel_lean_path` / `rel_py_path` (relative to `veribench_dataset/`) · `paired` (bool) · `pairing_method` · **`py_code`** (full Python source, `null` for 7 lean-only tasks) · **`lean_text`** (full gold Lean, or candidate Lean for agent variants) · `line_count` / `char_count` / `theorem_count` / `sorry_count` / `tactic_count_proxy` · `sha256_lean` / `sha256_py`.

## How the split was made

- Universe = every `*.lean` under `~/veribench/veribench_dataset/lean_src/veribench/` (the SCSC gold dir), 891 tasks across 34 families; veribench repo @ `a2256617`.
- Assignment = SHA-256 stable bucket on `task_id` → 0.80 / 0.10 / 0.10. Deterministic, no RNG, reproducible from the script. **Task-level**: all agent variants of a task share its split (no leakage).
- Identical assignment to the earlier manifest-only split (`experiments/00_ar_pros_cons/data/setup.py`, generated 2026-05-26): all 1258 shared variant ids agree.
- Python pairing: mirrored path `py_src/<same relpath>` with exact stem (602), CamelCase→snake_case (275), or underscore/case-insensitive match (7). 884/891 paired.

## Usage

### CE (cross-entropy) — LM / Diffusion baselines

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={
    "train": "splits/train.jsonl", "val": "splits/val.jsonl", "test": "splits/test.jsonl",
})
ex = ds["train"][0]
prompt, target = ex["py_code"], ex["lean_text"]   # Python → Lean translation pair
```

- Conditional CE: loss on `lean_text` tokens given `py_code` as context (skip the 7 `paired == false` rows). Unconditional LM / diffusion over Lean: use `lean_text` of all rows.
- Tune and model-select on **val**; report **test** once.

### EBT (energy-based transformer)

Positives = gold `(py_code, lean_text)` pairs from `train.jsonl`. Extra negatives for a task: `agent_variants.jsonl` rows with the same `task_id` (they carry the task's `py_code` and a candidate `lean_text`). Variants already carry the task's split — filter `split == "train"` for training negatives. Caveat: 12 variants are orphans of since-removed tasks (`2_heappush*`, `3_heappop*`, …; `paired == false`) — usable only as unconditioned negatives.

### SCSC scoring of generations

SCSC = Smooth Correctness via Structural Conjunction, the 5-factor geometric mean (IC1·IC2·TE1·D1·D2)^(1/5) from `veribench_metric` (lives in `~/veribench/veribench_metric/`, same package in the `veribench_gpt55_scsc` checkout). The metric discovers gold tasks by `task_name = <family>/<stem>` — exactly the `task_name` field here, so:

1. Generate Lean for each row of `test.jsonl` (prompt from `py_code`), keyed by `task_name`.
2. Score against the gold dir, e.g.:
   ```bash
   python -m veribench_metric.eval_all \
     --gold-dir ~/veribench/veribench_dataset/lean_src/veribench/ \
     --lake-dir ~/veribench/veribench_dataset/lean_src/ \
     --output-dir results/
   ```
   then restrict the per-task scores to the 97 test `task_name`s (or use `runner.score_single_task` per task).
3. Lab rule: the TE1 LLM-judge factor must run through the logged-in CLIs (`clauded` / `codex exec` / `gemini`) — no raw provider API keys.

## Gotchas

- **Terminology**: VeriBench papers call the *families* (`easy_set`, `humaneval_set`, …) "splits". Here `family` = that; `split` = train/val/test only.
- 7 gold tasks have no Python source (CLRS-textbook formalizations in `cs_set`: `heapsort_*`, `linear_search` — full list in `summary.json`): keep for lean-only LM training and SCSC, skip for conditional CE.
- One stem collision (`MyWriteEnableGate` in `hardware_set_real` + `hardware_set_synthetic`): both hash to the same bucket → same split; `task_name` stays unique.
- `rel_*_path`s resolve only against a `veribench_dataset` checkout; the JSONLs themselves are self-contained for training.

## Regenerate

```bash
cd experiments/08_vb_train_val_test
python3 make_split.py   # defaults: --veribench-root ~/veribench/veribench_dataset --output-dir splits/
```
