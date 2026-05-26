# Generated VeriBench split manifests

Run from `experiments/02_ar_pros_cons`:

```bash
python -m data.setup --smoke
python -m data.setup --include-generated-agents --smoke
```

The command writes `data/splits/` locally:

- `veribench_manifest.jsonl`
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `smoke.jsonl`
- `summary.json`

Those files are intentionally gitignored because they contain absolute paths to
the local `~/veribench` checkout. Regenerate them after cloning.
