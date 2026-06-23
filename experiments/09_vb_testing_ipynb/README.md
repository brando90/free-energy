# VeriBench Harbor + Qwen Baseline

This experiment verifies the public Harbor registry entry for `veribench@1.1`,
downloads the Harbor task bundle, and runs a local Hugging Face baseline with
`Qwen/Qwen3-8B` over a VeriBench split.

Runtime setup used here:

```bash
uv venv --python 3.11 .venv
uv pip install -r experiments/09_vb_ebt_ipynb/requirements.txt

uv venv --python 3.12 .venv-harbor
UV_PROJECT_ENVIRONMENT=.venv-harbor uv pip install --python .venv-harbor/bin/python harbor
```

Smoke run:

```bash
PATH="$PWD/.venv-harbor/bin:$PATH" \
  .venv/bin/python experiments/09_vb_ebt_ipynb/using_veribench_with_harbor.py \
  --download-harbor \
  --pull-model \
  --run-baseline \
  --max-tasks 20 \
  --max-new-tokens 1024 \
  --compile-first-prompt \
  --output-prefix qwen3_8b_smoke_20task_compilefirst_1024
```

Harbor task verification requires access to the Docker daemon. On this machine,
the Harbor CLI and Docker CLI are installed, but Docker daemon access is denied
for the current user, so the Qwen generation baseline runs and records that
Harbor verification is blocked.
