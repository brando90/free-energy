You are running on Brando's Mac inside the free-energy repo at
`/Users/brandomiranda/free-energy`. The goal of this codex session is to drive
a *full* run of the autoregressive-objections probes on Brando's snap-cluster
GPU hosts, email him the results, and push the outputs to `main`.

The orchestrator is already written and tested in smoke mode. Your job is to
run it, debug any failures, and keep going until a full run completes on at
least one host.

## What to run

```bash
cd /Users/brandomiranda/free-energy/experiments/00_ar_pros_cons
TAG=full DEVICE=cuda ./tools/drive_full_run.sh
```

That script walks `skampere2 -> skampere1 -> skampere3 -> mercury1 -> mercury2`
in order. On the first host that succeeds, it:

1. rsyncs `mnt/user-data/outputs/` back to this machine,
2. emails Brando at `brandojazz@gmail.com` via `tools/send_email.py`
   (Gmail app password at `~/keys/gmail_app_password.txt`),
3. commits `mnt/user-data/outputs/` and `logs/` and pushes to `origin/main`.

## What "full" means right now

Only probes 01, 03, 05 are implemented; the full tag for them just runs the
larger configs (more contexts/vocab/rank for 01, deeper stacks for 03, longer
parity lengths for 05). That is what `--tag full` (no `--smoke` flag) maps to
in each probe's argparse. Do **not** invent new probe files.

## Things you are allowed to change

- Fix bugs in `run_on_snap.sh`, `tools/drive_full_run.sh`, `tools/send_email.py`,
  or any `probes/*.py` if a real failure surfaces.
- Tweak the host list or device flags if a host is wedged.
- Add a retry / backoff loop around `drive_full_run.sh` if SSH is flaky.

## Things you must NOT do

- Do not invent new probes or change the experiment design.
- Do not commit secrets (passwords are in `~/keys/`, never inline them).
- Do not skip the email step.
- Do not push without first running the script to success on at least one host.

## Done conditions (all must hold)

1. `mnt/user-data/outputs/summary/full/summary.json` exists locally with
   `overall_control_passed: true` (or, if a control fails on the full config,
   document it in `FINDINGS.md` and still proceed).
2. An email with subject containing `[00_ar_pros_cons]` and `full run done`
   has been sent to `brandojazz@gmail.com`.
3. A commit on `origin/main` (not a branch) contains the new outputs.
4. Print the final commit SHA and the winning host before exiting.

Start by running the script. If a host fails, read the log under
`experiments/00_ar_pros_cons/logs/` and fix whatever the actual problem is.
Good luck.
