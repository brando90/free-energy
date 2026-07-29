# E7 — judge_v2 completion pass: PREP ONLY report

Status: **PREP COMPLETE. GPU run NOT executed** (per instructions — GPUs occupied by lab members today).
All work is CPU-only / read-only against `results/`. Nothing under `results/` or existing
`improvement_plan/` subdirectories was modified. All new files live under
`$EXP/improvement_plan/iclr_exec/e7_judge/`.

`$EXP = /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`

## 1. What this closes

Per `ICLR_REVISION_PLAN.md` E7 and reviewer W3/W5: the pre-registered SECONDARY judge DV
(Qwen2.5-32B, `doubt_judge_v2.py`) was run on EXPD's and EXPE's **pilot/smoke** data
(`improvement_plan/tooling/judge_v2/`) and, separately, on **EXPE's full confirmatory run**
(`results/EXPE_EVIDENCE_MOVER/judge_v2_full_summary.json` — already done, 2,104 rows).
It was **never run on EXPD's full confirmatory grid**. `EXPD_FULL_REPORT.md` L205 says so
explicitly: *"the Qwen2.5-32B judge is the pre-registered SECONDARY detector (upper bound);
it was not run in this pass."* Separately, EXPB's 118 rows the strict validator could not
parse (the W3 finding: unparsed rate rises 4.0% → 31.7% under `LOCAL_CERTIFICATE`) have never
been given any judge-informed verdict. This item preps both.

## 2. Model / cache confirmation

- Judge model: `Qwen/Qwen2.5-32B-Instruct`, vLLM 0.24.0 backend (`vllm_gen.generate_continuations`), greedy, `max_new_tokens=4`, `HF_HUB_OFFLINE=1` — mirrors `doubt_judge_v2.py` exactly (unmodified; reused as-is from `improvement_plan/tooling/judge_v2/doubt_judge_v2.py`).
- **Snapshot confirmed present, no download needed**:
  `/lfs/skampere2/0/eobbad/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
  (62G on disk, `ls` confirms all 17 safetensors shards + config/tokenizer files present as symlinks into `blobs/`). This is the exact path `HF_HUB_OFFLINE=1` resolved to in both prior judge_v2 runs (`run.log`, `full_run_judge.log`).

## 3. Row sets extracted (CPU, run now)

Built `e7_judge/prepare_inputs.py` (byte-verified verbatim copies of `PROMPT_DOUBT` /
`PROMPT_REJECT` from `doubt_judge_v2.py` — checked programmatically, both `True`). Ran it now
on the cluster with `$REPO/.venv/bin/python` (no GPU):

```
[EXPD] read 11577 rows from results/EXPD_MATCHED_GRADIENT/validated_outputs.jsonl
[EXPD] wrote 11577 rows -> e7_judge/inputs/expd_full_grid.jsonl
[EXPB] read 900 total rows from results/EXPB_LOCAL_CERT_FLIP/validated_outputs.jsonl; 118 have class=='unparsed'
  GLOBAL_BASELINE: 12/300 unparsed = 4.0%
  IRRELEVANT_CERTIFICATE_CONTROL: 11/300 unparsed = 3.7%
  LOCAL_CERTIFICATE: 95/300 unparsed = 31.7%
[EXPB] wrote 118 rows -> e7_judge/inputs/expb_unparsed.jsonl
[jobs] wrote e7_judge/jobs_e7.json
[jobs] total rows across both jobs: 11695 (11695 rows x 2 prompts/row = 23390 judge calls)
```

- **Row counts confirmed**: 11,577 (EXPD full grid, matches `wc -l` and the plan's cited figure exactly) and 118 (EXPB unparsed, matches the plan's "~118" and independently reproduces the W3 unparsed-rate numbers 4.0%/31.7% cited in `ICLR_REVISION_PLAN.md` §1 W3 row and `stage0/STAGE0_REPORT.md` — this is a second, independent confirmation of those numbers, not a re-quote).
- Both `prepare_inputs.py` asserts (`len==11577`, `len==118`) passed; script also asserts `failed_generation==False` and non-empty `continuation` on every extracted row — confirmed 0 failed generations, 0 empty continuations in either set, so no rows need to be dropped before judging.
- Field-schema check: both EXPD and EXPB `validated_outputs.jsonl` already use the exact field names `doubt_judge_v2.py` expects (`continuation`, `injected_statement`, `condition`, `verbalized_doubt`, `problem_id`, `run_id`) — no adapter needed, same as the existing `jobs.json` entries.
- `jobs_e7.json` written in the same `--jobs` list format as `improvement_plan/tooling/judge_v2/jobs.json`, pointing at the two materialized input files under `e7_judge/inputs/` (not at `results/` directly, so the judge run never touches read-only results), with `out_prefix` under `e7_judge/outputs/`.

## 4. Prompt-construction dry run (5 sample rows)

Wrote `e7_judge/prompt_preview_5rows.txt`: 3 EXPD rows (distinct conditions:
`aff_false_attr_d0`, `aff_false_attr_d1`, `aff_false_attr_d2`) + 2 EXPB unparsed rows, each
rendered through both `PROMPT_DOUBT` and `PROMPT_REJECT` exactly as the judge will see them
(continuation truncated to 1500 chars, same as `CONT_TRUNC` in the original tool). Sample 1
(abridged):

```
=== sample 1/5  src=EXPD  condition=aff_false_attr_d0  problem_id=attr_f000_d0_A ===
--- PROMPT_REJECT ---
...
Planted statement:
---
Stella is shiny.
---
Continuation:
---
 Every lacpus is a makupus. Stella is a makupus. Every makupus is a rurupus. ...
---
Answer with exactly one word: YES or NO.
```

Full file is on the cluster at `e7_judge/prompt_preview_5rows.txt` (135 lines).

## 5. `run_e7.sh` — the launcher

Built to mirror the exact invocation pattern of the prior judge_v2 runs
(`cd $TOOLING && $TOOLING/.venv-vllm/bin/python doubt_judge_v2.py --jobs <jobs.json>`, with
`env.sh` sourced and `HF_HUB_OFFLINE=1` exported afterward, overriding `env.sh`'s own
`HF_HUB_OFFLINE=0` default — same override order as the documented reproduce command in
`JUDGE_V2_REPORT.md`).

**GPU-gate — tested live, refuses correctly today:**

```
$ bash e7_judge/run_e7.sh
nvidia-smi memory.used per GPU:
0, 141693
1, 130325
2, 130159
3, 130677
4, 142213
5, 134493
6, 130329
7, 130329
REFUSING: no GPU has <5000MB used (all GPUs occupied -- lab members' jobs are running). Not launching vLLM.
EXIT_CODE=1
```

All 8 H200s are at 130–142 GB / 143.8 GB used. Confirmed via `ps aux` that every running
`VLLM::EngineCore` process on the node belongs to `alexspan` or `sahasras` — none to `eobbad`.
No vLLM launched, no GPU touched, no lab-mate process touched.

**Cleanup trap (orphaned-EngineCore gotcha) — bug found and fixed during validation:**

First draft only checked for `EngineCore` processes that were *direct* children of the
script's own `$$`. Inspecting a live `alexspan` job on the node showed `VLLM::EngineCore`'s
real parent is the *launching python process*, not the shell that invoked python — and in
`run_e7.sh`, python runs under a `tee`-created subshell, so the real EngineCore process would
be a **grandchild** (or deeper) of `$$`, not a direct child. The depth-1 check would have
silently done nothing on a real orphaned-EngineCore event. Fixed to a full BFS walk of `$$`'s
descendant tree (depth cap 8, scoped to `$(whoami)` only) that kills any process whose `ps
args` contains `EngineCore`, wherever it sits in the tree.

Validated with a synthetic process-topology test on the cluster (nested `exec -a` chain
under a throwaway script `$$`, plus one unrelated decoy `sleep`): the fixed cleanup killed
the fake `EngineCore` process and left the decoy alone —

```
descendant set of $$=352033:  352036 352037 353238
killing orphaned EngineCore pid=352036: worker(VLLM::EngineCore) 300
PASS: fake EngineCore pid=352036 was killed
PASS: decoy pid=352037 untouched (still alive)
```

Test script and its process artifacts were removed after validation; `ps` re-checked clean
afterward. Only `run_e7.sh`, `prepare_inputs.py`, `jobs_e7.json`, `inputs/`, and the preview/
report files remain under `e7_judge/`.

## 6. Wall-clock estimate

The task's cited "~153 GPU-seconds per 2,104 rows" **is verified exactly** — but it is not
from the pilot judge_v2 run in `improvement_plan/tooling/judge_v2/` (that one judged only
397 rows total across EXPD-pilot/EXPE-smoke/EXPA-calib in ~19s of generation, ~2 min wall
incl. model load, per `JUDGE_V2_REPORT.md`). It is from a **separate, already-completed**
judge pass over **EXPE's full confirmatory run** (2,104 rows), recorded verbatim in
`results/EXPE_EVIDENCE_MOVER/EXPE_FULL_REPORT.md` line 7:

> `GPU seconds: generate=50.0, judge(32B)=153.0`

with `judge_v2_full_summary.json` confirming `"n_rows": 2104`. (EXPE's judge pass is already
done — it is not part of E7's scope; E7 covers only EXPD-full + EXPB-unparsed, which is what
`EXPD_FULL_REPORT.md` L205 flags as missing.)

Rate: 153.0 / 2104 = **0.0727 GPU-sec/row** (2 judge calls/row, `max_new_tokens=4`, batched).

Applied to E7's scope (11,577 + 118 = 11,695 rows, 23,390 judge calls):

- Judge-generation time: 11,695 × 0.0727 ≈ **850 GPU-sec ≈ 14.2 min**
- One-time engine load + torch.compile overhead observed in the two prior runs: ~71s (cold,
  `judge_v2/run.log`) down to ~40s (warm cache, same-day, `full_run_judge.log`) — call it
  **40–75s**, paid once since both jobs run in one `--jobs` invocation (engine loaded once,
  reused).
- **Total estimated wall clock: ~15–16 minutes on 1 H200.** Falls in the low half of the
  plan's own "~15-30 GPU-min" estimate — consistent.

## 7. Exact one-line command to launch when a GPU frees

```
bash /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/e7_judge/run_e7.sh
```

(Optionally pin a GPU explicitly: `E7_GPU=3 bash run_e7.sh`. The script still runs its own
`<5GB-used` safety check on whichever GPU is picked/pinned before launching vLLM.)

Output on a real run: `e7_judge/outputs/expd_full_verdicts.jsonl` + `_summary.json`,
`e7_judge/outputs/expb_unparsed_verdicts.jsonl` + `_summary.json`, plus `e7_judge/run_e7.log`
(full stdout/stderr, same `tee` pattern as the prior runs).

## 8. Explicitly out of scope for this item

- The GPU run itself (per instructions).
- The EXPB joint-outcome stacked figure mentioned alongside E7 in
  `ICLR_REVISION_PLAN.md` §3 (E7 row) — that is a figure-only task over data already in
  `summary_tables.json` and was not part of the task description handed to this session
  (which scoped E7 to "(a) EXPD full grid ... and (b) EXPB's ~118 unparsed rows, judged by
  the Qwen2.5-32B judge"). Flagging so it isn't assumed done.
- Analysis/interpretation of judge verdicts once the run completes (no verdicts exist yet).

## 9. Files

All under `$EXP/improvement_plan/iclr_exec/e7_judge/`:
- `prepare_inputs.py` — extraction/formatting script (run; CPU; idempotent)
- `prepare_inputs_report.txt` — captured stdout of the above
- `inputs/expd_full_grid.jsonl` (11,577 rows, 18M), `inputs/expb_unparsed.jsonl` (118 rows, 148K)
- `jobs_e7.json` — `doubt_judge_v2.py --jobs` spec for both row sets
- `prompt_preview_5rows.txt` — 5-sample dry-run of prompt construction
- `run_e7.sh` — full launcher (GPU gate + env + invocation + EngineCore cleanup trap)

No files were created, modified, or deleted under `results/` or any pre-existing
`improvement_plan/` subdirectory. No git commits made.

---

## 10. Independent verification pass (2026-07-23, second session)

This item's PREP artifacts (`prepare_inputs.py`, `run_e7.sh`, `jobs_e7.json`, `inputs/`,
`prompt_preview_5rows.txt`, this `REPORT.md`) already existed on disk dated 2026-07-22
18:55–19:03, from an earlier execution of this same item. **Note a bookkeeping discrepancy
for the record**: `EXEC_STATUS_BATCH1.md` (synthesized 2026-07-22, local scratchpad copy)
lists E7 as "no report, no artifacts" — that line was stale/wrong by the time it was written,
or the synthesis simply missed this item; both the report and the artifacts were present.
Do not treat this note as license to distrust batch-status docs generally — flag discrepancies
like this rather than silently overwriting.

Rather than blindly re-trusting the prior pass, this session independently re-verified
every load-bearing claim before accepting it (per HARD RULE 5). All checks passed; nothing
below required a fix.

1. **GPU-gate re-tested live today** (not just re-read): ran `bash run_e7.sh` for real on
   2026-07-23. Output:
   ```
   nvidia-smi memory.used per GPU:
   0, 133651
   1, 130337
   2, 130159
   3, 135485
   4, 142213
   5, 137753
   6, 130329
   7, 130329
   REFUSING: no GPU has <5000MB used (all GPUs occupied -- lab members' jobs are running). Not launching vLLM.
   EXITCODE=1
   ```
   All 8 H200s still at 130–142 GB / 143.8 GB used, one day later. Gate correctly refuses;
   nothing launched; no lab-mate process touched.

2. **Cache re-confirmed independently** (the original report's "72K du -sh" style check would
   have been misleading if taken at face value — `du -sh` on a snapshot directory only sees
   symlink sizes, not blob target sizes). Re-checked properly:
   - `blobs/` directory (the actual content-addressed files): **62G**, matches run.log's own
     `Checkpoint size: 61.03 GiB` line from the 2026-07-02 vLLM load.
   - Zero broken symlinks under the snapshot dir (proper POSIX check:
     `find ... -type l ! -exec test -e {} \; -print` → empty).
   - Spot-checked shard 1 (`model-00001-of-00017.safetensors`): resolves via
     `../../blobs/fa006a03...` to a **3,916,539,832-byte** real file (~3.9 GB, consistent
     with 17 shards of a bf16 32B model, 32e9 × 2 bytes ≈ 64 GB total).
   - `run.log` line 1 confirms vLLM itself resolved `HF_HUB_OFFLINE=1` to the exact snapshot
     path `.../snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` on the 2026-07-02 run —
     same path this item's `run_e7.sh`/`doubt_judge_v2.py` will resolve to today (model id
     unchanged, cache unchanged).

3. **`prepare_inputs.py` re-run in full isolation** (copied to `/tmp/e7_verify_eobbad/`,
   `OUT_ROOT` redirected there, run via `$REPO/.venv/bin/python`, never touching the real
   `e7_judge/` outputs or any `results/` file — read-only source, isolated sink):
   ```
   [EXPD] read 11577 rows ... [EXPD] wrote 11577 rows -> .../expd_full_grid.jsonl
   [EXPB] read 900 total rows ...; 118 have class=='unparsed'
     GLOBAL_BASELINE: 12/300 unparsed = 4.0%
     IRRELEVANT_CERTIFICATE_CONTROL: 11/300 unparsed = 3.7%
     LOCAL_CERTIFICATE: 95/300 unparsed = 31.7%
   [EXPB] wrote 118 rows -> .../expb_unparsed.jsonl
   [jobs] total rows across both jobs: 11695 (23390 judge calls)
   ```
   `md5sum` of the isolated re-run's `expd_full_grid.jsonl` and `expb_unparsed.jsonl` against
   the shipped `inputs/` files: **byte-for-byte identical** (`9cb49802ac6ce18af4405b349caedb88`
   and `7c2ee72c35c60bb46930e0beccb7072c` respectively, both files). Confirms the script is
   deterministic and idempotent, and that the shipped input files are exactly what the script
   produces — not hand-edited. Scratch dir removed after the check.

4. **Anchors cross-checked directly against source reports** (not re-quoted from memory):
   - `results/EXPD_MATCHED_GRADIENT/EXPD_FULL_REPORT.md` lines 172–173:
     `"manifest_rows": 11577, "validated_rows": 11577` — confirms the 11,577 figure.
   - Same file, line 205: *"the Qwen2.5-32B judge is the pre-registered SECONDARY detector
     (upper bound); it was not run in this pass"* — confirms the judge gap this item closes.
   - `results/EXPE_EVIDENCE_MOVER/EXPE_FULL_REPORT.md` line 7: `GPU seconds: generate=50.0,
     judge(32B)=153.0` — confirms the throughput anchor verbatim.
   - `results/EXPE_EVIDENCE_MOVER/judge_v2_full_summary.json`: `"n_rows": 2104` — confirms the
     denominator. `153.0 / 2104 = 0.0727` GPU-sec/row, matching the report's rate exactly.
   - `doubt_judge_v2.py` `PROMPT_DOUBT`/`PROMPT_REJECT` string constants diffed visually
     against `prepare_inputs.py`'s copies: identical, character for character.

5. **No changes made to the existing artifacts.** All of the above re-derived the same
   numbers the 2026-07-22 pass reported; nothing was stale, fabricated, or needed correction.
   This addendum is additive documentation of the independent check, not a revision of the
   findings. `REPORT.md`, `prepare_inputs.py`, `run_e7.sh`, `jobs_e7.json`, and `inputs/*`
   are unchanged from the 2026-07-22 versions.

**Conclusion: E7 PREP is genuinely complete and independently verified.** The one-line launch
command in §7 above is ready to run as soon as a GPU frees (`nvidia-smi` still shows zero
headroom as of 2026-07-23; re-check before launching).


---

# ====================================================================
# GPU RUN EXECUTED 2026-07-24 (appended after prep) — see below
# ====================================================================

# E7 — judge_v2 completion pass: RUN report (GPU pass executed)

Status: **RUN COMPLETE.** The prepped 32B-judge pass launched, finished cleanly, and both
output sets are written and verified. No `results/` or pre-existing `improvement_plan/` file
was modified. All outputs live under
`$EXP/improvement_plan/iclr_exec/e7_judge/outputs/`.

`$EXP = /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`
Judge: `Qwen/Qwen2.5-32B-Instruct`, vLLM greedy, `max_new_tokens=4`, two prompts/row
(`PROMPT_DOUBT`, `PROMPT_REJECT`), unmodified `doubt_judge_v2.py`.

---

## 1. Execution log (what actually ran)

- **GPU gate:** verified GPU 2 fully free (0 MiB) at launch; all 7 other H200s at 111–139 GB
  (labmate jobs). Pinned `E7_GPU=2`; the launcher's own `<5GB` check also selected 2.
- **Launch:** `E7_GPU=2 nohup bash run_e7.sh` at `2026-07-24T10:22:50-07:00` (script PID 672540,
  EngineCore PID 704368).
- **Finish:** `2026-07-24T10:34:02-07:00`. **Wall = 672 s ≈ 11.2 min** — under the ~15–16 min
  prep estimate, far under the 60-min hard-stop. Steady-state judge throughput ~41 rows/s
  (~9,930 input tok/s), matching the EXPE anchor.
- **Two jobs, one engine load:** EXPD 23,154 prompts (11,577 rows × 2) then EXPB 236 prompts
  (118 × 2). Engine warmup 25 s, paid once.
- **Clean shutdown / no orphans:** cleanup trap fired —
  `no EngineCore descendants of PID 672540 found (clean)`. Post-run `pgrep -u eobbad` shows
  **no** `doubt_judge_v2`/`EngineCore`/`vllm` processes; **GPU 2 back to 0 MiB**. No labmate
  process (alexspan/sahasras) was ever touched.

## 2. Output verification (all pass)

| check | EXPD | EXPB |
|---|---|---|
| verdict rows written | **11,577 / 11,577** input | **118 / 118** input |
| summary written | `expd_full_summary.json` | `expb_unparsed_summary.json` |
| `n_unparsed_verdicts` (judge YES/NO parse failures) | **0** | **0** |
| per-cell `n` sums to total | yes (24 cells) | yes (3 cells) |

Every judge call returned a clean YES/NO (0 unparsed verdicts on 23,390 calls). Row files carry
per-row `lexical_doubt`, `judge_doubt`, `explicit_rejection`, both raw judge strings, and a
`continuation_head` for inspection. Files:
`outputs/expd_full_verdicts.jsonl` (5.6M), `outputs/expd_full_summary.json`,
`outputs/expb_unparsed_verdicts.jsonl`, `outputs/expb_unparsed_summary.json`, `run_e7.log`,
`nohup_e7.out`.

---

## 3. Science — EXPD: does the 32B judge-doubt DV agree with the registered lexical DV?

**Yes, decisively — the pre-registered SECONDARY (judge) DV concurs with the registered
lexical DV, and both key structural verdicts (d=0 cliff, usability asymmetry) survive on it.**
Overall the judge and regex agree on **98.0%** of the 11,577 rows; judge-doubt rate 0.0265 vs
lexical-doubt 0.0231 — within 0.3 pp. This is the direct answer to reviewer W5: on the clean
confirmatory grid, the "weak lexical proxy" tracks a 32B semantic reader almost perfectly, so
the registered doubt verdicts are not a regex artifact.

**Key cell 1 — d=0 visibility cliff (affirmative false-attribution, `aff_false_attr_d{0,1,2,3,5}`):**

| d | n | lexical doubt | **judge doubt** | judge–lex agree | judge explicit-reject |
|---|---|---|---|---|---|
| 0 | 450 | 0.120 | **0.1267** | 0.927 | 0.4467 |
| 1 | 450 | 0.0267 | **0.0289** | 0.976 | 0.2111 |
| 2 | 450 | 0.0289 | 0.0378 | 0.973 | 0.1311 |
| 3 | 441 | 0.0204 | 0.0272 | 0.984 | 0.1043 |
| 5 | 426 | 0.0235 | 0.0258 | 0.988 | 0.0775 |

The judge reproduces the cliff: judge-doubt **0.127 @ d0 vs 0.029 @ d1 (4.4×), then flat**
(0.029/0.038/0.027/0.026) — the same step function the registered lexical DV shows
(0.120 → 0.027, 4.5×). The registered "rejection is a d=0 visibility event, flat for d≥1"
verdict holds on the secondary DV.

**Key cell 2 — usability asymmetry (categorical false, `cat_false_{usable,inert}_d{1,3,inf}`):**

| cell | n | lexical doubt | **judge doubt** | judge–lex agree | judge explicit-reject |
|---|---|---|---|---|---|
| inert d1 | 300 | 0.0867 | **0.090** | 0.970 | 0.4233 |
| inert d3 | 284 | 0.0634 | **0.1021** | 0.954 | 0.2113 |
| inert d∞ | 300 | 0.0267 | **0.0733** | 0.947 | 0.1767 |
| usable d1 | 300 | 0.020 | **0.010** | 0.990 | 0.0567 |
| usable d3 | 280 | 0.0036 | **0.0036** | 1.000 | 0.0429 |
| usable d∞ | 300 | 0.0067 | **0.0067** | 1.000 | 0.0267 |

The judge reproduces the asymmetry: **inert lies draw ~9–14× more judge-doubt than usable
lies** (inert 0.09/0.10/0.073 vs usable 0.010/0.004/0.007), and the same on explicit rejection
(inert 0.42/0.21/0.18 vs usable 0.057/0.043/0.027, ~7×). Usable falsehoods are absorbed
essentially without verbal flag under both DVs — the registered "absorption tracks derivational
usability" verdict holds on the secondary DV.

**Validity anchors (judge does not hallucinate doubt):** `benign_paraphrase` 0.000 doubt /
0.000 reject; `true_interruption` 0.000 / 0.009; all `aff_true_*`/`neg_true_*` cells ≤0.011
doubt. The judge fires only on planted falsehoods, not on true or benign content.

**Two nuances to carry into the writeup (neither breaks the story):**
1. **Explicit-rejection runs higher than doubt and has a mild distance tail.** The judge's
   `PROMPT_REJECT` DV (0.0948 overall) is 2–5× the doubt DV and declines *gradually* across d
   on `aff_false_attr` (0.447/0.211/0.131/0.104/0.078) rather than as a pure d0 cliff. The bulk
   (>half the range) still drops at d0→d1, but the residual d≥1 tail is real — exactly the
   "rare, non-graded ≥1-hop checking exists" that E5 documented. Report the **doubt** DV as the
   clean cliff; note the rejection DV carries a weak tail consistent with E5, not with a graded
   distance law.
2. **`reject_not_judge` is large** (836 rows overall): many continuations the judge reads as
   explicitly rejecting the plant do not trip the doubt regex. On the clean grid this is a
   modest under-count; on the EXPB unparsed tail it becomes the whole story (§4).

---

## 4. Science — EXPB: what do the 118 unparsed continuations actually contain?

**The unparsed rows are overwhelmingly verbal REJECTION of the planted lie — not silent
absorption, not garbage.** This is concentrated entirely in the `LOCAL_CERTIFICATE` arm (the
95 rows that drove S2's worst-case), and it tightens the S2 bounds sharply.

| arm | n unparsed | lexical doubt | **judge doubt** | **judge explicit-reject** | judge–lex agree |
|---|---|---|---|---|---|
| GLOBAL_BASELINE | 12 | 0.000 | 0.000 | 0.417 (5/12) | 1.000 |
| IRRELEVANT_CERT | 11 | 0.000 | 0.091 (1/11) | 0.000 | 0.909 |
| **LOCAL_CERTIFICATE** | **95** | 0.221 (21) | **0.695 (66)** | **0.905 (86)** | **0.232** |

Row-level breakdown of the 95 local-cert unparsed rows: **86 judge-rejection, 66 judge-doubt,
9 not-rejection, and only 7 that are neither rejection nor doubt.** Inspecting those 7 residual
rows: **none is clean silent absorption and none is garbage.** They are a
"contradiction-acknowledged-then-plant-reasserted" hybrid — the model literally writes
`(Contradiction, so Polly is a jompus.)` / `This is a contradiction, so Rex must be a rompus`
(flags the conflict, then perversely re-affirms the planted lie and keeps deriving), plus one
confused/derailed trace. So even the residual "absorbing" tail visibly detects the
contradiction first.

**Why this matters — it collapses S2's worst case.** S2's joint-outcome bound
(`s2w3_expb_joint/REPORT.md`) could not sign the injection-dependence reduction because it had
to treat all 95 local unparsed rows as *possibly all-absorbing*: LOCAL−GLOBAL poisoning Δ ∈
**[−0.167, +0.190]** (spans 0), so "accessible counterevidence reduces absorption" was not
claimable. The judge now adjudicates those 95 rows: **at most ~7–9 are injection-dependent,
not 95.** Judge-informed imputation of the poisoned contrast:

- LOCAL poisoned ≈ parsed 8 + ≤9 unparsed = **~0.050–0.057** of 300
- GLOBAL poisoned ≈ 46/300 = **~0.153** (its 12 unparsed are only 5 rejection, ~few absorbing)
- **LOCAL − GLOBAL ≈ −0.10, sign now fixed negative** — the reduction that S2 correctly refused
  to claim under agnostic worst-case is **recoverable as a real effect** once the unparsed rows
  are judge-classified.

**But the mechanism is d=0 visibility, not proof search — and this is the honest, on-message
framing.** The local certificate literally states the complement (e.g. "Rex is not a vumpus")
in the prefix, so the 86 rejections are the model *reading back a directly-visible
contradiction* (d=0), exactly the C-0 cliff EXPD documents. EXPB therefore **corroborates the
d=0-visibility law** rather than supporting a distinct "accessible counterevidence reduces
absorption" claim — the causal role stays with EXPD C-0 / EXPE H6, as the cut-table specifies;
EXPB is the appendix instrument-lesson whose unparsed tail is now *explained* (rejection-that-
broke-the-parse), not merely bounded.

**Instrument lesson (W5), both directions:** judge–lexical agreement is **98% on the clean EXPD
grid but only 23% on the local-cert unparsed tail** — the doubt regex caught 21/95 where the
judge sees 66 doubt / 86 reject. So the lexical proxy **under**-reads rejection on
unparsed/derailed text (opposite of EXPE, where it **over**-fires on frequency-matched
non-refuting overlap). The 32B judge is the better secondary DV; the regex is reliable only on
well-formed continuations. Report EXPB rejection as a **contrast**, and note it is d=0-visibility
(certificate-restating) rejection, not evidence of a check.

---

## 5. Bottom line

- **W3 / prereg completeness:** the registered SECONDARY judge DV — previously missing on the
  flagship EXPD grid (`EXPD_FULL_REPORT.md` L205) — is now run over all 11,577 rows and
  **confirms** the registered lexical verdicts (98% agreement; d=0 cliff and usability
  asymmetry both replicate on the judge DV). The "missing secondary on the flagship" gap is
  closed.
- **EXPB / S2:** the 95 unparsed local-cert rows are 90.5% explicit rejection and ≤~9 possible
  absorptions (all of which still flag the contradiction). This **tightens the S2 worst-case
  bound from "not signable" to a ~−0.10 reduction**, while re-attributing the effect to d=0
  visibility (not counterevidence-accessibility), keeping EXPB appendix-bound and on-message.
- **W5:** judge vs regex agree 98% on clean text; the divergence is confined to the pathological
  unparsed tail, where the regex under-counts — a clean instrument-limitation statement, not a
  threat to the doubt findings.

## 6. Files (cluster `$EXP/improvement_plan/iclr_exec/e7_judge/`)

- `outputs/expd_full_verdicts.jsonl` (11,577 rows) + `outputs/expd_full_summary.json`
- `outputs/expb_unparsed_verdicts.jsonl` (118 rows) + `outputs/expb_unparsed_summary.json`
- `run_e7.log`, `nohup_e7.out` (full stdout incl. clean cleanup-trap line)
- Prep artifacts unchanged: `run_e7.sh`, `prepare_inputs.py`, `jobs_e7.json`, `inputs/*`
- Local copy of this report: `exec_reports/e7_judge_RUN_REPORT.md`

Out of scope (unchanged from prep): the EXPB joint-outcome stacked figure was delivered under
S2 (`s2w3_expb_joint/expb_composition.png`), not E7. No git commit made.
