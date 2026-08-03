"""E14 RECON staged runner -- Yee reconciliation 3x2 on the k=1 computed site.

Reuses the E11 machinery by IMPORT, not by copy:
  * worlds        recon_gen.py (built on gen_depth.build_program(1, seed))
  * audits        e11_generator/audit_depth.make_audit_fn  (fail-closed, re-run
                  at prepare on every world before it is admitted) PLUS
                  recon_gen.post_checks (the anchor / regime invariants; also
                  fail-closed at prepare -- this is the falsification instrument
                  and must not be trusted from the generator log alone)
  * injection     e11_run.build_injection_e11  (transform='value' branch: the
                  generator's byte-audited site_body spliced over the model's
                  OWN gold site line)
  * gold gate     e11_run.stage_generate  (own-trace gold + inject.gold_solve_eval
                  double filter), driven with the recon family taxonomy patched in
  * scoring       expg/validate.classify_run  (the SAME validator as E11; no
                  recon-specific scoring anywhere). stage_validate is a local
                  copy of e11_run.stage_validate that additionally PERSISTS the
                  parsed final-output integer and the two target values; the
                  classification itself is untouched.
  * stats         e11_run._cell_summary / wilson / cluster_boot_rate

Cells (6), all riding the SAME world -> perfectly paired within-world:
    recon_bare_pm10_20   ours          (bare site,  |delta| in {10,20})
    recon_full_pm10_20   ours + Yee B  (operands printed, |delta| in {10,20})
    recon_bare_pm1       ours + Yee A  (bare site,  |delta| == 1)
    recon_full_pm1       Yee corner    (operands printed, |delta| == 1)
    recon_bare_large     decoupler     (bare site,  |delta| in [100,400])
    recon_full_large     decoupler     (operands printed, |delta| in [100,400])

The pm10_20 cells are E11 ANCHORS: their site_body is byte-identical to
depth_k1_bare / depth_k1_full, so they must reproduce E11's k1_bare / k1_full
absorbed rates. Pass --e11-summary <E11 summary_tables.json> to have the
anchor deltas computed into the contrasts block (a path that does not exist is
a hard error, not a silent skip).

SIGN CONVENTION (v2). Every contrast in the summary is oriented
    FULL minus BARE   and   <regime> minus <reference regime>,
and every key name states its own direction. v1 had `opacity_effect_*` as
FULL-minus-BARE while the adjacent `difference_in_differences` was built the
other way round; the keys are renamed so that cannot recur.

MEASUREMENT SEPARATION (v2). The k=1 shape is out = Q1 + Q2 with Q1 = V + c and
Q2 = V + c', so |out_cf - out_true| = 2 * |delta|: exactly 2 in the pm1 arm,
20/40 in pm10_20, >=200 in large. An ordinary arithmetic slip in a genuinely
ABSORBING pm1 continuation can therefore land on out_true and be scored as
recovery, which biases the pm1-vs-pm10_20 contrast in the direction of our own
hypothesis. The DV is NOT changed. Instead every validated row now carries
`final_output_value`, `out_true`, `out_cf`, `separation`, and a bucket label,
and summarize emits per-cell near-miss diagnostics and a `bias_bound` -- the
number the paper quotes when asserting the contrast is not a measurement
artifact.

Stages (one per invocation, EXPG/E11 pattern):
  python recon_run.py prepare   --out-dir D --worlds W
  python recon_run.py generate  --out-dir D --model M --gpu 0 --cap 150 --reps 8
  python recon_run.py validate  --out-dir D --reps 8
  python recon_run.py summarize --out-dir D --model M
  python recon_run.py report    --out-dir D

Nothing runs at import time. --gpu is applied (CUDA_VISIBLE_DEVICES) inside
main(), before any backend is constructed.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

IP = os.environ.get(
    "E14_IP",
    "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan")
RW = os.path.join(IP, "iclr_exec", "recompute_wall")
OUTBASE_DEFAULT = os.path.join(RW, "e14_recon")
# DEFAULT IS THE 1800-WORLD POOL, NOT THE 240-WORLD FILE. E11's MEASURED k=1
# gold-solve rate is 0.1417 (Qwen2.5-7B) / 0.2039 (Llama-3.1-8B), so a 240-world
# pool yields only ~34 / ~49 worlds per cell -- "INVALID_underpowered" under
# E11's own power rule (<50). 1800 is E11's own per-k pool size, which is what
# produced its 255 eligible / 150 kept at k=1; using it power-matches the recon
# cells to the E11 anchors by construction. recon_worlds.jsonl (240) is exactly
# the first 240 lines of this file and is kept for quick smoke tests.
WORLDS_DEFAULT = os.path.join(OUTBASE_DEFAULT, "worlds",
                              "recon_worlds_pool1800.jsonl")

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.join(RW, "e11_run"), os.path.join(RW, "e11_generator"),
           os.path.join(IP, "expg")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import e11_run as e11                           # noqa: E402
import audit_depth as ad                        # noqa: E402
import gen_depth as gd                          # noqa: E402
import validate as val_mod                      # noqa: E402
import trace_format as tf                       # noqa: E402
import recon_gen as rg                          # noqa: E402
from interp import parse_program                # noqa: E402

K = 1
REGIMES = rg.REGIMES                            # ("pm10_20", "pm1", "large")
OPACITIES = rg.OPACITIES                        # ("bare", "full")
CELLS = rg.RECON_FAMILIES                       # 6 cells, bare-major order
ANCHOR_PAIRS = {"recon_bare_pm10_20": "k1_bare", "recon_full_pm10_20": "k1_full"}
DELTA_OF = {"recon_%s_%s" % (o, t): t for o in OPACITIES for t in REGIMES}
OPACITY_OF = {"recon_%s_%s" % (o, t): o for o in OPACITIES for t in REGIMES}

# Regime contrasts, all oriented "<A> minus <B>". pm10_20 (ours) is the
# reference; large-minus-pm1 is the decoupling read.
REGIME_CONTRASTS = (("pm1", "pm10_20"), ("large", "pm10_20"), ("large", "pm1"))

# --- measurement-separation diagnostics ------------------------------------
# A final output within +/-NEAR_WINDOW of a target, but on neither target, is a
# "near miss": observable evidence that arithmetic slips of that size occur in
# this cell. It is the only handle we have on how often a slip lands EXACTLY on
# the other target (which is unobservable, and which is what the pm1 separation
# of 2 makes dangerous).
NEAR_WINDOW = 3
BUCKETS = ("exact_cf", "exact_true", "near_cf", "near_true", "near_both",
           "other_far", "no_output_claim")
NEAR_BUCKETS = ("near_cf", "near_true", "near_both")


def output_bucket(v, out_true, out_cf, w=NEAR_WINDOW):
    """Classify the parsed final output. exact_cf is tested first: out_cf and
    out_true are audited distinct (audit_depth.audit_plant_validity rejects
    plant_washout), so the order only documents intent."""
    if v is None or out_true is None or out_cf is None:
        return "no_output_claim"
    if v == out_cf:
        return "exact_cf"
    if v == out_true:
        return "exact_true"
    dt, dc = abs(v - out_true), abs(v - out_cf)
    if dt <= w and dc <= w:
        return "near_both"
    if dt <= w:
        return "near_true"
    if dc <= w:
        return "near_cf"
    return "other_far"


# ------------------------------------------------- recon family taxonomy

def families_of(prog):
    """Only the six recon cells are SCORED. depth_k1_* stay on the world
    because audit_opacity_byte_identity requires depth_k1_bare to exist
    (audit_depth.py:330), but they are not generated against."""
    return [f for f in CELLS if f in prog["families"]]


def cell_of(prog, family):
    return prog["k"], prog["families"][family]["opacity"]


def _patch_e11():
    """Swap E11's family taxonomy for the recon one, and pin KS to [1] so the
    gold cohort is computed over the single k=1 stratum. Everything else in
    e11_run.stage_generate is used verbatim."""
    e11.KS = [K]
    e11.families_of = families_of
    e11.cell_of = cell_of


# ----------------------------------------------------------- prepare (CPU)

def stage_prepare(args):
    """Admit worlds from the recon_gen output into the run dir, re-running the
    FULL audit_depth stack AND recon_gen.post_checks fail-closed on each one.
    Refuses to write a world that fails any audit or lacks any of the six recon
    cells.

    post_checks is re-run here on purpose: the anchor byte-identity invariant is
    this experiment's falsification instrument, and a pool file that was edited,
    truncated, or generated by an older recon_gen would otherwise sail through
    prepare on the strength of a generator log nobody re-checked."""
    os.makedirs(args.out_dir, exist_ok=True)
    P = e11.paths(args.out_dir)
    if os.path.exists(P["programs"]) and not args.overwrite:
        raise SystemExit("programs.jsonl exists; use --overwrite")
    for key in ("programs", "manifest", "raw", "validated", "audit"):
        if os.path.exists(P[key]) and args.overwrite:
            os.remove(P[key])

    if not os.path.exists(args.worlds):
        raise SystemExit("--worlds %r does not exist" % args.worlds)
    worlds = e11.read_jsonl(args.worlds)
    if not worlds:
        raise SystemExit("no worlds at %s (run recon_gen.py first)" % args.worlds)

    t = ad.default_targets()
    t["nearest_anc_dist"] = 4                    # e11_run._targets_for(1)
    audit_fn = ad.make_audit_fn(t)

    kept, rejects = 0, Counter()
    for p in worlds:
        missing = [f for f in CELLS if f not in p.get("families", {})]
        if missing:
            rejects["missing_cell:%s" % missing[0]] += 1
            continue
        if p.get("k") != K:
            rejects["wrong_k"] += 1
            continue
        fails = audit_fn(p)
        if fails:
            rejects["audit:%s" % fails[0]] += 1
            continue
        pc = rg.post_checks(p)                   # FAIL-CLOSED anchor/regime check
        if pc:
            rejects["postcheck:%s" % pc[0]] += 1
            continue
        p.setdefault("audit_metrics", {})["nearest_anc_dist"] = \
            ad.nearest_ancestor_token_distance(p)
        e11.append_jsonl(P["programs"], p)
        kept += 1

    if kept == 0:
        raise SystemExit("prepare admitted 0 worlds; rejects=%s" % dict(rejects))

    meta = {
        "experiment": "E14_RECON_YEE",
        "created_at": e11.now_iso(), "seed": args.seed,
        # recorded so summarize never has to fall back to an argparse default
        # (the trap that mislabelled E11_llama8b's summary as Qwen)
        "model_at_prepare": args.model,
        "ks": [K], "cells": list(CELLS),
        "design": {
            "shape": "3x2 (delta regime x opacity)",
            "site": "computed k=1 (kc1 shape: V = o1 + o2, operands stated above)",
            "factor_A_delta": {
                "pm1": "|delta| == 1, last digit CHANGES (Yee-like)",
                "pm10_20": "last-digit-PRESERVING +/-10/20 (ours)",
                "large": ("|delta| in [100,400] inside the attainable-sum "
                          "window, last digit CHANGES (decoupler)")},
            "factor_B_opacity": {"bare": "result only (ours)",
                                 "full": "operands printed on the site line (Yee-like)"},
            "pairing": "all six cells ride ONE shared world (within-world paired)",
            "anchors": ANCHOR_PAIRS,
            "sign_convention": ("every contrast is FULL minus BARE and "
                                "<regime> minus <reference regime>; the key "
                                "names state their own direction"),
            "prediction": ("absorption collapses in recon_full_pm1 (Yee corner) and "
                           "stays ~1.0 in recon_bare_pm10_20 (ours); the other two "
                           "intermediate"),
            "large_arm_interpretation": (
                "if `large` behaves like pm10_20 the effect is delta magnitude / "
                "plausibility; if it behaves like pm1 the effect is the "
                "last-digit cue; if neither, both channels are live"),
            "separation_note": (
                "|out_cf - out_true| = 2*|delta| (out = Q1 + Q2, Q1 = V + c, "
                "Q2 = V + c'): 2 in pm1, 20/40 in pm10_20, >=200 in large. The "
                "DV is unchanged; see contrasts.separation_bias for the bound."),
        },
        "worlds_source": os.path.abspath(args.worlds),
        "worlds_admitted": kept, "worlds_offered": len(worlds),
        "prepare_rejects": dict(rejects),
        "prepare_postchecks": "recon_gen.post_checks re-run fail-closed",
        "n_target_per_cell": args.cap, "R_samples": args.reps,
        "sampling_temperature": args.temp,
        "decoding": {"greedy_rollout": True, "R_at_T": args.temp,
                     "max_new_gold": args.max_new_gold,
                     "max_new_cont": args.max_new_cont},
        "near_window": NEAR_WINDOW,
        "prompt_sha256": tf.prompt_sha256(), "gold_prefill": tf.GOLD_PREFILL,
        "flag_channel": "judge_reject if judged else doubt_lex (frozen lexicon)",
        "doubt_lexicon": val_mod.DOUBT_LEX_RE.pattern,
        "dknobs": {kk: gd.DKNOBS[kk] for kk in gd.DKNOBS},
        "attainable_window": list(rg.attainable_window()),
    }
    e11.write_json(P["metadata"], meta)
    print(json.dumps({"admitted": kept, "offered": len(worlds),
                      "rejects": dict(rejects)}, indent=2))
    print("[E14:prepare] %d worlds -> %s" % (kept, P["programs"]))


# ---------------------------------------------------------- generate (GPU)

def stage_generate(args):
    _patch_e11()
    e11.stage_generate(args)


# ---------------------------------------------------------- validate (CPU)

def stage_validate(args):
    """e11_run.stage_validate, with three additional persisted fields per row:
    `final_output_value` (the integer the model actually printed on the output
    line, which the grader already parses), `out_true`, and `out_cf`, plus the
    derived `separation` and `output_bucket`.

    The CLASSIFIER is untouched -- val_mod.classify_run and
    e11.three_way_label are called exactly as E11 calls them, and the label is
    still derived only from the E11 metric keys. The extra fields are
    diagnostics: without them a near-miss is indistinguishable from a wild miss
    after the fact, and the pm1 separation-of-2 bias cannot be bounded."""
    _patch_e11()
    P = e11.paths(args.out_dir)
    programs = {p["program_id"]: p for p in e11.read_jsonl(P["programs"])}
    manifest = e11.read_jsonl(P["manifest"])
    raw = {r["run_id"]: r for r in e11.read_jsonl(P["raw"])}
    if os.path.exists(P["validated"]):
        os.remove(P["validated"])
    n = 0
    buckets = Counter()
    for m in manifest:
        p = programs[m["program_id"]]
        stmts = parse_program(p["stmt_texts"])
        inj = {kk: m[kk] for kk in ("cf", "planted_var", "planted_value",
                                    "force_after_line", "expected_lines_from")}
        spec = p["families"][m["family"]]
        out_true = p["out_true"]
        out_cf = spec.get("out_cf")
        for roll in range(0, args.R + 1):
            rid = e11.cont_run_id(m["program_id"], m["family"], roll, m["seed"])
            r = raw.get(rid)
            rec = {"program_id": m["program_id"], "family": m["family"],
                   "k": m["k"], "opacity": m["opacity"], "rollout": roll,
                   "run_id": rid, "planted_value": m["planted_value"],
                   "true_value": m["true_value"],
                   # ---- v2 separation bookkeeping (same for every rollout) --
                   "delta_regime": spec.get("delta_regime"),
                   "abs_delta": spec.get("abs_delta"),
                   "out_true": out_true, "out_cf": out_cf,
                   "separation": (abs(out_cf - out_true)
                                  if None not in (out_cf, out_true) else None)}
            if r is None or r.get("failed_generation"):
                rec.update({"generation_failed": True, "label": "unresolved",
                            "final_output_value": None,
                            "output_bucket": "no_output_claim"})
                buckets[(m["family"], "no_output_claim")] += 1
                e11.append_jsonl(P["validated"], rec)
                n += 1
                continue
            met = val_mod.classify_run(stmts, p, inj, r["continuation"])
            rec.update({kk: met[kk] for kk in
                        ("final_output_absorbed", "final_output_valid",
                         "repair_event", "next_read_absorbed", "next_read_class",
                         "trace_valid", "doubt_lex", "doubt_broad", "unparsed",
                         "line_skip_rate")})
            # the grader already parses the output claim; persist the integer
            outc = next((c for c in met["claims"] if c["kind"] == "output"), None)
            rec["final_output_value"] = outc["value"] if outc else None
            rec["no_output_claim"] = met["no_output_claim"]
            rec["output_bucket"] = output_bucket(rec["final_output_value"],
                                                 out_true, out_cf)
            buckets[(m["family"], rec["output_bucket"])] += 1
            rec["judge_reject"] = r.get("judge_reject")  # None unless a judge ran
            rec["generation_failed"] = False
            rec["label"] = e11.three_way_label(rec)
            # cross-check: the DV must agree with the persisted integer, or the
            # two code paths have drifted apart.
            if rec.get("final_output_absorbed") is not None:
                assert bool(rec["final_output_absorbed"]) == \
                    (rec["output_bucket"] == "exact_cf"), \
                    "final_output_absorbed disagrees with final_output_value"
            e11.append_jsonl(P["validated"], rec)
            n += 1
    print("[E14] validated %d rollout-rows -> %s" % (n, P["validated"]))
    per_cell = defaultdict(dict)
    for (fam, b), c in buckets.items():
        per_cell[fam][b] = c
    print("[E14] output buckets by cell: %s"
          % json.dumps({f: per_cell.get(f, {}) for f in CELLS}, sort_keys=True))


# --------------------------------------------------------- summarize (CPU)

def _rate(rows, lab="absorbed"):
    return (sum(1 for r in rows if r["label"] == lab) / len(rows)) if rows else None


def _pooled(rows, pred):
    sel = [r for r in rows if pred(r)]
    return sel, _rate(sel)


def _r4(x):
    return round(x, 4) if x is not None else None


def _paired_diff(rows_a, rows_b, seed=0, n_boot=1000):
    """Within-world paired difference in absorbed rate (A minus B), with a
    cluster bootstrap over program_id. Uses only worlds present in BOTH cells,
    which is every world by construction."""
    def by_prog(rows):
        d = defaultdict(list)
        for r in rows:
            d[r["program_id"]].append(1 if r["label"] == "absorbed" else 0)
        return d
    A, B = by_prog(rows_a), by_prog(rows_b)
    ids = sorted(set(A) & set(B))
    if not ids:
        return {"n_paired_programs": 0, "diff": None, "boot95": [None, None]}
    per = {i: (sum(A[i]) / len(A[i])) - (sum(B[i]) / len(B[i])) for i in ids}
    point = sum(per.values()) / len(ids)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [rng.choice(ids) for _ in ids]
        vals.append(sum(per[i] for i in samp) / len(samp))
    vals.sort()
    return {"n_paired_programs": len(ids), "diff": round(point, 4),
            "boot95": [round(vals[int(0.025 * (len(vals) - 1))], 4),
                       round(vals[int(0.975 * (len(vals) - 1))], 4)]}


def separation_diagnostics(rows):
    """Per-cell (or per-pool) measurement-separation diagnostics.

    `near_miss` = final output on NEITHER target but within +/-NEAR_WINDOW of
    one of them: the observable arithmetic-slip population.

    `bias_bound` = how far the absorbed rate could move if that whole
    population were reassigned. Two numbers are reported:
      * the LITERAL reassignment asked for by the design review -- every
        near-miss row counted as absorbed (near-miss rows are never absorbed
        today: `absorbed` requires final output == out_cf exactly), giving the
        one-sided upper bound;
      * a SYMMETRIC envelope +/- near_miss_rate, which additionally covers the
        unobservable direction: a slip in a genuinely absorbing continuation
        that landed exactly on out_true and was scored as recovery. Its mass
        cannot exceed the observed slip mass under the assumption that slips
        are no more likely to hit a target exactly than to miss it by 1..3.
    """
    n = len(rows)
    if not n:
        return None
    b = Counter(r.get("output_bucket") for r in rows)
    near = sum(b.get(x, 0) for x in NEAR_BUCKETS)
    absorbed = sum(1 for r in rows if r["label"] == "absorbed")
    sep_rows = Counter(r.get("separation") for r in rows
                       if r.get("separation") is not None)
    sep_prog = {}
    for r in rows:
        if r.get("separation") is not None:
            sep_prog[r["program_id"]] = r["separation"]
    sep_prog_hist = Counter(sep_prog.values())
    a = absorbed / n
    nm = near / n
    return {
        "n_rows": n,
        "output_bucket_counts": {x: b.get(x, 0) for x in BUCKETS},
        "exact_true": b.get("exact_true", 0),
        "exact_cf": b.get("exact_cf", 0),
        "other": (b.get("other_far", 0) + b.get("no_output_claim", 0)),
        "other_breakdown": {"other_far": b.get("other_far", 0),
                            "no_output_claim": b.get("no_output_claim", 0)},
        "near_miss": {"count": near, "rate": _r4(nm),
                      "window": NEAR_WINDOW,
                      "by_bucket": {x: b.get(x, 0) for x in NEAR_BUCKETS}},
        "separation": {"rows": {str(kk): vv for kk, vv in sorted(sep_rows.items())},
                       "programs": {str(kk): vv
                                    for kk, vv in sorted(sep_prog_hist.items())},
                       "min": min(sep_rows) if sep_rows else None,
                       "max": max(sep_rows) if sep_rows else None},
        "bias_bound": {
            "definition": (
                "upper bound on the absorbed rate if EVERY near-miss row (final "
                "output within +/-%d of a target but on neither) were reassigned "
                "to the absorbed bucket; plus a symmetric envelope of the same "
                "width covering the unobservable reverse case." % NEAR_WINDOW),
            "absorbed_rate": _r4(a),
            "near_miss_rate": _r4(nm),
            "max_absorbed_rate_shift": _r4(nm),
            "absorbed_rate_if_all_near_miss_reassigned_to_absorbed": _r4(min(1.0, a + nm)),
            "absorbed_rate_symmetric_envelope": [_r4(max(0.0, a - nm)),
                                                 _r4(min(1.0, a + nm))],
        },
    }


def _regime_rows(rows, regime):
    return [r for r in rows if DELTA_OF.get(r["family"]) == regime]


def stage_summarize(args):
    P = e11.paths(args.out_dir)
    rows = e11.read_jsonl(P["validated"])
    if not rows:
        raise SystemExit("run validate first")
    meta = json.load(open(P["metadata"])) if os.path.exists(P["metadata"]) else {}
    cohort = json.load(open(P["cohort"])) if os.path.exists(P["cohort"]) else {}
    status = json.load(open(P["status"])) if os.path.exists(P["status"]) else {}

    # MODEL PROVENANCE. E11's stage_prepare never writes `model` into
    # run_metadata.json, so its stage_summarize (`meta.get("model") or
    # args.model`) silently falls back to the argparse DEFAULT when --model is
    # omitted -- which is why e11_run/results/E11_llama8b/summary_tables.json is
    # stamped "Qwen/Qwen2.5-7B-Instruct" while its queue_status.json correctly
    # says NousResearch/Meta-Llama-3.1-8B-Instruct. Here the model that actually
    # generated (recorded by stage_generate into queue_status.json) always wins,
    # and any disagreement is recorded rather than hidden.
    gen_model = status.get("model")
    model = gen_model or meta.get("model") or args.model
    model_prov = {"from_queue_status_generated_with": gen_model,
                  "from_run_metadata": meta.get("model"),
                  "from_args": args.model,
                  "resolved": model,
                  "args_disagrees_with_generated":
                      bool(gen_model and gen_model != args.model)}

    by_cell = {c: [r for r in rows if r["family"] == c] for c in CELLS}
    cells = {c: e11._cell_summary(v, args.seed) for c, v in by_cell.items() if v}
    sep_diag = {c: separation_diagnostics(v) for c, v in by_cell.items() if v}

    power = {}
    for name, c in cells.items():
        n = c["program_n"]
        power[name] = ("ok" if n >= 150 else "scored_flagged" if n >= 100 else
                       "ci_widened_flagged" if n >= 50 else "INVALID_underpowered")

    A = {c: (cells[c]["absorbed"]["rate"] if c in cells else None) for c in CELLS}

    # ---- anchor check: pm10_20 cells must reproduce E11 k1_bare / k1_full
    anchor = {"note": ("recon_*_pm10_20 site_body is byte-identical to E11 "
                       "depth_k1_{bare,full}; these rates must match E11's "
                       "k1_bare / k1_full or the build is wrong"),
              "recon_rates": {c: A[c] for c in ANCHOR_PAIRS}}
    if args.e11_summary:
        # FAIL LOUDLY. v1 silently skipped a mistyped path and swallowed every
        # parse error, so a typo produced a summary with no anchor check and no
        # sign that one was requested.
        if not os.path.exists(args.e11_summary):
            raise SystemExit("--e11-summary %r does not exist (refusing to "
                             "write a summary with a silently missing anchor "
                             "check)" % args.e11_summary)
        try:
            es = json.load(open(args.e11_summary))
        except Exception as exc:
            raise SystemExit("--e11-summary %r is not readable JSON: %r"
                             % (args.e11_summary, exc))
        ec = es.get("cells")
        if not isinstance(ec, dict):
            raise SystemExit("--e11-summary %r has no 'cells' object"
                             % args.e11_summary)
        anchor["e11_rates"] = {}
        anchor["anchor_delta_recon_minus_e11"] = {}
        for rc, e11c in ANCHOR_PAIRS.items():
            if e11c not in ec:
                raise SystemExit("--e11-summary %r lacks cell %r"
                                 % (args.e11_summary, e11c))
            ev = (ec.get(e11c) or {}).get("absorbed", {}).get("rate")
            anchor["e11_rates"][e11c] = ev
            anchor["anchor_delta_recon_minus_e11"][rc] = (
                _r4(A[rc] - ev) if None not in (A[rc], ev) else None)
        anchor["e11_summary_path"] = os.path.abspath(args.e11_summary)
        anchor["e11_model"] = es.get("model")

    # ---- main effects (pooled over the other factor) -----------------------
    # CONVENTION: every effect key is "<A>_minus_<B>" and computes rate(A) - rate(B).
    reg_rows = {t: _regime_rows(rows, t) for t in REGIMES}
    reg_rate = {t: _rate(reg_rows[t]) for t in REGIMES}
    me_delta = {"convention": "effect_<A>_minus_<B> == rate(A) - rate(B)",
                "rate_by_regime": {t: _r4(reg_rate[t]) for t in REGIMES}}
    for a_, b_ in REGIME_CONTRASTS:
        key = "effect_%s_minus_%s" % (a_, b_)
        me_delta[key] = (_r4(reg_rate[a_] - reg_rate[b_])
                         if None not in (reg_rate[a_], reg_rate[b_]) else None)
        me_delta["paired_%s_minus_%s" % (a_, b_)] = _paired_diff(
            reg_rows[a_], reg_rows[b_], args.seed)

    bare_rows, bare_r = _pooled(rows, lambda r: OPACITY_OF.get(r["family"]) == "bare")
    full_rows, full_r = _pooled(rows, lambda r: OPACITY_OF.get(r["family"]) == "full")
    me_opac = {"convention": "effect_full_minus_bare == rate(full) - rate(bare)",
               "bare": _r4(bare_r), "full": _r4(full_r),
               "effect_full_minus_bare": (_r4(full_r - bare_r)
                                          if None not in (bare_r, full_r) else None),
               "paired_full_minus_bare": _paired_diff(full_rows, bare_rows, args.seed)}

    # ---- 3x2 interaction --------------------------------------------------
    # v1 BUG: `opacity_effect_within_*` was FULL-minus-BARE while the adjacent
    # `difference_in_differences` was (bare-minus-full) differenced the other
    # way, so two neighbouring keys carried opposite signs. Every key below is
    # FULL minus BARE, then <regime> minus <reference regime>, and says so.
    opac_eff = {}
    for t in REGIMES:
        f_, b_ = A["recon_full_%s" % t], A["recon_bare_%s" % t]
        opac_eff[t] = _r4(f_ - b_) if None not in (f_, b_) else None
    inter = {"convention": ("FULL minus BARE within a regime; then "
                            "<regime> minus <reference regime> across regimes"),
             "cell_rates": A,
             "opacity_effect_full_minus_bare_by_regime": opac_eff}
    for a_, b_ in REGIME_CONTRASTS:
        inter["did_%s_minus_%s_of_full_minus_bare" % (a_, b_)] = (
            _r4(opac_eff[a_] - opac_eff[b_])
            if None not in (opac_eff[a_], opac_eff[b_]) else None)

    # ---- corner contrasts --------------------------------------------------
    corner = {
        "convention": ("<A>_minus_<B> == rate(A) - rate(B); the canonical "
                       "direction is FULL/pm1 minus BARE/pm10_20, and "
                       "gap_our_minus_yee_corner is its negation, named "
                       "explicitly because the narrative quotes it positive"),
        "cells": {"yee_corner": "recon_full_pm1",
                  "our_corner": "recon_bare_pm10_20",
                  "large_full_corner": "recon_full_large",
                  "large_bare_corner": "recon_bare_large"},
        "rates": {"yee_corner": A["recon_full_pm1"],
                  "our_corner": A["recon_bare_pm10_20"],
                  "large_full_corner": A["recon_full_large"],
                  "large_bare_corner": A["recon_bare_large"]},
    }

    def _d(x, y):
        return _r4(A[x] - A[y]) if None not in (A[x], A[y]) else None

    corner["yee_corner_minus_our_corner"] = _d("recon_full_pm1", "recon_bare_pm10_20")
    corner["gap_our_minus_yee_corner"] = _d("recon_bare_pm10_20", "recon_full_pm1")
    corner["large_full_minus_our_corner"] = _d("recon_full_large", "recon_bare_pm10_20")
    corner["large_full_minus_yee_corner"] = _d("recon_full_large", "recon_full_pm1")
    corner["paired_yee_minus_our_corner"] = _paired_diff(
        by_cell["recon_full_pm1"], by_cell["recon_bare_pm10_20"], args.seed)
    corner["paired_our_minus_yee_corner"] = _paired_diff(
        by_cell["recon_bare_pm10_20"], by_cell["recon_full_pm1"], args.seed)
    # Yee's reported recovery band, for reference only (not a fitted quantity)
    corner["yee_reported_recovery_band"] = [0.74, 0.79]
    corner["yee_audited_recovery_band"] = [0.68, 0.71]
    if A["recon_full_pm1"] is not None:
        corner["recon_full_pm1_recovery_proxy_1_minus_absorbed"] = \
            _r4(1 - A["recon_full_pm1"])

    # ---- separation bias ---------------------------------------------------
    # The number the paper quotes when asserting that a regime contrast is not
    # an artifact of |out_cf - out_true| differing across regimes.
    reg_diag = {t: separation_diagnostics(reg_rows[t]) for t in REGIMES
                if reg_rows[t]}
    sep_bias = {
        "why": ("out = Q1 + Q2 with Q1 = V + c and Q2 = V + c', so "
                "|out_cf - out_true| = 2*|delta|. pm1 separates the absorbed "
                "and corrected answers by only 2, so an arithmetic slip in an "
                "absorbing continuation can land on out_true and be scored as "
                "recovery. This bounds that."),
        "near_window": NEAR_WINDOW,
        "per_cell": {c: {"separation": sep_diag[c]["separation"],
                         "near_miss": sep_diag[c]["near_miss"],
                         "exact_true": sep_diag[c]["exact_true"],
                         "exact_cf": sep_diag[c]["exact_cf"],
                         "other": sep_diag[c]["other"],
                         "bias_bound": sep_diag[c]["bias_bound"]}
                     for c in sep_diag},
        "per_regime": {t: {"separation": reg_diag[t]["separation"],
                           "near_miss": reg_diag[t]["near_miss"],
                           "bias_bound": reg_diag[t]["bias_bound"]}
                       for t in reg_diag},
    }
    rc = {}
    for a_, b_ in REGIME_CONTRASTS:
        if a_ not in reg_diag or b_ not in reg_diag:
            continue
        obs = me_delta.get("effect_%s_minus_%s" % (a_, b_))
        # worst case: both regimes' near-miss populations move in OPPOSITE
        # directions, so the bound on the contrast is the sum of the two
        # per-regime bounds.
        bnd = (reg_diag[a_]["bias_bound"]["max_absorbed_rate_shift"]
               + reg_diag[b_]["bias_bound"]["max_absorbed_rate_shift"])
        rc["%s_minus_%s" % (a_, b_)] = {
            "observed_effect": obs,
            "bias_bound_on_contrast": _r4(bnd),
            "contrast_exceeds_bias_bound":
                (abs(obs) > bnd if obs is not None else None),
            "worst_case_interval": ([_r4(obs - bnd), _r4(obs + bnd)]
                                    if obs is not None else None)}
    sep_bias["regime_contrast_bias_bounds"] = rc

    # ---- prediction check, stated RELATIVE to the model's own E11 anchor.
    # Absolute thresholds are deliberately avoided: E11's k1_bare absorbed rate
    # is strongly model-dependent (0.7844 on Qwen2.5-7B vs 0.9859 on
    # Llama-3.1-8B), so "stays ~1.0 in our corner" must be read against that
    # model's own anchor, not against 1.0, or a correct Qwen build looks failed.
    e11_bare = (anchor.get("e11_rates") or {}).get("k1_bare")
    ours, yee = A["recon_bare_pm10_20"], A["recon_full_pm1"]
    pred = {"statement": (meta.get("design", {}) or {}).get("prediction"),
            "our_corner_absorbed": ours, "yee_corner_absorbed": yee,
            "e11_k1_bare_reference": e11_bare,
            "collapse_in_yee_corner_vs_our_corner":
                (_r4(ours - yee) >= 0.15 if None not in (ours, yee) else None),
            "our_corner_matches_e11_anchor_within_0p05":
                (abs(ours - e11_bare) <= 0.05
                 if None not in (ours, e11_bare) else None),
            "threshold_note": ("relative, not absolute: E11 k1_bare absorbed is "
                               "0.7844 (Qwen2.5-7B) / 0.9859 (Llama-3.1-8B); "
                               "pass --e11-summary to populate the reference"),
            "others_intermediate": (
                None if any(A[c] is None for c in
                            ("recon_bare_pm1", "recon_full_pm10_20",
                             "recon_bare_pm10_20", "recon_full_pm1")) else
                bool(min(A["recon_bare_pm1"], A["recon_full_pm10_20"])
                     >= A["recon_full_pm1"] - 1e-9
                     and max(A["recon_bare_pm1"], A["recon_full_pm10_20"])
                     <= A["recon_bare_pm10_20"] + 1e-9))}

    # ---- the decoupling read the `large` arm exists for -------------------
    d_lp10 = me_delta.get("effect_large_minus_pm10_20")
    d_lp1 = me_delta.get("effect_large_minus_pm1")
    verdict = None
    if None not in (d_lp10, d_lp1):
        if abs(d_lp10) < abs(d_lp1):
            verdict = ("large_resembles_pm10_20 -> the driver is delta "
                       "MAGNITUDE / PLAUSIBILITY, not the last-digit cue")
        elif abs(d_lp1) < abs(d_lp10):
            verdict = ("large_resembles_pm1 -> the driver is the LAST-DIGIT "
                       "CUE, not delta magnitude")
        else:
            verdict = "large_equidistant -> both channels loaded; report both"
    decouple = {
        "why": ("in v1 |delta| was perfectly collinear with the unit-digit cue "
                "(pm10_20 preserved it 1800/1800, pm1 changed it 1800/1800); "
                "`large` changes the unit digit at a LARGE magnitude, so the "
                "two channels separate"),
        "rate_by_regime": me_delta["rate_by_regime"],
        "effect_large_minus_pm10_20": d_lp10,
        "effect_large_minus_pm1": d_lp1,
        "verdict_heuristic": verdict,
        "caveat": ("this is a point-estimate heuristic; read it against "
                   "main_effect_delta.paired_* CIs and the separation bias "
                   "bounds before quoting it"),
    }

    summary = {
        "experiment": "E14_RECON_YEE", "created_at": e11.now_iso(),
        "design_shape": "3x2", "cells_list": list(CELLS),
        "model": model, "model_provenance": model_prov,
        "cells": cells, "power_flags": power,
        "separation_diagnostics": sep_diag,
        "gold_cohort": cohort.get("by_k"),
        "contrasts": {"anchor_check": anchor,
                      "main_effect_delta": me_delta,
                      "main_effect_opacity": me_opac,
                      "interaction_3x2": inter,
                      "corner_contrast": corner,
                      "separation_bias": sep_bias,
                      "decoupling_read": decouple,
                      "prediction_check": pred},
        "design": meta.get("design"),
        "flag_channel": meta.get("flag_channel"),
        "stats_note": ("Wilson95 + program-cluster bootstrap95 (n_boot=1000); "
                       "cluster=program family (program_id). Paired diffs are "
                       "within-world (all 6 cells share one world) with a "
                       "cluster bootstrap over program_id. Every contrast key "
                       "names its own direction: FULL minus BARE, <A> minus <B>."),
    }
    e11.write_json(P["summary"], summary)
    if model_prov["args_disagrees_with_generated"]:
        print("[E14][WARN] --model=%r but these outputs were generated with %r; "
              "using the latter." % (args.model, gen_model))
    print(json.dumps({"model": model, "model_provenance": model_prov,
                      "cell_absorbed": A, "power_flags": power,
                      "contrasts": summary["contrasts"]}, indent=2, default=str))
    print("[E14] summary -> %s" % P["summary"])


# ------------------------------------------------------------ report (CPU)

def stage_report(args):
    P = e11.paths(args.out_dir)
    s = json.load(open(P["summary"]))
    L = []
    ap = L.append
    ap("# E14 RECON (Yee reconciliation) report -- %s" % s.get("model"))
    ap("")
    ap("Generated %s. Flag channel: %s." % (s["created_at"], s.get("flag_channel")))
    ap("")
    ap("3x2 on the COMPUTED k=1 site: delta regime (pm1 = Yee-like | pm10_20 =")
    ap("ours | large = last-digit decoupler) x opacity (full = operands printed,")
    ap("Yee-like | bare = result only, ours).")
    ap("All six cells ride ONE shared world -> within-world paired.")
    ap("recon_*_pm10_20 are byte-identical E11 anchors (k1_bare / k1_full).")
    ap("Sign convention: every contrast is FULL minus BARE and <A> minus <B>,")
    ap("stated in the key name.")
    ap("")
    ap("## Cells (3-way DV; absorbed rate is the headline)")
    ap("")
    cols = ["cell", "n", "prog_n", "power", "absorbed", "silently_corr",
            "flagged", "unresolved"]
    ap("| " + " | ".join(cols) + " |")
    ap("|" + "---|" * len(cols))
    for name in CELLS:
        c = s["cells"].get(name)
        if not c:
            continue

        def r(x, _c=c):
            m = _c[x]
            lo, hi = m["wilson95"]
            ci = " [%.2f,%.2f]" % (lo, hi) if lo is not None else ""
            return "%.3f%s" % (m["rate"], ci) if m["rate"] is not None else "n.m."
        ap("| %s | %d | %d | %s | %s | %s | %s | %s |" %
           (name, c["n"], c["program_n"], s["power_flags"].get(name, "?"),
            r("absorbed"), r("silently_corrected"), r("flagged"), r("unresolved")))
    ap("")
    ap("## Measurement separation (|out_cf - out_true| = 2*|delta|)")
    ap("")
    scols = ["cell", "sep(min..max)", "exact_cf", "exact_true", "near_miss",
             "other", "absorbed", "bias_bound(+/-)", "envelope"]
    ap("| " + " | ".join(scols) + " |")
    ap("|" + "---|" * len(scols))
    for name in CELLS:
        d = (s.get("separation_diagnostics") or {}).get(name)
        if not d:
            continue
        bb = d["bias_bound"]
        ap("| %s | %s..%s | %d | %d | %d (%.3f) | %d | %.3f | %.3f | [%.3f,%.3f] |"
           % (name, d["separation"]["min"], d["separation"]["max"],
              d["exact_cf"], d["exact_true"], d["near_miss"]["count"],
              d["near_miss"]["rate"] or 0.0, d["other"],
              bb["absorbed_rate"] or 0.0, bb["max_absorbed_rate_shift"] or 0.0,
              bb["absorbed_rate_symmetric_envelope"][0] or 0.0,
              bb["absorbed_rate_symmetric_envelope"][1] or 0.0))
    ap("")
    ap("`near_miss` = final output on neither target but within +/-%s of one."
       % (s["contrasts"].get("separation_bias") or {}).get("near_window"))
    ap("`bias_bound` = max shift in the absorbed rate if that whole population")
    ap("were reassigned. Regime-level bounds are in")
    ap("`contrasts.separation_bias.regime_contrast_bias_bounds`.")
    ap("")
    ap("## Contrasts")
    ap("")
    ap("```json")
    ap(json.dumps(s["contrasts"], indent=2, sort_keys=True, default=str))
    ap("```")
    ap("")
    ap("Gold cohort: `%s`" % json.dumps(s.get("gold_cohort")))
    ap("")
    ap("Notes: %s" % s["stats_note"])
    with open(P["report"], "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[E14] report -> %s" % P["report"])


# ----------------------------------------------------------------- cli

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E14 recon runner (Yee reconciliation)")
    p.add_argument("stage", choices=["prepare", "generate", "validate",
                                     "summarize", "report"])
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES value, applied in main()")
    p.add_argument("--cap", type=int, default=150,
                   help="eligible worlds kept per cell after the gold gate")
    p.add_argument("--reps", type=int, default=8,
                   help="sampled rollouts at T>0, in addition to greedy")
    p.add_argument("--out-dir", default=None,
                   help="default $OUTBASE/EXPG_RECON_<model basename>")
    p.add_argument("--worlds", default=WORLDS_DEFAULT)
    p.add_argument("--outbase", default=OUTBASE_DEFAULT)
    p.add_argument("--e11-summary", default=None,
                   help="E11 summary_tables.json for the anchor delta check; a "
                        "path that does not exist is a HARD ERROR")
    p.add_argument("--revision", default=None)
    p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-new-gold", type=int, default=512)
    p.add_argument("--max-new-cont", type=int, default=384)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.out_dir is None:
        a.out_dir = os.path.join(a.outbase,
                                 "EXPG_RECON_%s" % a.model.split("/")[-1])
    if a.revision is None:
        a.revision = e11.PINNED.get(a.model)
    # e11_run.stage_generate reads these names
    a.n_target = a.cap
    a.R = a.reps
    a.world_pool = None
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    {"prepare": stage_prepare, "generate": stage_generate,
     "validate": stage_validate, "summarize": stage_summarize,
     "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
