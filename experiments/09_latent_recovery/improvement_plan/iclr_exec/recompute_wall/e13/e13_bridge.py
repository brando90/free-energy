"""E13c bridge ATTRIBUTION arms (the provenance fix), copy-derived from
e10_api_bridge/e10bridge.py WITHOUT modifying it.

Vary ONLY the attribution sentence of the bridge template; every other byte of
the user message (instruction, exemplars, program, partial trace, continuation
request) is identical across arms. Arms (sentences pinned below, sha256 in the
run manifest BEFORE generation):

  baseline     unattributed -- template byte-identical to
               e10bridge.bridge_user_message (asserted at import parity)
  self         "You wrote the following partial trace earlier."
  other_model  "Another AI model wrote the following partial trace."
  human        "A person wrote the following partial trace."

Template note (disclosed for Review): the baseline first sentence carries the
locator clause 'it is shown below under "Trace:"' inside one sentence; the
three attributed arms use the pinned spec sentence followed by the constant
locator sentence 'It is shown below under "Trace:".' so the locator is held
constant and ONLY the attribution proposition varies. Exact bytes + sha256
per arm in the manifest.

Cells: onehop_kc1, adjacent_contradiction, opfree_kr1 on the EXISTING regime2
worlds (same programs.jsonl as e10bridge; provenance fingerprint written).
Models: claude-opus-4-8_bridge, claude-sonnet-5_bridge, gpt-4.1_bridge
(e10bridge MODELS configs reused by import -- thinking/temperature policy
unchanged). Golds are REUSED READ-ONLY from e10_api_bridge raw_gold_*.jsonl
(gold prompts contain no partial trace, hence no attribution sentence; arm-
independent). n = first --cell-cap eligible worlds per cell (default 70).

Scoring: EXPG validate.classify_run (existing path) + 3-way summary with
Wilson + program-cluster bootstrap. --dry assembles prompts from the real
stored golds and writes examples + per-arm spend projection; ZERO API calls.

Usage:
  python e13_bridge.py --dry --out-dir results/E13_bridge
  source <keys env> && python e13_bridge.py --out-dir results/E13_bridge \
      --models claude-opus-4-8_bridge,claude-sonnet-5_bridge,gpt-4.1_bridge \
      --arms baseline,self,other_model,human
"""
import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e13_lib as L                        # noqa: E402
import e11_run as e11                      # noqa: E402
import trace_format as tf                  # noqa: E402
import inject as inj_mod                   # noqa: E402
import validate as val_mod                 # noqa: E402
from interp import parse_program           # noqa: E402

E10B_DIR = os.path.join(L.RW, "e10_api_bridge")
sys.path.insert(0, E10B_DIR)
import e10bridge as B                      # noqa: E402  (read-only reuse)

try:
    import api_gen as A                    # noqa: E402
except Exception:
    A = None
import exph_common as C                    # noqa: E402

SEED = 0
MAX_CONT = 384
CELLS = ("onehop_kc1", "adjacent_contradiction", "opfree_kr1")
DEFAULT_MODELS = ("claude-opus-4-8_bridge", "claude-sonnet-5_bridge",
                  "gpt-4.1_bridge")

LOCATOR = 'It is shown below under "Trace:".'
BASELINE_SENT = ('A partial trace of this program has already been written; '
                 'it is shown below under "Trace:".')
ATTR_SENTS = {
    "baseline": BASELINE_SENT,
    "self": 'You wrote the following partial trace earlier. ' + LOCATOR,
    "other_model": 'Another AI model wrote the following partial trace. ' + LOCATOR,
    "human": 'A person wrote the following partial trace. ' + LOCATOR,
}
ARM_ORDER = ("baseline", "self", "other_model", "human")


def attributed_user_message(arm, listing_text, prefix_text):
    return (tf.INSTRUCTION + tf.EXEMPLARS + "Program:\n" + listing_text +
            "\n\n" + ATTR_SENTS[arm] + " Continue it. Write only the remaining "
            "trace lines, in order, in the exact same format, picking up "
            "immediately after the last line shown. Do not repeat lines that "
            "are already written.\n\nTrace:\n" + prefix_text.rstrip("\n"))


def _assert_baseline_parity():
    lst, pfx = "1: a = 1", "line 1: a = 1; a = 1\n"
    ours = attributed_user_message("baseline", lst, pfx)
    orig = B.bridge_user_message(lst, pfx)
    if ours != orig:
        raise SystemExit("baseline template NOT byte-identical to "
                         "e10bridge.bridge_user_message -- refusing to run")


def rid(arm, model_id, cell, pid):
    return C.sha_row(["E13C", arm, model_id, cell, pid, SEED])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--cell-cap", type=int, default=70)
    ap.add_argument("--usd-cap", type=float, default=35.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    _assert_baseline_parity()
    models = [m for m in a.models.split(",") if m]
    arms = [x for x in a.arms.split(",") if x]
    for arm in arms:
        if arm not in ATTR_SENTS:
            raise SystemExit("unknown arm %s" % arm)
    os.makedirs(a.out_dir, exist_ok=True)

    progs = B.programs()
    prog_by = {p["program_id"]: p for p in progs}
    fp = C.sha_row([p["program_id"] for p in progs])

    man = {
        "experiment": "E13C_BRIDGE_ATTRIBUTION", "created_at": e11.now_iso(),
        "arms": {arm: {"sentence_verbatim": ATTR_SENTS[arm],
                       "sha256": L.sha256_text(ATTR_SENTS[arm])}
                 for arm in ARM_ORDER},
        "template_note": ("baseline byte-identical to e10bridge."
                          "bridge_user_message (asserted); attributed arms = "
                          "pinned spec sentence + constant locator sentence; "
                          "ONLY the attribution proposition varies"),
        "models": models, "cells": list(CELLS), "cell_cap": a.cell_cap,
        "seed": SEED, "dry": bool(a.dry),
        "worlds": {"source": B.PROG_SRC, "n_programs": len(progs),
                   "program_id_fingerprint": fp,
                   "file_sha256": L.sha256_file(B.PROG_SRC)},
        "gold_policy": ("golds REUSED READ-ONLY from e10_api_bridge "
                        "raw_gold_<model>.jsonl (own-trace, 2026-07-27); gold "
                        "prompts carry no partial trace so they are "
                        "attribution-arm-independent"),
        "scoring": "validate.classify_run (existing EXPG 2-world path)",
        "prompt_sha256": tf.prompt_sha256(),
    }
    e11.write_json(os.path.join(a.out_dir, "run_manifest.json"), man)

    led = None
    if not a.dry:
        if A is None:
            raise SystemExit("api_gen not importable; live arm needs it")
        need = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
        for mk in models:
            prov = B.get_cfg(mk)["provider"]
            if not os.environ.get(need[prov]):
                raise SystemExit("%s not set (needed for %s). Source the keys "
                                 "env file first." % (need[prov], mk))
        led = A.CostLedger(os.path.join(a.out_dir, "cost_ledger.json"),
                           extra_paths=L.all_ledgers(
                               os.path.join(a.out_dir, "cost_ledger.json")))

    examples = defaultdict(list)
    projection = {}
    for mk in models:
        cfg = B.get_cfg(mk)
        model_id = cfg["model_id"]
        # ---- eligible worlds from the EXISTING e10bridge gold cohort
        raw_gold_path, coh_path = B.gold_paths(cfg)
        if not (os.path.exists(raw_gold_path) and os.path.exists(coh_path)):
            raise SystemExit("missing e10bridge golds for %s" % model_id)
        cohort = json.load(open(coh_path))
        gold_raw = {r["program_id"]: r for r in C.read_jsonl(raw_gold_path)}
        elig = defaultdict(list)
        for r in cohort["per_program"]:
            if r["eligible"]:
                elig[r["shape"]].append(r["program_id"])

        # ---- manifest rows (arm x cell x world), injection fail-closed
        man_path = os.path.join(a.out_dir, "manifest_%s.jsonl" % model_id)
        audit_path = os.path.join(a.out_dir, "audit_%s.jsonl" % model_id)
        mrows = C.read_jsonl(man_path)
        if not mrows:
            for cell in CELLS:
                shape = B.CELL_SHAPE[cell]
                for pid in elig[shape][:a.cell_cap]:
                    p = prog_by[pid]
                    gfull = B.gold_full_text(cfg,
                                             gold_raw[pid]["continuation"])
                    try:
                        b = inj_mod.build_injection(p, cell, gfull)
                    except inj_mod.InjectError as e:
                        C.append_jsonl(audit_path, {
                            "stage": "injection", "cell": cell,
                            "program_id": pid, "rejected": True,
                            "reason": e.reason})
                        continue
                    for arm in arms:
                        row = {"run_id": rid(arm, model_id, cell, pid),
                               "arm": arm, "cell": cell, "condition": cell,
                               "program_id": pid, "shape": shape,
                               "model_id": model_id, "model_key": mk,
                               "mode": cfg["mode"], "seed": SEED,
                               "created_at": B.now_iso()}
                        row.update({k: b[k] for k in (
                            "family", "injected_trace_line",
                            "original_trace_line", "prefix_text",
                            "expected_lines_from", "cf", "planted_var",
                            "planted_value", "true_value",
                            "force_after_line", "delta_policy")})
                        mrows.append(row)
            C.write_jsonl(man_path, mrows)

        # ---- prompts + projection + examples
        jobs = []
        jobs_by_arm = defaultdict(list)
        for m in mrows:
            if m["arm"] not in arms:
                continue
            p = prog_by[m["program_id"]]
            listing = tf.make_listing_text(p["stmt_texts"])
            um = attributed_user_message(m["arm"], listing, m["prefix_text"])
            payload = [{"role": "user", "content": um}]
            base = {"run_id": m["run_id"], "arm": m["arm"], "cell": m["cell"],
                    "condition": m["cell"], "program_id": m["program_id"],
                    "shape": m["shape"], "model_id": model_id}
            jobs.append((base, payload))
            jobs_by_arm[m["arm"]].append((L.est_tokens(um), 1))
            ek = "%s__%s" % (model_id, m["arm"])
            if len(examples[ek]) < 2:
                examples[ek].append({"program_id": m["program_id"],
                                     "cell": m["cell"],
                                     "api_messages": payload})
        pin, pout = A.PRICES.get(model_id, (5.0, 25.0)) if A else (5.0, 25.0)
        projection[model_id] = L.project_spend(
            "e13_bridge %s" % model_id, jobs_by_arm, pin, pout, 300)
        print("[e13_bridge] %s manifest: %d rows by arm x cell = %s"
              % (model_id, len([m for m in mrows if m["arm"] in arms]),
                 json.dumps(dict(Counter("%s|%s" % (m["arm"], m["cell"])
                                         for m in mrows
                                         if m["arm"] in arms)))))
        if a.dry:
            continue

        # ---- live generation (resume-safe on run_id), then validate
        raw_path = os.path.join(a.out_dir, "raw_pert_%s.jsonl" % model_id)
        try:
            led.check_budget()
        except A.BudgetExceeded as e:
            print("[e13_bridge] %s" % e)
            break
        if led.cost_usd() > a.usd_cap:
            print("[e13_bridge] soft cap hit ($%.2f > $%.2f); stopping"
                  % (led.cost_usd(), a.usd_cap))
            break
        B.run_jobs(cfg, jobs, led, raw_path, MAX_CONT)

        raw = {r["run_id"]: r for r in C.read_jsonl(raw_path)}
        out_rows = []
        for m in mrows:
            if m["arm"] not in arms:
                continue
            p = prog_by[m["program_id"]]
            r = raw.get(m["run_id"])
            rec = dict(m)
            if r is None or r.get("failed_generation"):
                rec.update({"generation_failed": True,
                            "class": "generation_failed"})
                out_rows.append(rec)
                continue
            inj = {k: m[k] for k in ("cf", "planted_var", "planted_value",
                                     "force_after_line",
                                     "expected_lines_from")}
            met = val_mod.classify_run(parse_program(p["stmt_texts"]), p, inj,
                                       r["continuation"])
            rec.update(met)
            rec["generation_failed"] = False
            rec["continuation"] = r["continuation"]
            out_rows.append(rec)
        C.write_jsonl(os.path.join(a.out_dir,
                                   "validated_%s.jsonl" % model_id), out_rows)
        led.save()

    L.dump_examples(os.path.join(a.out_dir, "example_prompts.json"),
                    dict(examples))
    e11.write_json(os.path.join(a.out_dir, "spend_projection.json"),
                   {"per_model": projection,
                    "total_usd_est": round(sum(
                        v["total_usd_est"] for v in projection.values()), 2)})
    if a.dry:
        print("[e13_bridge:dry] DONE (zero API calls made)")
        return
    summarize(a.out_dir, models, arms)


def summarize(out_dir, models, arms):
    out = {"created_at": B.now_iso(), "models": {}, "arms": arms,
           "cells": list(CELLS)}
    for mk in models:
        model_id = B.get_cfg(mk)["model_id"]
        path = os.path.join(out_dir, "validated_%s.jsonl" % model_id)
        if not os.path.exists(path):
            continue
        rows = C.read_jsonl(path)
        per = {}
        for arm in arms:
            for cell in CELLS:
                sub = [r for r in rows
                       if r["arm"] == arm and r["cell"] == cell]
                if not sub:
                    continue
                live = [r for r in sub if not r.get("generation_failed")]
                absorbed = [bool(r.get("final_output_absorbed")) for r in live]
                k, n = sum(absorbed), len(live)
                per["%s|%s" % (arm, cell)] = {
                    "n": len(sub), "n_live": n,
                    "program_n": len({r["program_id"] for r in sub}),
                    "absorbed": {
                        "count": k, "rate": round(k / n, 4) if n else None,
                        "wilson95": e11.wilson(k, n),
                        "cluster_boot95": e11.cluster_boot_rate(
                            live, lambda r: bool(r.get("final_output_absorbed")))},
                    "flagged_doubt_lex": round(
                        sum(bool(r.get("doubt_lex")) for r in live) / n, 4)
                        if n else None,
                    "silently_corrected": round(
                        sum(1 for r in live
                            if not r.get("doubt_lex")
                            and not r.get("final_output_absorbed")
                            and r.get("final_output_valid")) / n, 4)
                        if n else None}
        out["models"][model_id] = per
    e11.write_json(os.path.join(out_dir, "summary_attribution.json"), out)
    print("[e13_bridge] summary -> summary_attribution.json")


if __name__ == "__main__":
    main()
