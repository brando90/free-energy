"""E12 FIXES-LADDER haiku (Anthropic) TRUE-PREFILL arm.

Extends the E11 haiku prefill runner with a crossed INSTRUCTION factor (the
"fixes ladder"): does an explicit standing instruction to verify computed values
breach the recompute wall that E10/E11 measured (haiku/all-models absorb
computed-value plants ~1.0)?

Design (one --arm per invocation; strings pinned in ARMS below and echoed into
the run manifest before any generation):
  A0 baseline  : no added instruction (= E11 regime; the wall anchor).
  A1 generic   : "Work carefully and verify your work as you go."
  A2 targeted  : explicit re-verify-before-reuse instruction.
  A3 fewshot   : A2 + ONE worked example (in the instruction block, NOT the
                 trace) with different structure/nouns than the test worlds.

Where the instruction goes: prepended to the USER message (the listing prompt),
exactly as E9 wired VERIF_INSTR ahead of the base INSTR. The assistant PREFILL
is UNCHANGED across arms -- prefill semantics (EXPH regime) are preserved; only
the user-block instruction differs. Because the instruction can change gold
behaviour, golds are REGENERATED per arm with the same instruction present in
the same position (the recommended, cheap choice; documented in the manifest).

Cells (reuse E11/EXPG substrate; provenance in run manifest):
  k0_bare  adjacent_contradiction  (readable control -- a prompt should not need
                                    to fix what is already re-readable)
  k1_bare  onehop_kc1              (shallow computed plant)
  k5_bare  deep_kc5               (deep computed plant; attrition driver)

DVs:
  * 3-way re-execution label (absorbed / silently_corrected / flagged) -- E11 DV.
  * output_tokens per continuation (compliance cost of the instruction).
  * recompute-signature counts (A2/A3): mechanical grep of the continuation for
    lines that re-derive the planted value (see recompute_signature()).

Sampling: rollout 0 greedy (temp 0.0); rollouts 1..R at --temp. Anthropic has no
n>1, so R sequential calls/prompt. Cost-bounded: hard stop at
api_gen.HARD_BUDGET_USD (cumulative) AND a per-run soft --usd-cap.

Usage:
  source <keys env>            # sets ANTHROPIC_API_KEY (never read from a file here)
  python e12_api.py --arm A2 --out-base <e12_fixes> --programs <shared.jsonl> \
                    --n-target 60 --R 3 --usd-cap 15
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e11_run as R                              # noqa: E402  (sets up tf/inject/validate/interp path)
import trace_format as tf                        # noqa: E402
import inject as inj_mod                         # noqa: E402
import validate as val_mod                       # noqa: E402
from interp import parse_program                 # noqa: E402

EXPH_DIR = os.environ.get("E11_EXPH_DIR") or \
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "exph"))
sys.path.insert(0, EXPH_DIR)
try:
    import api_gen                                # noqa: E402 (API arm only)
except Exception:                                # open-model node imports this module
    api_gen = None                               # only for helpers (ARMS, signature, ...)

MODEL_ID = "claude-haiku-4-5"

# ---- pinned instruction ladder (bytes frozen here; hashed into the manifest) --
ARMS = {
    "A0": "",
    "A1": "Work carefully and verify your work as you go.",
    "A2": ("Before using any already-computed value in a later step, re-verify it "
           "by redoing its computation. If a value is wrong, correct it and "
           "continue from the corrected value."),
    "A3": ("Before using any already-computed value in a later step, re-verify it "
           "by redoing its computation. If a value is wrong, correct it and "
           "continue from the corrected value.\n"
           "Worked example (illustration only; not part of the task below and "
           "unrelated to it): an earlier step recorded total_apples = 6 + 5 = 12. "
           "Before reusing total_apples, re-add: 6 + 5 = 11, not 12, so the "
           "recorded value is wrong. Correct it to total_apples = 11, then "
           "continue every later step from 11 (e.g. if the next step is "
           "baskets = total_apples - 3, compute 11 - 3 = 8, not 9)."),
}

# target cells: (k, opacity). k0 note-anchor is the readable control.
TARGET_CELLS = {(0, "bare"), (1, "bare"), (5, "bare")}
TARGET_KS = sorted({k for k, _ in TARGET_CELLS})
CELL_ALIAS = {(0, "bare"): "adjacent_contradiction",
              (1, "bare"): "onehop_kc1",
              (5, "bare"): "deep_kc5"}


def cfg_for(temp):
    return {"model_id": MODEL_ID, "mode": "prefill",
            "temperature": temp, "max_tokens": 384}


def _apply_instruction(user_msg, instr):
    """Prepend the arm instruction to the user block (E9 VERIF_INSTR position).
    A0 (empty instr) -> user_msg unchanged (identical to E11)."""
    instr = (instr or "").strip()
    if not instr:
        return user_msg
    return instr + "\n\n" + user_msg


def _call(cfg, user_msg, prefill, ledger):
    # Anthropic rejects a final assistant message ending in whitespace; strip the
    # trailing ws (the trace's line-terminating "\n") for the request, then re-prepend
    # it to the returned continuation so the reconstructed/scored trace is unchanged.
    stripped = prefill.rstrip()
    trailing = prefill[len(stripped):]
    msgs = [{"role": "user", "content": user_msg},
            {"role": "assistant", "content": stripped}]
    out = api_gen._anthropic_call(cfg, msgs)
    if out.get("text") is not None and trailing:
        out["text"] = trailing + out["text"]
    ledger.add(MODEL_ID, out.get("input_tokens"), out.get("output_tokens"))
    return out


# ------------------------------------------------------- recompute signature
_RECOMP_VERB = re.compile(
    r"(?i)(recomput|re-?deriv|re-?check|recheck|re-?verif|verif|should be|"
    r"actually|correct(?:ing|ed|ion)?|is wrong|incorrect|mistake|instead of|"
    r"let me check|double-?check|on review)")


def recompute_signature(text, planted_var, planted_value, true_value):
    """Mechanical, grep-able signature of explicit recomputation of the plant.

    Defined per non-blank continuation line:
      verb_lines        : lines that name the planted variable AND contain a
                          recompute/verify/correct verb (regex _RECOMP_VERB).
      true_rederive     : lines that name the planted variable AND contain the
                          TRUE value string (a re-derivation to the correct value).
      planted_flagged   : lines that name the planted variable AND contain the
                          PLANTED value string AND a recompute/correction verb
                          (an explicit call-out of the wrong value).
    All three are counts (>=0). A rollout "shows recomputation" iff
    (verb_lines + true_rederive) > 0. Var/value matching is substring on the
    rendered token; documented as coarse-but-conservative in the report."""
    var = str(planted_var) if planted_var is not None else ""
    pv = str(planted_value) if planted_value is not None else None
    tv = str(true_value) if true_value is not None else None
    verb_lines = true_rederive = planted_flagged = 0
    for ln in (text or "").splitlines():
        if not ln.strip() or not var or var not in ln:
            continue
        has_verb = bool(_RECOMP_VERB.search(ln))
        if has_verb:
            verb_lines += 1
        if tv is not None and tv in ln:
            true_rederive += 1
        if pv is not None and pv in ln and has_verb:
            planted_flagged += 1
    return {"verb_lines": verb_lines, "true_rederive": true_rederive,
            "planted_flagged": planted_flagged,
            "shows_recompute": (verb_lines + true_rederive) > 0}


# ------------------------------------------------------------------- run ids
def gid(arm, pid, seed):
    return R.sha(["E12", arm, pid, "gold", seed])


def cid(arm, pid, fam, roll, seed):
    return R.sha(["E12", arm, pid, fam, roll, seed])


def _sibling_ledgers(base):
    # cumulative hard-stop scope = the recompute_wall tree (stable via HERE=e11_run),
    # matching E11; NOT derived from out_base (which could resolve above the project).
    root = os.path.abspath(os.path.join(HERE, ".."))
    found = []
    for dp, dn, fn in os.walk(root):
        if ".venv" in dp or "__pycache__" in dp:
            continue
        for f in fn:
            if f == "cost_ledger.json":
                found.append(os.path.join(dp, f))
    return found


def wilson(k, n):
    return R.wilson(k, n)


def _cell_summary(rows):
    n = len(rows)
    dist = Counter(r["label"] for r in rows)
    out = {"n": n, "program_n": len({r["program_id"] for r in rows}),
           "label_counts": dict(dist)}
    for lab in ("absorbed", "silently_corrected", "flagged", "unresolved"):
        k = dist.get(lab, 0)
        out[lab] = {"count": k, "rate": round(k / n, 4) if n else None,
                    "wilson95": wilson(k, n)}
    toks = [r["output_tokens"] for r in rows if r.get("output_tokens")]
    out["output_tokens_mean"] = round(sum(toks) / len(toks), 1) if toks else None
    sr = [r for r in rows if r.get("shows_recompute")]
    out["shows_recompute_rate"] = round(len(sr) / n, 4) if n else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out-base", required=True, help="e12_fixes root; per-arm subdir created")
    ap.add_argument("--programs", required=True, help="shared programs.jsonl (identical worlds across arms)")
    ap.add_argument("--n-target", type=int, default=60)
    ap.add_argument("--world-cap", type=int, default=0,
                    help="if >0, process at most this many programs per k (gold cost bound / smoke test)")
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-gold", type=int, default=512)
    ap.add_argument("--usd-cap", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if api_gen is None:
        raise SystemExit("api_gen (exph) not importable on this node; the API arm needs it.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Source the keys env file, then rerun. "
                         "This script will not prompt for or store a credential.")
    arm = a.arm
    instr = ARMS[arm]
    out_dir = os.path.join(a.out_base, arm)
    os.makedirs(out_dir, exist_ok=True)
    P = {"raw": os.path.join(out_dir, "raw_generations.jsonl"),
         "validated": os.path.join(out_dir, "validated_outputs.jsonl"),
         "summary": os.path.join(out_dir, "summary_tables.json"),
         "manifest": os.path.join(out_dir, "run_manifest.json"),
         "status": os.path.join(out_dir, "queue_status.json"),
         "ledger": os.path.join(out_dir, "cost_ledger.json")}

    programs = [p for p in R.read_jsonl(a.programs) if p["k"] in TARGET_KS]
    if a.world_cap and a.world_cap > 0:
        capped, per_k = [], defaultdict(int)
        for p in sorted(programs, key=lambda x: (x["k"], x["program_id"])):
            if per_k[p["k"]] < a.world_cap:
                capped.append(p)
                per_k[p["k"]] += 1
        programs = capped
    prog_by_id = {p["program_id"]: p for p in programs}
    ledger = api_gen.CostLedger(P["ledger"], extra_paths=_sibling_ledgers(a.out_base))

    # ---- run manifest: pin the exact instruction bytes + provenance BEFORE gen
    manifest = {
        "experiment": "E12_FIXES_LADDER", "arm": arm, "model": MODEL_ID,
        "created_at": R.now_iso(), "seed": a.seed, "R": a.R, "temp": a.temp,
        "n_target_per_cell": a.n_target,
        "instruction_verbatim": instr,
        "instruction_sha256": R.sha([instr]),
        "instruction_position": "prepended to user message (E9 VERIF_INSTR position); prefill unchanged",
        "gold_policy": "golds REGENERATED per arm with the arm instruction present in the same position",
        "target_cells": {"%d_%s" % (k, o): CELL_ALIAS[(k, o)] for k, o in sorted(TARGET_CELLS)},
        "programs_path": os.path.abspath(a.programs),
        "n_programs_in_target_ks": len(programs),
        "prompt_sha256": tf.prompt_sha256(), "gold_prefill": tf.GOLD_PREFILL,
        "flag_channel": "doubt_lex frozen lexicon (no 32B judge on API arm)",
        "doubt_lexicon": val_mod.DOUBT_LEX_RE.pattern,
        "recompute_signature_def": recompute_signature.__doc__,
        "arms_all": ARMS,
    }
    R.write_json(P["manifest"], manifest)

    def budget_ok():
        try:
            ledger.check_budget()
        except api_gen.BudgetExceeded as e:
            print("[E12:%s] %s" % (arm, e))
            return False
        if ledger.cost_usd(MODEL_ID) > a.usd_cap:
            print("[E12:%s] soft cap hit: this-run $%.2f > $%.2f"
                  % (arm, ledger.cost_usd(MODEL_ID), a.usd_cap))
            return False
        return True

    def set_status(**kw):
        st = {}
        if os.path.exists(P["status"]):
            try:
                st = json.load(open(P["status"]))
            except Exception:
                st = {}
        st.update(kw)
        st["arm"] = arm
        st["updated_at"] = R.now_iso()
        st["this_run_usd"] = round(ledger.cost_usd(MODEL_ID), 4)
        R.write_json(P["status"], st)

    existing = {r["run_id"] for r in R.read_jsonl(P["raw"])}
    raw_by_id = {r["run_id"]: r for r in R.read_jsonl(P["raw"])}

    # concurrency plumbing (8-way, matching api_gen.generate_batch); resume-safe on
    # run_id; cooperative budget stop shared across workers.
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    stop = {"halt": False}

    def check_stop():
        if stop["halt"]:
            return True
        if not budget_ok():
            stop["halt"] = True
            return True
        return False

    # ---- phase 1: gold (own-trace, prefill "line 1:", instruction in user block)
    set_status(stage="gold", n_worlds=len(programs))
    gold_todo = [p for p in programs
                 if gid(arm, p["program_id"], a.seed) not in existing]

    def gold_work(p):
        if check_stop():
            return
        rid = gid(arm, p["program_id"], a.seed)
        um = _apply_instruction(
            tf.make_user_message(tf.make_listing_text(p["stmt_texts"])), instr)
        out = _call({**cfg_for(0.0), "max_tokens": a.max_new_gold},
                    um, tf.GOLD_PREFILL, ledger)
        g = {"run_id": rid, "program_id": p["program_id"], "k": p["k"],
             "condition": "gold", "continuation": out.get("text"),
             "failed_generation": out.get("text") is None,
             "output_tokens": out.get("output_tokens")}
        with lock:
            R.append_jsonl(P["raw"], g)
            existing.add(rid)
            raw_by_id[rid] = g
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(gold_work, gold_todo))
    ledger.save()

    # eligibility over ALL golds present (resumed + new)
    by_k = defaultdict(list)
    for p in programs:
        g = raw_by_id.get(gid(arm, p["program_id"], a.seed))
        if g and not g.get("failed_generation"):
            ev = inj_mod.gold_solve_eval(p, tf.GOLD_PREFILL + g["continuation"])
            if ev["solved"]:
                by_k[p["k"]].append((p, g))
    gold_cohort = {}

    # ---- phase 2: continuations for target cells only (build job list, run 8-way)
    set_status(stage="continuations")
    cont_jobs = []
    for k in TARGET_KS:
        elig = sorted(by_k[k], key=lambda pg: pg[0]["program_id"])[:a.n_target]
        gold_cohort[str(k)] = {"eligible": len(by_k[k]), "kept": len(elig)}
        for p, g in elig:
            for fam in R.families_of(p):
                kk, opac = R.cell_of(p, fam)
                if (kk, opac) not in TARGET_CELLS:
                    continue
                try:
                    b = R.build_injection_e11(p, fam, tf.GOLD_PREFILL + g["continuation"])
                except inj_mod.InjectError:
                    continue
                um = _apply_instruction(
                    tf.make_user_message(tf.make_listing_text(p["stmt_texts"])), instr)
                for roll in range(0, a.R + 1):
                    rid = cid(arm, p["program_id"], fam, roll, a.seed)
                    if rid in existing:
                        continue
                    cont_jobs.append((p, fam, kk, opac, b, roll, rid, um))

    def cont_work(job):
        p, fam, kk, opac, b, roll, rid, um = job
        if check_stop():
            return
        cfg = cfg_for(0.0 if roll == 0 else a.temp)
        out = _call(cfg, um, b["prefix_text"], ledger)
        rec = {"run_id": rid, "program_id": p["program_id"], "family": fam,
               "k": kk, "opacity": opac, "cell": CELL_ALIAS[(kk, opac)],
               "arm": arm, "rollout": roll,
               "planted_value": b["planted_value"], "true_value": b["true_value"],
               "continuation": out.get("text"),
               "failed_generation": out.get("text") is None,
               "output_tokens": out.get("output_tokens"),
               "condition": "cont"}
        with lock:
            R.append_jsonl(P["raw"], rec)
            existing.add(rid)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(cont_work, cont_jobs))
    ledger.save()

    # ---- validate (3-way DV) over target continuations
    if os.path.exists(P["validated"]):
        os.remove(P["validated"])
    validated = []
    raw = R.read_jsonl(P["raw"])
    cont_rows = [r for r in raw if r.get("condition") == "cont"]
    raw_by_id = {r["run_id"]: r for r in raw}
    for r in cont_rows:
        p = prog_by_id[r["program_id"]]
        g = raw_by_id.get(gid(arm, r["program_id"], a.seed))
        stmts = parse_program(p["stmt_texts"])
        try:
            b = R.build_injection_e11(p, r["family"], tf.GOLD_PREFILL + g["continuation"])
        except Exception:
            continue
        inj = {x: b[x] for x in ("cf", "planted_var", "planted_value",
                                 "force_after_line", "expected_lines_from")}
        rec = {x: r[x] for x in ("program_id", "family", "k", "opacity", "cell",
                                 "arm", "rollout", "planted_value", "true_value",
                                 "output_tokens", "run_id")}
        text = r.get("continuation")
        if r.get("failed_generation") or text is None:
            rec.update({"generation_failed": True, "label": "unresolved",
                        "shows_recompute": False})
        else:
            met = val_mod.classify_run(stmts, p, inj, text)
            rec.update({x: met[x] for x in ("final_output_absorbed", "final_output_valid",
                        "repair_event", "trace_valid", "next_read_absorbed",
                        "doubt_lex", "unparsed")})
            rec["judge_reject"] = None
            rec["label"] = R.three_way_label(rec)
            sig = recompute_signature(text, b["planted_var"], b["planted_value"],
                                      b["true_value"])
            rec.update({"sig_verb_lines": sig["verb_lines"],
                        "sig_true_rederive": sig["true_rederive"],
                        "sig_planted_flagged": sig["planted_flagged"],
                        "shows_recompute": sig["shows_recompute"]})
        validated.append(rec)
        R.append_jsonl(P["validated"], rec)

    # ---- summary per cell
    cells = {}
    for (k, opac) in sorted(TARGET_CELLS):
        crows = [r for r in validated if r["k"] == k and r["opacity"] == opac]
        if crows:
            cells["k%d_%s" % (k, opac)] = dict(_cell_summary(crows),
                                               alias=CELL_ALIAS[(k, opac)])
    summ = {"experiment": "E12_FIXES_LADDER", "arm": arm, "model": MODEL_ID,
            "instruction_sha256": manifest["instruction_sha256"],
            "created_at": R.now_iso(), "cells": cells, "gold_cohort": gold_cohort,
            "this_run_usd": round(ledger.cost_usd(MODEL_ID), 4),
            "flag_channel": manifest["flag_channel"]}
    R.write_json(P["summary"], summ)
    ledger.save()
    set_status(stage="done", done=True, gold_cohort=gold_cohort)
    print(json.dumps({"arm": arm, "gold_cohort": gold_cohort,
                      "absorbed_by_cell": {n: c["absorbed"]["rate"] for n, c in cells.items()},
                      "shows_recompute_by_cell": {n: c["shows_recompute_rate"] for n, c in cells.items()},
                      "this_run_usd": summ["this_run_usd"]}, indent=2))


if __name__ == "__main__":
    main()
