#!/usr/bin/env python3
"""Natural-error persistence: when a model makes ITS OWN computed slip while
solving the base task (gold generation, no perturbation), does it ever recover
downstream? Uses the shipped e10_widen gold traces (read-only) + the paper's own
interpreter. Writes results to iclr_exec/recompute_wall/natural_persistence/.

Definitions per gold trace:
  first slip   = first parsed assign-claim whose value != true value at that
                 line, with every earlier claim correct (clean-until-slip).
  slip type    = computed (stmt at that line is an operation) vs copy (constant
                 or bare copy) — read from the program stmt text.
  self-error world = execute_cf forcing slip_var := slip_value after slip line
                 (identical machinery the paper uses to grade planted errors).
  downstream claims graded against BOTH worlds where they differ:
                 carried    -> matches the model's own wrong value world
                 recovered  -> matches the true world (model got back on track)
                 other      -> matches neither (further independent slips)
  trace outcome (on traces where the two worlds differ at the final output):
                 carried_to_output / recovered_at_output / derailed / no_output
  flagged      = paper's frozen doubt lexicon fires anywhere in the continuation.
"""
import json, os, sys, re, collections

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
IP = EXP + "/improvement_plan"
for p in (IP + "/expg",):
    sys.path.insert(0, p)
import interp, trace_format as tf, validate as val_mod  # noqa: E402

E10 = IP + "/iclr_exec/recompute_wall/e10_widen/results_skampere1"
OUT = IP + "/iclr_exec/recompute_wall/natural_persistence"
os.makedirs(OUT, exist_ok=True)

OP_RE = re.compile(r"[+\-*]")

def stmt_kind(stmt_text):
    # "x = 5" -> copy/const ; "x = a + b" or "x = a + 3" -> computed ; "x = y" -> copy
    rhs = stmt_text.split("=", 1)[1] if "=" in stmt_text else ""
    return "computed" if OP_RE.search(rhs) else "copy"

def analyze_model(mdir):
    progs = {}
    for ln in open(os.path.join(mdir, "programs.jsonl")):
        p = json.loads(ln); progs[p["program_id"]] = p
    rows = []
    for ln in open(os.path.join(mdir, "raw_generations.jsonl")):
        r = json.loads(ln)
        if r.get("condition") == "gold" and not r.get("failed_generation"):
            rows.append(r)
    stats = collections.Counter()
    per_trace = []
    for r in rows:
        p = progs[r["program_id"]]
        text = tf.GOLD_PREFILL + (r.get("continuation") or "")
        stmts = interp.parse_program(p["stmt_texts"])
        wt = interp.execute(stmts)
        scan = tf.scan_trace(text)
        claims = [c for c in scan["claims"]
                  if c.get("kind") == "assign" and c.get("line") is not None
                  and c.get("value") is not None]
        stats["gold_traces"] += 1
        # find first wrong claim with all earlier claims correct
        slip = None; clean = True
        for c in claims:
            vt = interp.value_at(wt, c["line"], c["var"])
            if vt is None:
                continue  # claim about unknown var/line: skip, keep scanning
            if c["value"] == vt:
                continue
            slip = c
            break
        if slip is None:
            stats["no_slip_detected"] += 1
            continue
        kind = stmt_kind(p["stmt_texts"][slip["line"] - 1]) \
            if 1 <= slip["line"] <= len(p["stmt_texts"]) else "unknown"
        stats["slip_" + kind] += 1
        if kind != "computed":
            continue  # persistence question is about computed slips
        # self-error world
        try:
            we = interp.execute_cf(stmts, slip["var"], slip["value"], slip["line"])
        except Exception:
            stats["cf_failed"] += 1
            continue
        out_t, out_e = wt["out_value"], we["out_value"]
        downstream = [c for c in claims if c["line"] > slip["line"]]
        carried = recovered = other = nondisc = 0
        for c in downstream:
            vt = interp.value_at(wt, c["line"], c["var"])
            ve = interp.value_at(we, c["line"], c["var"])
            if vt is None and ve is None:
                continue
            if vt == ve:
                nondisc += 1
            elif c["value"] == ve:
                carried += 1
            elif c["value"] == vt:
                recovered += 1
            else:
                other += 1
        outc = next((c for c in scan["claims"] if c.get("kind") == "output"), None)
        if out_t == out_e:
            outcome = "output_nondiscriminating"
        elif not outc or outc.get("value") is None:
            outcome = "no_output"
        elif outc["value"] == out_e:
            outcome = "carried_to_output"
        elif outc["value"] == out_t:
            outcome = "recovered_at_output"
        else:
            outcome = "derailed"
        flagged = bool(val_mod.DOUBT_LEX_RE.search(text))
        per_trace.append({
            "program_id": r["program_id"], "slip_line": slip["line"],
            "slip_var": slip["var"], "slip_value": slip["value"],
            "true_value": interp.value_at(wt, slip["line"], slip["var"]),
            "downstream": {"carried": carried, "recovered": recovered,
                           "other": other, "nondiscriminating": nondisc},
            "outcome": outcome, "flagged": flagged})
        stats["computed_slip_traces"] += 1
        stats["outcome_" + outcome] += 1
        if flagged:
            stats["flagged"] += 1
        if recovered > 0:
            stats["any_downstream_recovery_claim"] += 1
        if carried > 0 and recovered == 0:
            stats["pure_carry"] += 1
    return dict(stats), per_trace

result = {}
for d in sorted(os.listdir(E10)):
    if not d.startswith("EXPG_PROGTRACE_"):
        continue
    m = d.replace("EXPG_PROGTRACE_", "")
    stats, per_trace = analyze_model(os.path.join(E10, d))
    result[m] = stats
    with open(os.path.join(OUT, "per_trace_%s.jsonl" % m), "w") as f:
        for row in per_trace:
            f.write(json.dumps(row) + "\n")
    # headline numbers
    n = stats.get("computed_slip_traces", 0)
    disc = sum(stats.get("outcome_" + k, 0) for k in
               ("carried_to_output", "recovered_at_output", "derailed", "no_output"))
    print("== %s: gold=%d  slips: computed=%d copy=%d | discriminating=%d" % (
        m, stats.get("gold_traces", 0), stats.get("slip_computed", 0),
        stats.get("slip_copy", 0), disc))
    if disc:
        for k in ("carried_to_output", "recovered_at_output", "derailed", "no_output"):
            v = stats.get("outcome_" + k, 0)
            print("     %-22s %4d  (%.3f)" % (k, v, v / disc))
        print("     flagged own slip     %4d  (%.3f)" % (
            stats.get("flagged", 0), stats.get("flagged", 0) / max(1, n)))
        print("     any downstream true-world claim: %d (%.3f)" % (
            stats.get("any_downstream_recovery_claim", 0),
            stats.get("any_downstream_recovery_claim", 0) / max(1, n)))
json.dump(result, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("\nsaved ->", OUT)
