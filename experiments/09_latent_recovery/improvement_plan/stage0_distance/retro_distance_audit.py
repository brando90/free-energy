"""Stage-0 M5: retroactive refutation-distance audit of all locked injections.

Computes, for every locked injection row that can be parsed, the refutation
distance d(a*) of the planted claim a*:

    d = BFS depth over DIRECT rule applications from the visible prefix state
        (question premises for the entity + entity facts in the prefix proof
        steps) needed to derive the COMPLEMENT of a*.

    d=0  : complement already in the prefix state
    d=k  : k direct rule applications
    d=inf: complement underivable (falsity certified only by closed-world
           closure search, if at all)

The distance function is the code-native owner `shortest_rule_distance`
(src/expc_polarity_control.py:340-356), imported unmodified, applied with its
own helpers (`world_state`, `statement_predicate`, `opposite_pred`). Nothing
under src/, results/, or data/ is modified; this script only READS them and
writes to its --out directory.

Datasets audited:
  1. EXPA_GLOBAL_EXPANSION: results/EXPA_GLOBAL_EXPANSION/validated_outputs.jsonl
     joined to manifest.jsonl on run_id (manifest carries question / entity /
     prefix_steps / injected_statement).
  2. Legacy families: every results/perturbed*/runs.jsonl (negstep, neghop2-5,
     contradiction, distractor, falsehood, wrong-category, ...). Prefix steps
     are reconstructed exactly as src/validator.py:main does: gold rollout
     gen_text split into sentences, prefix = steps[:sent_idx], entity =
     steps[0].split()[0], restricted to the gold-validated cohort.

Outputs (in --out):
  retro_distance_audit.json  : machine-readable distributions + re-binned rates
  RETRO_DISTANCE_REPORT.md   : human-readable report
  row_level_distances.jsonl  : one row per audited injection (for downstream use)

Usage (on cluster, from project root):
  /lfs/skampere2/0/eobbad/free-energy/.venv/bin/python \
      improvement_plan/stage0_distance/retro_distance_audit.py \
      --root /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
"""
import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

D_BINS = ["0", "1", "2", "3", "4", "5+", "inf", "unparseable"]

# Designed distances per family (for check (a)).
DESIGNED_D = {
    "EXPA:global_falsehood": "inf",
    "EXPA:one_hop_falsehood": "1",
    "EXPA:benign_paraphrase": "true_control(no design)",
    "EXPA:true_interruption": "true_control(no design)",
    "legacy:perturbed": "none(wrong-category, v1; often entailed-true)",
    "legacy:perturbed_falsehood": "inf",
    "legacy:perturbed_contradiction": "0",
    "legacy:perturbed_distractor": "rule-injection(unparseable as entity fact)",
    "legacy:perturbed_negstep": "1",
    "legacy:perturbed_neghop2": "2",
    "legacy:perturbed_neghop3": "3",
    "legacy:perturbed_neghop4": "4",
    "legacy:perturbed_neghop5": "5",
}


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def load_code(src_dir):
    sys.path.insert(0, src_dir)
    import validator  # noqa: F401
    import expc_polarity_control as expc  # noqa: F401
    return validator, expc


def refutation_path(opp, state, direct, max_depth=64):
    """Diagnostic-only BFS mirroring shortest_rule_distance but tracking the
    rule chain. Returns list of 'lhs -> rhs' strings, or None."""
    if opp in state:
        return []
    frontier = [(a[1], 0, [f"state:{a}"]) for a in sorted(state, key=str) if a[0] == "cat"]
    seen = {c for c, _, _ in frontier}
    while frontier:
        cat, depth, path = frontier.pop(0)
        if depth >= max_depth:
            continue
        for rhs in sorted(direct.get(cat, set()), key=str):
            nd = depth + 1
            npath = path + [f"{cat} -> {rhs}"]
            if rhs == opp:
                return npath
            if rhs[0] == "cat" and rhs[1] not in seen:
                seen.add(rhs[1])
                frontier.append((rhs[1], nd, npath))
    return None


def d_bin(d_native, d_deep):
    """Bin from the depth-64 BFS (exact for this fragment: BFS visits each
    category once, chains are short); d_native (max_depth=8 default) reported
    separately."""
    if d_deep is None:
        return "inf"
    if d_deep >= 5:
        return "5+"
    return str(d_deep)


def compute_row(expc, validator, question, entity, prefix_steps, statement):
    """Full code-native distance computation for one locked injection."""
    out = {}
    pred = expc.statement_predicate(statement, entity)
    if pred is None:
        out.update({
            "planted_pred": None,
            "complement_pred": None,
            "d_native_max8": None,
            "d": None,
            "d_bin": "unparseable",
            "complement_closure_derivable_from_prefix": None,
            "world_unparsed_sentences": None,
        })
        return out
    opp = expc.opposite_pred(pred)
    _, direct, reach, state, unparsed = expc.world_state(question, entity, prefix_steps)
    d_native = expc.shortest_rule_distance(opp, state, direct)  # code-native default max_depth=8
    d_deep = expc.shortest_rule_distance(opp, state, direct, max_depth=64)
    closure_deriv = validator.derivable(opp, state, reach)
    out.update({
        "planted_pred": list(pred),
        "complement_pred": list(opp),
        "d_native_max8": d_native,
        "d": d_deep,
        "d_bin": d_bin(d_native, d_deep),
        "complement_closure_derivable_from_prefix": bool(closure_deriv),
        "world_unparsed_sentences": len(unparsed),
    })
    if d_deep is not None and d_deep >= 1:
        path = refutation_path(opp, state, direct)
        out["refutation_path"] = path
    return out


# ---------------------------------------------------------------------------
# EXPA
# ---------------------------------------------------------------------------

def audit_expa(root, expc, validator, row_sink, problems):
    res_dir = os.path.join(root, "results", "EXPA_GLOBAL_EXPANSION")
    man_path = os.path.join(res_dir, "manifest.jsonl")
    val_path = os.path.join(res_dir, "validated_outputs.jsonl")
    if not (os.path.exists(man_path) and os.path.exists(val_path)):
        problems.append(f"EXPA: missing {man_path} or {val_path}; skipped.")
        return [], []
    manifest = {r["run_id"]: r for r in read_jsonl(man_path)}
    validated = read_jsonl(val_path)
    audited = []
    missing_manifest = 0
    for r in validated:
        m = manifest.get(r.get("run_id"))
        if m is None:
            missing_manifest += 1
            continue
        need = ("question", "entity", "prefix_steps", "injected_statement", "injection_position", "condition")
        if any(k not in m for k in need):
            problems.append(f"EXPA run {r.get('run_id')}: manifest missing one of {need}")
            continue
        comp = compute_row(expc, validator, m["question"], m["entity"], m["prefix_steps"], m["injected_statement"])
        rec = {
            "dataset": "EXPA",
            "family": m["condition"],
            "position": m["injection_position"],
            "run_id": r["run_id"],
            "problem_id": m.get("problem_id"),
            "injected_statement": m["injected_statement"],
            "audited_truth_status": m.get("audited_truth_status"),
            "sent_idx": m.get("sent_idx"),
            "n_prefix_steps": len(m["prefix_steps"]),
            # stored outcome fields for re-binning
            "class": r.get("class"),
            "verbalized_doubt": bool(r.get("verbalized_doubt")),
            "acknowledged": bool(r.get("acknowledged")),
            "n_poisoned": r.get("n_poisoned"),
            "failed_generation": bool(r.get("failed_generation")),
            **comp,
        }
        audited.append(rec)
        row_sink.write(json.dumps(rec, sort_keys=True) + "\n")
    if missing_manifest:
        problems.append(f"EXPA: {missing_manifest} validated rows had no manifest entry (excluded).")
    return audited, validated


# ---------------------------------------------------------------------------
# Legacy families
# ---------------------------------------------------------------------------

def legacy_split_steps(text):
    return [x.strip() for x in re.split(r"(?<=\.)\s+", text.strip()) if x.strip()]


def audit_legacy(root, expc, validator, row_sink, problems):
    data_path = os.path.join(root, "data", "pilot.jsonl")
    gold_path = os.path.join(root, "results", "gold", "rollouts.jsonl")
    if not os.path.exists(data_path) or not os.path.exists(gold_path):
        problems.append(
            f"legacy: missing {data_path if not os.path.exists(data_path) else gold_path}; "
            "legacy families skipped entirely."
        )
        return []
    data = {r["id"]: r for r in read_jsonl(data_path)}
    gold = {r["id"]: r for r in read_jsonl(gold_path)}

    # Gold-validated cohort, exactly as validator.main defines it.
    cohort = {}
    for gid, g in gold.items():
        if not g.get("solved"):
            continue
        inst = data.get(gid)
        if inst is None:
            continue
        steps = legacy_split_steps(g["gen_text"])
        if not steps:
            continue
        entity = steps[0].split()[0]
        gv = validator.validate_continuation(inst["question"], [], None, g["gen_text"], inst["target"], entity)
        if gv["class"] == "valid_rederivation":
            cohort[gid] = (inst, steps, entity)

    audited = []
    run_files = sorted(glob.glob(os.path.join(root, "results", "perturbed*", "runs.jsonl")))
    if not run_files:
        problems.append("legacy: no results/perturbed*/runs.jsonl files found.")
    for rf in run_files:
        family = os.path.basename(os.path.dirname(rf))
        rows = read_jsonl(rf)
        n_skip = n_not_cohort = n_schema = 0
        schema_missing = Counter()
        for r in rows:
            if "skip" in r:
                n_skip += 1
                continue
            rid = r.get("id")
            if rid not in cohort:
                n_not_cohort += 1
                continue
            missing = [k for k in ("sent_idx", "corrupted_step", "point") if k not in r]
            if missing:
                n_schema += 1
                schema_missing.update(missing)
                continue
            inst, steps, entity = cohort[rid]
            prefix_steps = steps[: r["sent_idx"]]
            comp = compute_row(expc, validator, inst["question"], entity, prefix_steps, r["corrupted_step"])
            rec = {
                "dataset": "legacy",
                "family": family,
                "position": r["point"],
                "id": rid,
                "injected_statement": r["corrupted_step"],
                "sent_idx": r["sent_idx"],
                "n_prefix_steps": len(prefix_steps),
                **comp,
            }
            audited.append(rec)
            row_sink.write(json.dumps(rec, sort_keys=True) + "\n")
        note = (
            f"legacy family {family}: rows={len(rows)} audited={sum(1 for a in audited if a['family'] == family)} "
            f"skip-marked={n_skip} not-in-gold-validated-cohort={n_not_cohort} schema-missing={n_schema}"
        )
        if n_schema:
            note += f" (missing fields: {dict(schema_missing)}) — prefix NOT recoverable for those rows"
        problems.append(note)
    return audited


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def distribution(rows):
    c = Counter(r["d_bin"] for r in rows)
    total = len(rows)
    finite = sum(v for k, v in c.items() if k not in ("inf", "unparseable"))
    return {
        "n": total,
        "counts": {b: c.get(b, 0) for b in D_BINS},
        "finite_d_count": finite,
        "finite_d_fraction": round(finite / total, 4) if total else None,
        "native_max8_disagreements": sum(
            1 for r in rows if r.get("d") is not None and r.get("d_native_max8") is None
        ),
    }


def aggregate(rows):
    out = {}
    fams = sorted({(r["dataset"], r["family"]) for r in rows})
    for ds, fam in fams:
        frows = [r for r in rows if r["dataset"] == ds and r["family"] == fam]
        fkey = f"{ds}:{fam}"
        entry = {
            "designed_d": DESIGNED_D.get(fkey, "unknown"),
            "pooled": distribution(frows),
            "by_position": {},
        }
        for pos in sorted({r["position"] for r in frows}):
            entry["by_position"][pos] = distribution([r for r in frows if r["position"] == pos])
        out[fkey] = entry
    return out


def rates(rows):
    """Headline rates from stored fields (EXPA validated schema)."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    def rate(k):
        return {"count": k, "rate": round(k / n, 4), "wilson95": wilson(k, n)}
    return {
        "n": n,
        "problem_n": len({r.get("problem_id") for r in rows}),
        "closure_valid": rate(sum(1 for r in rows if r.get("class") == "valid_rederivation")),
        "poisoned_injection_dependence": rate(sum(1 for r in rows if r.get("class") == "poisoned")),
        "doubt": rate(sum(1 for r in rows if r.get("verbalized_doubt"))),
        "parroted": rate(sum(1 for r in rows if r.get("class") == "parroted")),
        "derailed": rate(sum(1 for r in rows if r.get("class") == "derailed")),
        "unparsed": rate(sum(1 for r in rows if r.get("class") == "unparsed")),
        "generation_failed": rate(sum(1 for r in rows if r.get("class") == "generation_failed")),
    }


def rebin_expa_global(expa_rows):
    grows = [r for r in expa_rows if r["family"] == "global_falsehood"]
    if not grows:
        return None
    contaminated = [r for r in grows if r["d_bin"] not in ("inf", "unparseable")]
    strict = [r for r in grows if r["d_bin"] == "inf"]
    unparseable = [r for r in grows if r["d_bin"] == "unparseable"]
    out = {
        "n_global_rows": len(grows),
        "n_finite_d_contaminated": len(contaminated),
        "contaminated_fraction": round(len(contaminated) / len(grows), 4),
        "contaminated_fraction_wilson95": wilson(len(contaminated), len(grows)),
        "n_unparseable_planted": len(unparseable),
        "contaminated_examples": [
            {
                "run_id": r["run_id"],
                "position": r["position"],
                "injected_statement": r["injected_statement"],
                "d": r["d"],
                "refutation_path": r.get("refutation_path"),
            }
            for r in contaminated[:25]
        ],
        "rates_full_cell": rates(grows),
        "rates_strict_d_inf_subset": rates(strict),
        "rates_finite_d_subset": rates(contaminated),
    }
    out["by_position"] = {}
    for pos in sorted({r["position"] for r in grows}):
        p_all = [r for r in grows if r["position"] == pos]
        p_strict = [r for r in p_all if r["d_bin"] == "inf"]
        p_cont = [r for r in p_all if r["d_bin"] not in ("inf", "unparseable")]
        out["by_position"][pos] = {
            "contaminated_fraction": round(len(p_cont) / len(p_all), 4) if p_all else None,
            "rates_full_cell": rates(p_all),
            "rates_strict_d_inf_subset": rates(p_strict),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def dist_table_md(agg):
    lines = [
        "| family | position | designed d | n | d=0 | d=1 | d=2 | d=3 | d=4 | d=5+ | d=inf | unparseable | finite-d frac |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fkey in sorted(agg):
        e = agg[fkey]
        rows = [("pooled", e["pooled"])] + sorted(e["by_position"].items())
        for pos, d in rows:
            c = d["counts"]
            lines.append(
                f"| {fkey} | {pos} | {e['designed_d']} | {d['n']} | {c['0']} | {c['1']} | {c['2']} | "
                f"{c['3']} | {c['4']} | {c['5+']} | {c['inf']} | {c['unparseable']} | {d['finite_d_fraction']} |"
            )
    return "\n".join(lines)


def rates_row_md(label, rr):
    if not rr or rr.get("n", 0) == 0:
        return f"| {label} | 0 | - | - | - | - | - |"
    def f(m):
        v = rr[m]
        return f"{v['rate']} [{v['wilson95'][0]}, {v['wilson95'][1]}]"
    return (
        f"| {label} | {rr['n']} | {f('closure_valid')} | {f('poisoned_injection_dependence')} | "
        f"{f('doubt')} | {f('parroted')} | {f('derailed')} |"
    )


def write_report(out_dir, result):
    agg = result["distance_distributions"]
    reb = result["expa_global_rebin"]
    lines = [
        "# Stage-0 M5 — Retroactive Refutation-Distance Audit",
        "",
        f"Generated: {result['created_at']}",
        "",
        "Distance owner: `shortest_rule_distance` (src/expc_polarity_control.py:340-356), imported",
        "unmodified; prefix state built by code-native `world_state(question, entity, prefix_steps)`;",
        "planted-claim complement via `opposite_pred(statement_predicate(...))`.",
        "d is reported from a depth-64 BFS (identical algorithm, larger cap); rows where the",
        "code-native default cap of 8 would disagree are counted per family",
        "(`native_max8_disagreements`).",
        "",
        "## d-distribution per family x position",
        "",
        dist_table_md(agg),
        "",
        "## Key question (a): do designed distances hold?",
        "",
    ]
    for fkey in sorted(agg):
        e = agg[fkey]
        designed = e["designed_d"]
        d = e["pooled"]
        if designed in ("0", "1", "2", "3", "4", "5"):
            bin_key = designed if designed in d["counts"] else "5+"
            frac = d["counts"].get(bin_key, 0) / d["n"] if d["n"] else None
            lines.append(f"- **{fkey}** (designed d={designed}): {round(frac, 4) if frac is not None else 'n/a'} of {d['n']} rows measure exactly d={designed}.")
        elif designed == "inf":
            frac = d["counts"]["inf"] / d["n"] if d["n"] else None
            lines.append(f"- **{fkey}** (designed d=inf): {round(frac, 4) if frac is not None else 'n/a'} of {d['n']} rows measure d=inf; finite-d fraction = {d['finite_d_fraction']}.")
        else:
            lines.append(f"- **{fkey}**: {designed}; finite-d fraction = {d['finite_d_fraction']}, unparseable = {d['counts']['unparseable']}/{d['n']}.")
    lines += ["", "## Key question (b): EXPA global-cell contamination", ""]
    if reb:
        lines += [
            f"- Globally-checkable-only (`global_falsehood`) rows audited: **{reb['n_global_rows']}**",
            f"- Rows with FINITE refutation distance (mis-binned): **{reb['n_finite_d_contaminated']}**",
            f"- Contaminated fraction: **{reb['contaminated_fraction']}** (Wilson95 {reb['contaminated_fraction_wilson95']})",
            f"- Unparseable planted claims in this cell: {reb['n_unparseable_planted']}",
            "",
        ]
        threshold_hit = reb["contaminated_fraction"] is not None and reb["contaminated_fraction"] > 0.02
        lines.append(
            "Contamination exceeds the 2% disclosure threshold; re-binned rates below."
            if threshold_hit
            else "Contamination is at or below the 2% threshold; re-binned rates reported anyway for completeness."
        )
        lines += [
            "",
            "### Headline rates: full cell vs strictly-d=inf subset (pooled)",
            "",
            "| subset | n | closure_valid | poisoned (inj-dep) | doubt | parroted | derailed |",
            "|---|---|---|---|---|---|---|",
            rates_row_md("full cell", reb["rates_full_cell"]),
            rates_row_md("strict d=inf", reb["rates_strict_d_inf_subset"]),
            rates_row_md("finite-d (contaminated)", reb["rates_finite_d_subset"]),
            "",
            "### By position",
            "",
            "| position | subset | n | closure_valid | poisoned (inj-dep) | doubt | parroted | derailed |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for pos, pd in reb["by_position"].items():
            fr = pd["rates_full_cell"]
            sr = pd["rates_strict_d_inf_subset"]
            lines.append("| " + pos + " " + rates_row_md("full", fr).lstrip("|"))
            lines.append("| " + pos + " " + rates_row_md("strict d=inf", sr).lstrip("|"))
        if reb["contaminated_examples"]:
            lines += ["", "### Contaminated examples (first 25)", ""]
            for ex in reb["contaminated_examples"]:
                lines.append(
                    f"- `{ex['run_id']}` ({ex['position']}): \"{ex['injected_statement']}\" d={ex['d']}; "
                    f"path: {' | '.join(ex['refutation_path'] or [])}"
                )
    else:
        lines.append("- EXPA global_falsehood rows not found; contamination not computable.")
    lines += ["", "## Coverage / schema notes", ""]
    for p in result["coverage_notes"]:
        lines.append(f"- {p}")
    lines += [
        "",
        "## Semantics notes",
        "",
        "- `d` uses only DIRECT rule applications from cat-atoms in the prefix state, exactly as",
        "  `shortest_rule_distance` implements (BFS over `direct_rule_map`); `d=0` means the",
        "  complement is literally an atom of the prefix state.",
        "- `unparseable` = the planted statement does not parse as an entity fact for the proof",
        "  entity (`statement_predicate` returned None). Rule injections (distractor family) land",
        "  here by construction; refutation distance is not defined for them in this fragment.",
        "- `complement_closure_derivable_from_prefix` (row-level output) cross-checks the BFS:",
        "  it must agree with finite d in this fragment (closure == unbounded BFS).",
        "- Legacy rows are restricted to the gold-validated cohort (gold rollout solved AND",
        "  `valid_rederivation`), mirroring `src/validator.py:main`, so distances correspond to",
        "  the locked, analyzed rows.",
    ]
    with open(os.path.join(out_dir, "RETRO_DISTANCE_REPORT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="project root (has results/, data/, src/)")
    ap.add_argument("--src", default=None, help="src dir with validator.py + expc_polarity_control.py (default root/src)")
    ap.add_argument("--out", default=None, help="output dir (default root/improvement_plan/stage0_distance)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    src = args.src or os.path.join(root, "src")
    out_dir = args.out or os.path.join(root, "improvement_plan", "stage0_distance")
    os.makedirs(out_dir, exist_ok=True)

    validator, expc = load_code(src)

    problems = []
    row_path = os.path.join(out_dir, "row_level_distances.jsonl")
    with open(row_path, "w") as row_sink:
        expa_rows, _ = audit_expa(root, expc, validator, row_sink, problems)
        legacy_rows = audit_legacy(root, expc, validator, row_sink, problems)

    all_rows = expa_rows + legacy_rows
    # Cross-check: closure-derivability must agree with finite d.
    xdisagree = [
        r for r in all_rows
        if r.get("planted_pred") is not None
        and bool(r.get("complement_closure_derivable_from_prefix")) != (r.get("d") is not None)
    ]
    if xdisagree:
        problems.append(
            f"CROSS-CHECK FAILURE: {len(xdisagree)} rows where closure-derivability of the complement "
            f"disagrees with BFS finiteness (first: {xdisagree[0].get('run_id') or xdisagree[0].get('id')})."
        )

    result = {
        "created_at": now_iso(),
        "root": root,
        "distance_owner": "src/expc_polarity_control.py:shortest_rule_distance (imported, unmodified)",
        "n_rows_audited": len(all_rows),
        "distance_distributions": aggregate(all_rows),
        "expa_global_rebin": rebin_expa_global(expa_rows),
        "coverage_notes": problems,
        "cross_check_disagreements": len(xdisagree),
    }
    with open(os.path.join(out_dir, "retro_distance_audit.json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    write_report(out_dir, result)
    print(json.dumps({
        "n_rows_audited": len(all_rows),
        "families": {k: v["pooled"]["counts"] for k, v in result["distance_distributions"].items()},
        "expa_global_contaminated_fraction": (result["expa_global_rebin"] or {}).get("contaminated_fraction"),
        "coverage_notes": problems,
    }, indent=2))


if __name__ == "__main__":
    main()
