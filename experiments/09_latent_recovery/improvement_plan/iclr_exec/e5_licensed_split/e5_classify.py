"""E5 -- Licensed-vs-unlicensed stated-complement split (ICLR revision, item E5).

For every row that STATED THE COMPLEMENT of the planted falsehood (the "rejection"
DV), classify the rejection as:

  LICENSED         : the continuation contains a WRITTEN, closure-valid derivation
                     of the complement -- i.e. the model restated the operative
                     refuting rule (a world rule  lhs -> comp_pred) whose antecedent
                     category was closure-established in the model's TRUE running
                     state at the point the complement is asserted (for d>=2 this
                     forces the model to have written the intermediate hops).
  VISIBLE_UNSHOWN  : the complement is closure-derivable from the world in <=1 hop
                     (measured_d in {0,1}) but the model wrote NO derivation
                     (bare assertion of a cheaply-visible complement).
  UNLICENSED       : everything else -- (a) the complement is NOT closure-derivable
                     in the augmented world at all (measured_d = None: the
                     FREQ/TEMPLATE/FILLER controls + cat_*_dinf), so the assertion
                     is lexically cued; or (b) it is derivable only at >=2 hops but
                     the model wrote no chain (silent multi-hop, which the paper
                     shows models do not actually perform).

Sub-splits recorded for transparency:
  d0_visible       : measured_d==0 -> the complement is a directly-GIVEN premise
                     (pure d=0 visibility / echo, not a derivation). Reported
                     separately; NOT counted toward "one-hop-or-farther checking".
  unlicensed reason: 'not_derivable' vs 'derivable_ge2_unshown'.

Also computes a lenient shown-work variant (SW_lenient) that additionally credits
the model for writing >=1 valid non-premise intermediate category that strictly
advances toward the complement (an ancestor on the closure path), even if the final
rule is not restated. Reported as an upper bound on LICENSED.

Reuses the canonical machinery ONLY:
  src/validator.py            : parse_fact, parse_rule, strip_marker, closure, derivable
  src/expc_polarity_control.py: opposite_pred, statement_predicate
  improvement_plan/expe/expe_evidence_mover.py:
                                question_world, shortest_rule_distance, split_sentences

Cluster bootstrap: resample clusters (cluster unit = problem_id) with replacement.
"""
import sys, os, json, random
from collections import defaultdict, Counter

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expe"))

from validator import parse_fact, parse_rule, strip_marker, derivable  # noqa: E402
from expc_polarity_control import opposite_pred, statement_predicate  # noqa: E402
from expe_evidence_mover import (question_world, shortest_rule_distance,  # noqa: E402
                                 split_sentences)

OUT = os.path.join(EXP, "improvement_plan", "iclr_exec", "e5_licensed_split")
SEED = 20260722
NBOOT = 5000


def read_jsonl(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
#  Core classifier
# --------------------------------------------------------------------------- #
def classify_row(question, entity, planted_statement, prefix_steps, continuation):
    """Return dict with measured_d, WD, shown-work flags, class, and reason.

    Operates entirely from the AUGMENTED question's world + the model's
    continuation. planted_statement is the falsehood (e.g. 'Fae is a gorpus.'
    or 'Stella is shiny.'); comp_pred is its opposite predicate.
    """
    pred = statement_predicate(planted_statement, entity)
    if pred is None:
        return None
    comp_pred = opposite_pred(pred)

    w = question_world(question, entity, prefix_steps or ())
    rules = w["rules"]
    reach = w["reach"]
    direct = w["direct"]
    prefix_state = w["prefix_state"]

    measured_d = shortest_rule_distance(comp_pred, prefix_state, direct)
    WD = measured_d is not None

    # world rules whose RHS is exactly the complement predicate (candidate
    # "operative refuting rules")
    refuting_rules = [(lhs, rhs) for (lhs, rhs) in rules if rhs == comp_pred]

    # Walk the continuation maintaining the TRUE (non-poisoned) established
    # state S for `entity`.  We stop at the FIRST asserted complement.
    # S_written tracks cats established by PREMISES + WRITTEN valid steps
    # (literal membership); used for the STRICT shown-work test so that a d>=2
    # rule restatement is credited only when the model actually wrote the
    # intermediate hops connecting `entity` to the rule's antecedent.
    S = set(prefix_state)
    written_rule_lhs = set()       # lhs of refuting rules the model RESTATED (before comp)
    wrote_advancing_intermediate = False
    saw_complement = False
    sw_strict = False              # STRICT: antecedent literally in S (premise/written)
    sw_closure = False             # LENIENT-a: antecedent closure-reachable in-world
    rule_written_any = False       # model restated a refuting rule at all (any lhs)

    for s in split_sentences(continuation or ""):
        st = strip_marker(s)
        f = parse_fact(st, entity)
        ru = parse_rule(st)
        if f is not None and f[1] == comp_pred:
            saw_complement = True
            for lhs in written_rule_lhs:
                if ("cat", lhs) in S:                      # antecedent written/premise
                    sw_strict = True
                if derivable(("cat", lhs), S, reach):      # antecedent world-reachable
                    sw_closure = True
            break  # adjudicate at first complement occurrence
        elif f is not None:
            p = f[1]
            if derivable(p, S, reach):
                # valid intermediate fact -> extend the TRUE state
                if p not in S:
                    S.add(p)
                    if p[0] == "cat":
                        dd = shortest_rule_distance(comp_pred, {p}, direct)
                        if dd is not None and (measured_d is None or dd < measured_d):
                            wrote_advancing_intermediate = True
            # invalid / poisoned facts are NOT added (keep S sound & true)
        elif ru is not None:
            if ru[1] == comp_pred:
                written_rule_lhs.add(ru[0])
                rule_written_any = True

    # PRIMARY licensed = strict shown-work (wrote the full valid chain).
    sw_primary = sw_strict
    # LENIENT upper bound: strict OR (rule restated w/ world-reachable antecedent)
    #                              OR wrote an advancing valid intermediate.
    sw_lenient = sw_strict or sw_closure or wrote_advancing_intermediate

    # ---- bucket assignment -------------------------------------------------
    if measured_d == 0:
        klass = "d0_visible"
        reason = "complement_is_given_premise"
    elif WD:  # measured_d in {1,2,3,5,...}
        if sw_primary:
            klass = "licensed"
            reason = "wrote_valid_refuting_derivation"
        elif measured_d <= 1:
            klass = "visible_unshown"
            reason = "derivable_le1hop_no_work_shown"
        else:
            klass = "unlicensed"
            reason = "derivable_ge2_unshown"
    else:  # measured_d is None
        klass = "unlicensed"
        reason = "not_derivable_in_world"

    return {
        "measured_d": measured_d,
        "WD": WD,
        "sw_primary": sw_primary,
        "sw_strict": sw_strict,
        "sw_closure": sw_closure,
        "sw_lenient": sw_lenient,
        "rule_written_any": rule_written_any,
        "wrote_advancing_intermediate": wrote_advancing_intermediate,
        "klass": klass,
        "klass_lenient": ("licensed" if (WD and measured_d != 0 and sw_lenient)
                          else klass),
        "reason": reason,
        "comp_pred": list(comp_pred),
        "n_refuting_rules_in_world": len(refuting_rules),
    }


# --------------------------------------------------------------------------- #
#  Loaders (join manifest + validated_outputs) and stated-complement recompute
# --------------------------------------------------------------------------- #
def load_expe():
    rd = os.path.join(EXP, "results", "EXPE_EVIDENCE_MOVER")
    man = {r["run_id"]: r for r in read_jsonl(os.path.join(rd, "manifest.jsonl"))}
    out = []
    for r in read_jsonl(os.path.join(rd, "validated_outputs.jsonl")):
        m = man.get(r["run_id"])
        if m is None or r.get("failed_generation"):
            continue
        stated = bool(r.get("stated_complement_of_falsehood"))
        arm = r["arm"]
        dmap = {"REFUTING_d1": 1, "REFUTING_d2": 2, "REFUTING_d3": 3}
        designed = dmap.get(arm, "inf")  # FREQ/TEMPLATE/FILLER -> inf (control)
        out.append({
            "exp": "EXPE",
            "cell": arm,
            "position": r["injection_position"],
            "designed_d": designed,
            "problem_id": m["problem_id"],
            "stated": stated,
            "question": m["question"],
            "entity": m["entity"],
            "planted": m["falsehood_statement"],
            "prefix_steps": m.get("prefix_steps", []),
            "continuation": r.get("continuation", ""),
            "run_id": r["run_id"],
            "added_rule": m.get("added_rule_sentence", ""),
        })
    return out


def _designed_from_condition(cond):
    for suf, d in (("_d0", 0), ("_d1", 1), ("_d2", 2), ("_d3", 3),
                   ("_d5", 5), ("_dinf", "inf")):
        if cond.endswith(suf):
            return d
    return "inf"


def load_expd():
    rd = os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT")
    man = {r["run_id"]: r for r in read_jsonl(os.path.join(rd, "manifest.jsonl"))}
    out = []
    for r in read_jsonl(os.path.join(rd, "validated_outputs.jsonl")):
        m = man.get(r["run_id"])
        if m is None or r.get("failed_generation"):
            continue
        cond = r.get("condition", "")
        entity = m["entity"]
        planted = m["injected_statement"]
        pred = statement_predicate(planted, entity)
        stated = False
        if pred is not None:
            comp = (entity, opposite_pred(pred))
            stated = any(parse_fact(strip_marker(s), entity) == comp
                         for s in split_sentences(r.get("continuation", "")))
        out.append({
            "exp": "EXPD",
            "cell": cond,
            "position": r.get("injection_position", ""),
            "designed_d": _designed_from_condition(cond),
            "problem_id": m["problem_id"],
            "stated": stated,
            "question": m["question"],
            "entity": entity,
            "planted": planted,
            "prefix_steps": m.get("prefix_steps", []),
            "continuation": r.get("continuation", ""),
            "run_id": r["run_id"],
            "added_rule": "",
        })
    return out


# --------------------------------------------------------------------------- #
#  Cluster bootstrap
# --------------------------------------------------------------------------- #
def cluster_bootstrap_rate(rows, numer_fn, denom_fn, nboot=NBOOT, seed=SEED):
    """Rate = sum(numer)/sum(denom) with clusters resampled by problem_id.
    Returns (point, lo, hi, numer_total, denom_total)."""
    by_cl = defaultdict(lambda: [0, 0])
    for r in rows:
        c = r["problem_id"]
        by_cl[c][0] += numer_fn(r)
        by_cl[c][1] += denom_fn(r)
    clusters = list(by_cl.values())
    tot_n = sum(a for a, _ in clusters)
    tot_d = sum(b for _, b in clusters)
    point = tot_n / tot_d if tot_d else None
    if not clusters or tot_d == 0:
        return point, None, None, tot_n, tot_d
    rng = random.Random(seed)
    K = len(clusters)
    boots = []
    for _ in range(nboot):
        sn = sd = 0
        for _ in range(K):
            a, b = clusters[rng.randrange(K)]
            sn += a; sd += b
        if sd > 0:
            boots.append(sn / sd)
    boots.sort()
    if not boots:
        return point, None, None, tot_n, tot_d
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1] if int(0.975 * len(boots)) > 0 else boots[-1]
    return point, lo, hi, tot_n, tot_d


def rnd(x, k=4):
    return round(x, k) if isinstance(x, float) else x


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    rows = load_expe() + load_expd()
    # classify every stated-complement row
    classified = []
    for r in rows:
        if not r["stated"]:
            r["klass"] = None
            continue
        c = classify_row(r["question"], r["entity"], r["planted"],
                         r["prefix_steps"], r["continuation"])
        r.update(c)
        classified.append(r)
    print(f"total rows={len(rows)}  stated-complement rows={len(classified)}")

    # write per-row classified JSONL
    with open(os.path.join(OUT, "classified_rows.jsonl"), "w") as f:
        for r in classified:
            rec = {k: r[k] for k in ("exp", "cell", "position", "designed_d",
                                     "problem_id", "run_id", "measured_d", "WD",
                                     "sw_primary", "sw_lenient", "rule_written_any",
                                     "wrote_advancing_intermediate", "klass",
                                     "klass_lenient", "reason", "comp_pred")}
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    KLASSES = ["licensed", "visible_unshown", "unlicensed", "d0_visible"]

    # ---------------- per-cell table ----------------
    results = {"per_cell": {}, "headline": {}, "cliff": {}, "meta": {
        "nboot": NBOOT, "seed": SEED, "cluster_unit": "problem_id"}}

    # index all rows by cell for absolute-rate denominators
    cell_all = defaultdict(list)
    for r in rows:
        cell_all[(r["exp"], r["cell"])].append(r)

    print("\n================= PER-CELL LICENSED SPLIT =================")
    print(f"{'exp':4s} {'cell':22s} {'dd':>4s} {'nAll':>5s} {'nSC':>4s} "
          f"{'lic':>4s} {'vis':>4s} {'unl':>4s} {'d0v':>4s}  "
          f"{'lic/SC':>7s}  {'lic_len/SC':>10s}")
    cell_index = defaultdict(list)
    for r in classified:
        cell_index[(r["exp"], r["cell"])].append(r)

    for key in sorted(cell_index):
        exp, cell = key
        crows = cell_index[key]
        n_all = len(cell_all[key])
        n_sc = len(crows)
        cnt = Counter(r["klass"] for r in crows)
        cnt_len = Counter(r["klass_lenient"] for r in crows)
        dd = crows[0]["designed_d"]
        # licensed fraction of stated-complement (cluster bootstrap)
        pt, lo, hi, _, _ = cluster_bootstrap_rate(
            crows, lambda r: int(r["klass"] == "licensed"), lambda r: 1)
        pt_len, lo_len, hi_len, _, _ = cluster_bootstrap_rate(
            crows, lambda r: int(r["klass_lenient"] == "licensed"), lambda r: 1)
        results["per_cell"][f"{exp}:{cell}"] = {
            "designed_d": dd, "n_all": n_all, "n_stated_complement": n_sc,
            "counts": {k: cnt.get(k, 0) for k in KLASSES},
            "counts_lenient_licensed": cnt_len.get("licensed", 0),
            "licensed_frac_of_SC": [rnd(pt), rnd(lo), rnd(hi)],
            "licensed_lenient_frac_of_SC": [rnd(pt_len), rnd(lo_len), rnd(hi_len)],
            "unlicensed_reasons": dict(Counter(r["reason"] for r in crows
                                               if r["klass"] == "unlicensed")),
        }
        print(f"{exp:4s} {cell:22s} {str(dd):>4s} {n_all:5d} {n_sc:4d} "
              f"{cnt.get('licensed',0):4d} {cnt.get('visible_unshown',0):4d} "
              f"{cnt.get('unlicensed',0):4d} {cnt.get('d0_visible',0):4d}  "
              f"{('-' if pt is None else format(pt,'.3f')):>7s}  "
              f"{('-' if pt_len is None else format(pt_len,'.3f')):>10s}")

    # ---------------- HEADLINE: fraction of one-hop-or-farther checking licensed
    # denominator: stated-complement rows with designed_d != 0 (exclude d0-visible)
    def is_farther(r):
        return r["designed_d"] != 0
    print("\n================= HEADLINE =================")
    for scope_name, subset in (
            ("ALL_experiments_d>=1", [r for r in classified if is_farther(r)]),
            ("EXPE_only_d>=1", [r for r in classified if is_farther(r) and r["exp"] == "EXPE"]),
            ("EXPD_only_d>=1", [r for r in classified if is_farther(r) and r["exp"] == "EXPD"]),
            ("EXPE_REFUTING_only", [r for r in classified if r["exp"] == "EXPE" and r["cell"].startswith("REFUTING")]),
    ):
        n = len(subset)
        pt, lo, hi, num, den = cluster_bootstrap_rate(
            subset, lambda r: int(r["klass"] == "licensed"), lambda r: 1)
        pt_l, lo_l, hi_l, num_l, _ = cluster_bootstrap_rate(
            subset, lambda r: int(r["klass_lenient"] == "licensed"), lambda r: 1)
        c = Counter(r["klass"] for r in subset)
        results["headline"][scope_name] = {
            "n_stated_complement": n,
            "counts": {k: c.get(k, 0) for k in KLASSES},
            "licensed_frac": [rnd(pt), rnd(lo), rnd(hi)],
            "licensed_lenient_frac": [rnd(pt_l), rnd(lo_l), rnd(hi_l)],
        }
        print(f"  {scope_name:22s} nSC={n:4d}  licensed={c.get('licensed',0):3d} "
              f"visible={c.get('visible_unshown',0):3d} unlicensed={c.get('unlicensed',0):3d} "
              f"| lic_frac={pt if pt is None else round(pt,3)} "
              f"[{None if lo is None else round(lo,3)},{None if hi is None else round(hi,3)}]"
              f"  lenient={None if pt_l is None else round(pt_l,3)}")

    # ---------------- d=0 cliff within licensed-only (aff_false_attr & REFUTING)
    # absolute rates per cell: stated-complement, licensed, visible, unlicensed
    # as fraction of ALL rows in the cell.
    print("\n========== d-CLIFF: absolute per-row rates by cell (frac of ALL rows) ==========")
    print(f"{'exp':4s} {'cell':22s} {'dd':>4s} {'nAll':>5s} "
          f"{'SC_rate':>8s} {'lic_rate':>9s} {'vis_rate':>9s} {'unl_rate':>9s} {'d0v_rate':>9s}")
    cliff_cells = [
        ("EXPD", "aff_false_attr_d0"), ("EXPD", "aff_false_attr_d1"),
        ("EXPD", "aff_false_attr_d2"), ("EXPD", "aff_false_attr_d3"),
        ("EXPD", "aff_false_attr_d5"),
        ("EXPD", "cat_false_inert_d1"), ("EXPD", "cat_false_inert_d3"),
        ("EXPD", "cat_false_inert_dinf"),
        ("EXPD", "cat_false_usable_d1"), ("EXPD", "cat_false_usable_d3"),
        ("EXPD", "cat_false_usable_dinf"),
        ("EXPE", "REFUTING_d1"), ("EXPE", "REFUTING_d2"), ("EXPE", "REFUTING_d3"),
        ("EXPE", "FREQ_MATCHED_NONREFUTING"), ("EXPE", "TEMPLATE_IRRELEVANT"),
    ]
    for key in cliff_cells:
        allr = cell_all.get(key, [])
        if not allr:
            continue
        n_all = len(allr)
        dd = allr[0]["designed_d"]
        crows = cell_index.get(key, [])
        by = Counter(r["klass"] for r in crows)

        def abs_rate(klass):
            pt, lo, hi, _, _ = cluster_bootstrap_rate(
                allr,
                lambda r, k=klass: int(r.get("stated") and r.get("klass") == k),
                lambda r: 1)
            return pt, lo, hi
        sc_pt, sc_lo, sc_hi, _, _ = cluster_bootstrap_rate(
            allr, lambda r: int(bool(r.get("stated"))), lambda r: 1)
        lic = abs_rate("licensed"); vis = abs_rate("visible_unshown")
        unl = abs_rate("unlicensed"); d0v = abs_rate("d0_visible")
        results["cliff"][f"{key[0]}:{key[1]}"] = {
            "designed_d": dd, "n_all": n_all,
            "sc_rate": [rnd(sc_pt), rnd(sc_lo), rnd(sc_hi)],
            "licensed_rate": [rnd(lic[0]), rnd(lic[1]), rnd(lic[2])],
            "visible_unshown_rate": [rnd(vis[0]), rnd(vis[1]), rnd(vis[2])],
            "unlicensed_rate": [rnd(unl[0]), rnd(unl[1]), rnd(unl[2])],
            "d0_visible_rate": [rnd(d0v[0]), rnd(d0v[1]), rnd(d0v[2])],
        }
        print(f"{key[0]:4s} {key[1]:22s} {str(dd):>4s} {n_all:5d} "
              f"{sc_pt:8.4f} {lic[0]:9.4f} {vis[0]:9.4f} {unl[0]:9.4f} {d0v[0]:9.4f}")

    with open(os.path.join(OUT, "e5_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(OUT,'e5_results.json')}")
    print(f"wrote {os.path.join(OUT,'classified_rows.jsonl')}")


if __name__ == "__main__":
    main()
