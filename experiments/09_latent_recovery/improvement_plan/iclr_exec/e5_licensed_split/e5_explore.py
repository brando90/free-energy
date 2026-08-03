"""E5 exploration/validation step 1.
- Join manifest + validated_outputs for EXPE and EXPD.
- Reproduce published stated-complement rates per cell (SANITY ANCHORS).
- Cross-check stored measured_d (world-derivability of the complement) vs a fresh
  recomputation using the canonical closure machinery (src/validator + expc helpers).
No new file writes except stdout.
"""
import sys, os, json
from collections import defaultdict, Counter

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expe"))

import validator as V
from validator import parse_fact, strip_marker
from expc_polarity_control import opposite_pred, statement_predicate  # noqa
# EXPE helpers (world building + shortest_rule_distance) — canonical
from expe_evidence_mover import (question_world, shortest_rule_distance,
                                 direct_rule_map, split_sentences)


def read_jsonl(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_join(resdir, manifest_name="manifest.jsonl", val_name="validated_outputs.jsonl"):
    man = {r["run_id"]: r for r in read_jsonl(os.path.join(resdir, manifest_name))}
    rows = []
    for r in read_jsonl(os.path.join(resdir, val_name)):
        m = man.get(r["run_id"])
        rows.append((r, m))
    return rows, man


def rate(num, den):
    return round(num / den, 4) if den else None


# ---------------- EXPE ----------------
def expe():
    resdir = os.path.join(EXP, "results", "EXPE_EVIDENCE_MOVER")
    rows, man = load_join(resdir)
    print(f"[EXPE] validated rows={len(rows)}  manifest rows={len(man)}")
    # cell = (arm, position)
    by_cell = defaultdict(lambda: {"n": 0, "sc": 0})
    for r, m in rows:
        if r.get("failed_generation"):
            continue
        cell = (r["arm"], r["injection_position"])
        by_cell[cell]["n"] += 1
        if r.get("stated_complement_of_falsehood"):
            by_cell[cell]["sc"] += 1
    print("\n[EXPE] stated-complement rate per (arm,position)  [ANCHOR check]")
    for cell in sorted(by_cell):
        d = by_cell[cell]
        print(f"  {cell[0]:28s}|{cell[1]:5s}  n={d['n']:3d}  sc={d['sc']:3d}  rate={rate(d['sc'],d['n'])}")

    # Cross-check measured_d_augmented (stored) vs recomputed complement closure distance
    print("\n[EXPE] measured_d_augmented cross-check (stored vs recomputed shortest_rule_distance)")
    mism = 0
    checked = 0
    by_arm_d = defaultdict(Counter)
    for r, m in rows:
        if r.get("failed_generation"):
            continue
        stored = m.get("measured_d_augmented")
        # recompute
        w = question_world(m["question"], m["entity"], m.get("prefix_steps", ()))
        pred = statement_predicate(m["falsehood_statement"], m["entity"])
        comp = opposite_pred(pred)
        d_re = shortest_rule_distance(comp, w["prefix_state"], w["direct"])
        checked += 1
        by_arm_d[r["arm"]][(stored, d_re)] += 1
        if stored != d_re:
            mism += 1
            if mism <= 8:
                print(f"   MISMATCH arm={r['arm']} stored={stored} recomputed={d_re} pid={m['problem_id'][:40]}")
    print(f"   checked={checked} mismatches={mism}")
    print("   per-arm (stored,recomputed) distribution:")
    for arm in sorted(by_arm_d):
        print(f"     {arm:28s} {dict(by_arm_d[arm])}")


# ---------------- EXPD ----------------
def expd():
    resdir = os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT")
    rows, man = load_join(resdir)
    print(f"\n[EXPD] validated rows={len(rows)}  manifest rows={len(man)}")
    # recompute stated_complement the way the EXPD analysis does
    by_cell = defaultdict(lambda: {"n": 0, "sc": 0})
    for r, m in rows:
        if r.get("failed_generation") or m is None:
            continue
        cell = r.get("cell_group") or r.get("condition")
        # emulate EXPD cell naming: condition already like aff_false_attr_d0 etc?
        cond = r.get("condition")
        pred = statement_predicate(m["injected_statement"], m["entity"])
        sc = False
        if pred is not None:
            comp = (m["entity"], opposite_pred(pred))
            sc = any(parse_fact(strip_marker(s), m["entity"]) == comp
                     for s in split_sentences(r.get("continuation", "")))
        by_cell[cond]["n"] += 1
        if sc:
            by_cell[cond]["sc"] += 1
    print("\n[EXPD] stated-complement rate per condition  [ANCHOR check: aff_false_attr_d0=0.2467, _d1=0.0311, cat_false_inert_d1=0.1933, dinf=0.0233]")
    for cell in sorted(by_cell):
        d = by_cell[cell]
        print(f"  {cell:28s}  n={d['n']:3d}  sc={d['sc']:3d}  rate={rate(d['sc'],d['n'])}")

    # measured_d cross check for EXPD
    print("\n[EXPD] measured_d_from_prefix cross-check")
    mism = 0; checked = 0
    for r, m in rows:
        if r.get("failed_generation") or m is None:
            continue
        stored = m.get("measured_d_from_prefix")
        w = question_world(m["question"], m["entity"], m.get("prefix_steps", ()))
        pred = statement_predicate(m["injected_statement"], m["entity"])
        comp = opposite_pred(pred)
        d_re = shortest_rule_distance(comp, w["prefix_state"], w["direct"])
        checked += 1
        if stored != d_re:
            mism += 1
            if mism <= 10:
                print(f"   MISMATCH cond={r.get('condition')} stored={stored} recomputed={d_re}")
    print(f"   checked={checked} mismatches={mism}")


if __name__ == "__main__":
    expe()
    expd()
