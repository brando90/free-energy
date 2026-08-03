"""Targeted validation: eyeball the STRICT classifier's licensed vs unlicensed
calls at d>=2 (aff_false_attr_d2, cat_false_inert_d3, aff_false_attr_d3), which
is where the current classifier diverges most from the stale results JSON."""
import sys, os, random
EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expe"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "iclr_exec", "e5_licensed_split"))
from validator import parse_fact, parse_rule, strip_marker, derivable
from expc_polarity_control import opposite_pred, statement_predicate
from expe_evidence_mover import question_world, shortest_rule_distance, split_sentences
from e5_classify import classify_row, load_expe, load_expd


def trace(r):
    entity = r["entity"]
    pred = statement_predicate(r["planted"], entity)
    comp = opposite_pred(pred)
    w = question_world(r["question"], entity, r["prefix_steps"] or ())
    S = set(w["prefix_state"]); reach = w["reach"]
    print(f"    entity={entity} planted='{r['planted']}' comp={comp}")
    print(f"    prefix_state(given cats)={sorted(a for a in w['prefix_state'] if a[0]=='cat')}")
    print(f"    measured_d={r['measured_d']} KLASS={r['klass']} reason={r['reason']} "
          f"sw_strict={r['sw_strict']} sw_closure={r['sw_closure']} advint={r['wrote_advancing_intermediate']}")
    for s in split_sentences(r["continuation"] or ""):
        st = strip_marker(s)
        f = parse_fact(st, entity); ru = parse_rule(st)
        tag = ""
        if f is not None and f[1] == comp:
            tag = "  <<<< COMPLEMENT"
        elif f is not None:
            dv = derivable(f[1], S, reach)
            tag = f"  [fact {f[1]} deriv={dv}]"
            if dv:
                S.add(f[1])
        elif ru is not None:
            mark = " *REFUTING*" if ru[1] == comp else ""
            tag = f"  [RULE {ru[0]}->{ru[1]}{mark}]"
        print(f"       | {s.strip()[:64]:64s}{tag}")
        if f is not None and f[1] == comp:
            break


def main():
    rows = load_expe() + load_expd()
    classified = []
    for r in rows:
        if not r["stated"]:
            continue
        c = classify_row(r["question"], r["entity"], r["planted"], r["prefix_steps"], r["continuation"])
        r.update(c)
        classified.append(r)
    rng = random.Random(11)
    targets = [
        ("EXPD", "aff_false_attr_d2", "licensed", 2),
        ("EXPD", "aff_false_attr_d2", "unlicensed", 4),
        ("EXPD", "cat_false_inert_d3", "licensed", 2),
        ("EXPD", "cat_false_inert_d3", "unlicensed", 4),
        ("EXPD", "aff_false_attr_d3", "unlicensed", 2),
    ]
    for exp, cell, klass, k in targets:
        pool = [r for r in classified if r["exp"] == exp and r["cell"] == cell and r["klass"] == klass]
        rng.shuffle(pool)
        print(f"\n########## {exp} {cell} klass={klass}  (n={len(pool)}, showing {min(k,len(pool))}) ##########")
        for r in pool[:k]:
            print(f"  --- pid={r['problem_id'][:40]} run={r['run_id'][:10]}")
            trace(r)


if __name__ == "__main__":
    main()
