"""E5 exploration step 2: verify recomputed complement closure-distance on the
EXPD FALSE cells (the only cells that produce stated complements), and inspect a
few REFUTING/FREQ continuations to design the licensed/unlicensed classifier.
"""
import sys, os, json
from collections import defaultdict, Counter

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expe"))
import validator as V
from validator import parse_fact, parse_rule, strip_marker
from expc_polarity_control import opposite_pred, statement_predicate
from expe_evidence_mover import question_world, shortest_rule_distance, split_sentences


def read_jsonl(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_join(resdir):
    man = {r["run_id"]: r for r in read_jsonl(os.path.join(resdir, "manifest.jsonl"))}
    rows = [(r, man.get(r["run_id"])) for r in read_jsonl(os.path.join(resdir, "validated_outputs.jsonl"))]
    return rows


FALSE_CELLS = ("aff_false_attr", "neg_false_attr", "cat_false_inert", "cat_false_usable")

def expd_falsecheck():
    rows = load_join(os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT"))
    dist = defaultdict(Counter)
    for r, m in rows:
        if r.get("failed_generation") or m is None:
            continue
        cond = r.get("condition", "")
        if not any(cond.startswith(fc) for fc in FALSE_CELLS):
            continue
        stored = m.get("measured_d_from_prefix")
        w = question_world(m["question"], m["entity"], m.get("prefix_steps", ()))
        pred = statement_predicate(m["injected_statement"], m["entity"])
        comp = opposite_pred(pred)
        d_re = shortest_rule_distance(comp, w["prefix_state"], w["direct"])
        dist[cond][(stored, d_re)] += 1
    print("[EXPD false cells] (stored measured_d_from_prefix, recomputed complement dist) counts:")
    for cond in sorted(dist):
        print(f"  {cond:24s} {dict(dist[cond])}")


def show_examples():
    rows = load_join(os.path.join(EXP, "results", "EXPE_EVIDENCE_MOVER"))
    print("\n=== EXPE example stated-complement continuations ===")
    shown = {"REFUTING_d1": 0, "FREQ_MATCHED_NONREFUTING": 0, "REFUTING_d2": 0, "REFUTING_d3": 0}
    for r, m in rows:
        if r.get("failed_generation") or not r.get("stated_complement_of_falsehood"):
            continue
        arm = r["arm"]
        if arm not in shown or shown[arm] >= 4:
            continue
        shown[arm] += 1
        entity = m["entity"]; fcat = m["falsehood_category"]
        comp = (entity, ("not_cat", fcat))
        w = question_world(m["question"], entity, m.get("prefix_steps", ()))
        # parse continuation sentences, tag each
        print(f"\n--- arm={arm} pid={m['problem_id'][:34]} entity={entity} F='{m['falsehood_statement']}' addedrule='{m['added_rule_sentence']}'")
        print(f"    prefix_steps={m['prefix_steps']}  measured_d_aug={m.get('measured_d_augmented')}")
        state = set(w["prefix_state"]); reach = w["reach"]
        for s in split_sentences(r.get("continuation", "")):
            st = strip_marker(s)
            f = parse_fact(st, entity)
            ru = parse_rule(st)
            tag = ""
            if f == comp:
                supported = V.derivable(comp, state, reach)
                tag = f"  <<COMPLEMENT  derivable_from_curstate={supported}"
            elif f is not None:
                dv = V.derivable(f[1], state, reach)
                tag = f"  [fact {f[1]} derivable={dv}]"
                if dv:
                    state.add(f[1])
            elif ru is not None:
                head = ru[1]
                tag = f"  [RULE {ru[0]}->{ru[1]}]"
            print(f"      | {s.strip()[:70]:70s}{tag}")


if __name__ == "__main__":
    expd_falsecheck()
    show_examples()
