"""Exp-1 (Batch-4): Context-controlled checkability.

Decompose "inferential distance" into two orthogonal factors, exploiting that the
rule closure is order-invariant (relocating a rule sentence in the prompt leaves the
logical world identical, changing only token positions):

  G (graph distance):  neghop k in {1,2,3} at FIXED early injection. Logical cost.
  T (token distance):  negstep (k=1, fixed proposition+position), the SAME world, with
                       only the licensing rule's prompt position varied
                       {front, mid, back} + random rule permutations. Recall cost.

Claim: decay is in G, flat in T. The T arm rules out lost-in-the-middle / attention-sink
recall failure as the driver of the dose-response.

Cohort = gold-solved instances whose gold rollout formally validates (headline cohort).
Output: results_ctx/runs.jsonl + results_ctx/summary.json
Model/decoding identical to perturb.py (greedy, no feedback, same budget).
"""
import os, json, re, time, random
from common import (load_model, make_prompt_ids, greedy, split_sentences, DATA, RESULTS,
                    make_neghop, make_negstep)
from perturb import injection_points
from validator import (parse_world, parse_rule, parse_fact, closure, derivable,
                       validate_continuation, DOUBT)

OUT = os.path.join(RESULTS, "..", "results_ctx")
RNG = random.Random(0)            # single fixed seed (decoding is greedy anyway)
N_PERM = 4                        # random rule permutations for the order-robustness arm


def sentences(question):
    return [s.strip() for s in re.split(r"(?<=\.)\s+", question.strip()) if s.strip()]


def rebuild(sents):
    return " ".join(sents)


def find_licensing_rule(question, correct_step, entity):
    """The rule sentence that licenses `correct_step` (E is [not] PRED): its rhs == PRED
    and its lhs is a category provably held by the entity from premises alone. Returns
    the original rule sentence string, or None."""
    rules, facts, _ = parse_world(question)
    reach = closure(rules)
    f = parse_fact(correct_step, entity)
    if not f:
        return None
    pred = f[1]
    premises = {p for e, p in facts if e == entity}
    cands = []
    for s in sentences(question):
        r = parse_rule(s)
        if r and r[1] == pred:
            lhs = ("cat", r[0])
            if lhs in premises or derivable(lhs, premises, reach):
                cands.append(s)
    return cands[0] if cands else None


def move_rule(question, rule_sentence, position):
    """Relocate one rule sentence within the question. position in {front,mid,back}.
    Closure is unchanged; only token offsets move. 'back' = just before the first fact."""
    sents = sentences(question)
    if rule_sentence not in sents:
        return None, None
    sents.remove(rule_sentence)
    fact_idxs = [i for i, s in enumerate(sents) if parse_fact(s) and not parse_rule(s)]
    first_fact = min(fact_idxs) if fact_idxs else len(sents)
    if position == "front":
        ins = 0
    elif position == "mid":
        ins = first_fact // 2
    else:  # back
        ins = first_fact
    sents.insert(ins, rule_sentence)
    new_q = rebuild(sents)
    off = new_q.index(rule_sentence) / max(1, len(new_q))   # fractional char offset
    return new_q, round(off, 3)


def permute_rules(question):
    """Shuffle rule sentences among themselves; facts kept in place. World identical."""
    sents = sentences(question)
    rule_pos = [i for i, s in enumerate(sents) if parse_rule(s)]
    rules = [sents[i] for i in rule_pos]
    RNG.shuffle(rules)
    for i, p in enumerate(rule_pos):
        sents[p] = rules[i]
    return rebuild(sents)


def run_one(tok, model, question, target, steps, si, corrupted, entity):
    prefix = " " + " ".join(steps[:si] + [corrupted])
    ids = make_prompt_ids(tok, question, target, answer_prefix=prefix)
    gen_text, _ = greedy(tok, model, ids, max_new=192)
    v = validate_continuation(question, steps[:si], corrupted, gen_text, target, entity)
    return gen_text, v


def main():
    os.makedirs(OUT, exist_ok=True)
    tok, model = load_model()
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = [json.loads(l) for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))]
    gold = [g for g in gold if g["solved"]]
    limit = int(os.environ.get("LR_LIMIT", "0"))
    if limit:
        gold = gold[:limit]

    t0 = time.time()
    n = 0
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for gi, g in enumerate(gold):
            inst = data[g["id"]]
            q0, target = inst["question"], inst["target"]
            steps = split_sentences(g["gen_text"])
            entity = steps[0].split()[0] if steps else None
            # cohort gate: gold must formally validate
            gv = validate_continuation(q0, [], None, g["gen_text"], target, entity)
            if gv["class"] != "valid_rederivation":
                continue
            pts = injection_points(steps)
            if pts is None:
                continue
            si = pts["early"]                      # FIXED injection position
            correct = steps[si]

            # ---- G arm: neghop k = 1,2,3 (logical distance), original question ----
            for k in (1, 2, 3):
                corrupted = make_neghop(steps, si, k)
                if corrupted is None:
                    continue
                gen, v = run_one(tok, model, q0, target, steps, si, corrupted, entity)
                fh.write(json.dumps({"id": g["id"], "arm": "G", "k": k, "tbucket": None,
                                     "corrupted_step": corrupted, "class": v["class"],
                                     "acknowledged": v["acknowledged"],
                                     "lexical_doubt": bool(DOUBT.search(gen))}) + "\n")

            # ---- T arm: negstep (k=1) with licensing rule relocated ----
            corrupted = make_negstep(steps, si)
            if corrupted is not None:
                rule = find_licensing_rule(q0, correct, entity)
                for pos in ("front", "mid", "back"):
                    q = q0
                    off = None
                    if rule is not None:
                        q, off = move_rule(q0, rule, pos)
                        if q is None:
                            q, off = q0, None
                    gen, v = run_one(tok, model, q, target, steps, si, corrupted, entity)
                    fh.write(json.dumps({"id": g["id"], "arm": "T", "k": 1, "tbucket": pos,
                                         "rule_offset": off, "found_rule": rule is not None,
                                         "corrupted_step": corrupted, "class": v["class"],
                                         "acknowledged": v["acknowledged"],
                                         "lexical_doubt": bool(DOUBT.search(gen))}) + "\n")
                # random-permutation order-robustness arm
                for pi in range(N_PERM):
                    q = permute_rules(q0)
                    gen, v = run_one(tok, model, q, target, steps, si, corrupted, entity)
                    fh.write(json.dumps({"id": g["id"], "arm": "Tperm", "k": 1,
                                         "tbucket": f"perm{pi}", "corrupted_step": corrupted,
                                         "class": v["class"], "acknowledged": v["acknowledged"],
                                         "lexical_doubt": bool(DOUBT.search(gen))}) + "\n")
            fh.flush()
            n += 1
            if n % 20 == 0:
                print(f"[{gi+1}/{len(gold)}] cohort={n} ({(time.time()-t0)/60:.1f} min)", flush=True)
    summarize()
    print(f"DONE context_check: cohort={n} ({(time.time()-t0)/60:.1f} min)", flush=True)


def summarize():
    from collections import defaultdict
    rows = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))]
    cells = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["arm"] == "G":
            key = f"G_k{r['k']}"
        elif r["arm"] == "T":
            key = f"T_{r['tbucket']}"
        else:
            key = "Tperm"
        c = cells[key]
        c["n"] += 1
        c[r["class"]] += 1
        c["doubt"] += int(r["lexical_doubt"])
    out = {}
    for k, c in cells.items():
        n = c["n"]
        out[k] = {"n": n,
                  "valid": round(c.get("valid_rederivation", 0) / n, 3),
                  "poisoned": round(c.get("poisoned", 0) / n, 3),
                  "parroted": round(c.get("parroted", 0) / n, 3),
                  "derailed": round(c.get("derailed", 0) / n, 3),
                  "doubt": round(c.get("doubt", 0) / n, 3)}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
