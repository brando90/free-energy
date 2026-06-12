"""Goal-anchor ablation (a): true/false QA format — statement visible, truth unknown.

Half the instances query the entailed target ("answer: true"), half its negation
("answer: false"), assigned deterministically. The model must derive, then judge.
Gold collection on the QA format, then mid-injection of {paraphrase, negstep} into the
model's own QA rollouts. Recovery = correct verdict + post-injection chain soundness.

Output: results/anchor_qa/{gold.jsonl,runs.jsonl,summary.json}
"""
import os, json, re, time, hashlib
import torch
from common import (load_model, split_sentences, RESULTS, DATA, make_paraphrase,
                    make_negstep)
from validator import parse_world, parse_fact, closure, derivable, strip_marker, DOUBT
from perturb import injection_points

OUT = os.path.join(RESULTS, "anchor_qa")
POINT = "mid"

QA_FEWSHOT = """Q: Every yumpus is a dumpus. Dumpuses are tumpuses. Tumpuses are not bright. Sam is a yumpus. True or false: Sam is not bright?
A: Sam is a yumpus. Every yumpus is a dumpus. Sam is a dumpus. Dumpuses are tumpuses. Sam is a tumpus. Tumpuses are not bright. Sam is not bright. So the statement is true.

Q: Each gorpus is a sterpus. Sterpuses are red. Every borpus is a gorpus. Alex is a borpus. True or false: Alex is not red?
A: Alex is a borpus. Every borpus is a gorpus. Alex is a gorpus. Each gorpus is a sterpus. Alex is a sterpus. Sterpuses are red. Alex is red. So the statement is false.

"""
QA_INSTR = ("You will be given facts and rules about fictional creatures, then a statement to judge. "
            "Reason step by step in the exact style of the examples, then conclude with "
            "'So the statement is true.' or 'So the statement is false.'\n\n")

VERDICT_RE = re.compile(r"statement is (true|false)", re.I)

def negate(stmt):
    return make_negstep([stmt], 0)

def qa_ids(tok, question, stmt, prefix=None):
    user = QA_INSTR + QA_FEWSHOT + f"Q: {question} True or false: {stmt.rstrip('.')}?\nA:"
    ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                  add_generation_prompt=True, return_tensors="pt")["input_ids"]
    if prefix:
        pre = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([ids, pre], dim=1)
    return ids

@torch.no_grad()
def greedy(tok, model, ids, max_new=256):
    out = model.generate(ids.to("cuda:0"), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def chain_sound(question, entity, prefix_steps, continuation):
    rules, facts, _ = parse_world(question)
    reach = closure(rules)
    state = {p for e, p in facts if e == entity}
    for s in prefix_steps:
        f = parse_fact(strip_marker(s), entity)
        if f:
            state.add(f[1])
    n_fact = bad = 0
    for s in split_sentences(continuation):
        s = strip_marker(s)
        f = parse_fact(s, entity)
        if f:
            n_fact += 1
            if derivable(f[1], state, reach):
                state.add(f[1])
            else:
                bad += 1
    return n_fact > 0 and bad == 0

def main():
    os.makedirs(OUT, exist_ok=True)
    tok, model = load_model()
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    cohort = {json.loads(l)["id"] for l in open(os.path.join(RESULTS, "validated.jsonl"))}
    insts = [data[i] for i in sorted(cohort)]
    t0 = time.time()

    # ---- QA gold ----
    gold_rows = []
    for inst in insts:
        h = int(hashlib.md5(inst["id"].encode()).hexdigest(), 16) % 2
        stmt = inst["target"] if h == 0 else negate(inst["target"])
        truth = "true" if h == 0 else "false"
        if stmt is None:
            continue
        txt = greedy(tok, model, qa_ids(tok, inst["question"], stmt))
        m = VERDICT_RE.search(txt)
        verdict = m.group(1).lower() if m else None
        steps = split_sentences(txt)
        chain = [s for s in steps if not VERDICT_RE.search(s)]
        entity = chain[0].split()[0] if chain else None
        ok = (verdict == truth) and entity and chain_sound(inst["question"], entity, [], " ".join(chain))
        gold_rows.append({"id": inst["id"], "stmt": stmt, "truth": truth, "gen": txt,
                          "verdict": verdict, "solved": bool(ok)})
    with open(os.path.join(OUT, "gold.jsonl"), "w") as fh:
        for g in gold_rows:
            fh.write(json.dumps(g) + "\n")
    solved = [g for g in gold_rows if g["solved"]]
    print(f"QA gold: solved {len(solved)}/{len(gold_rows)} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- mid-injection on QA rollouts ----
    out_rows = []
    for g in solved:
        inst = data[g["id"]]
        steps = split_sentences(g["gen"])
        chain = [s for s in steps if not VERDICT_RE.search(s)]
        pts = injection_points(chain)
        if pts is None:
            continue
        si = pts[POINT]
        for fam, mk in [("paraphrase", lambda: make_paraphrase(chain, si, 1)),
                        ("negstep", lambda: make_negstep(chain, si))]:
            corrupted = mk()
            if corrupted is None:
                continue
            prefix = " " + " ".join(chain[:si] + [corrupted])
            cont = greedy(tok, model, qa_ids(tok, inst["question"], g["stmt"], prefix=prefix), max_new=192)
            m = VERDICT_RE.search(cont)
            verdict = m.group(1).lower() if m else None
            entity = chain[0].split()[0]
            sound = chain_sound(inst["question"], entity, chain[:si], cont)
            out_rows.append({"id": g["id"], "family": fam, "stmt": g["stmt"],
                             "truth": g["truth"], "corrupted_step": corrupted,
                             "continuation": cont, "verdict": verdict,
                             "verdict_ok": verdict == g["truth"],
                             "chain_sound": bool(sound),
                             "recovered": (verdict == g["truth"]) and bool(sound),
                             "ack": bool(DOUBT.search(cont))})
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    summary = {}
    for fam in ["paraphrase", "negstep"]:
        sub = [r for r in out_rows if r["family"] == fam]
        if not sub:
            continue
        summary[fam] = {"n": len(sub),
                        "verdict_ok": round(sum(r["verdict_ok"] for r in sub) / len(sub), 4),
                        "chain_sound": round(sum(r["chain_sound"] for r in sub) / len(sub), 4),
                        "recovered": round(sum(r["recovered"] for r in sub) / len(sub), 4),
                        "ack": round(sum(r["ack"] for r in sub) / len(sub), 4)}
    summary["_gold"] = {"solved": len(solved), "attempted": len(gold_rows)}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE anchor_qa", flush=True)

if __name__ == "__main__":
    main()
