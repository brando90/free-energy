"""Step 2: perturbed rollouts + aligned gold states.

For each solved instance: parse the model's OWN greedy CoT into sentences, find the
intermediate derived facts (entity-subject sentences, excluding the premise restatement
and the final conclusion), and inject a wrong-but-well-formed replacement at
early / mid / late points. Continue greedy decoding with no feedback.

Alignment contract: for both gold and perturbed rollouts we measure hidden states at
token offsets counted from the END of the (correct|corrupted) injected sentence in each
sequence's own tokenization. Prefix-tokenization stability is asserted per instance.

Output:
  results/perturbed/runs.jsonl
  results/perturbed/hs/{id}__{point}.npz       continuation states
  results/aligned_gold/{id}.npz                gold states + boundaries in meta
"""
import os, json, re, time
import torch
from common import (load_model, make_prompt_ids, greedy, solved, split_sentences, norm,
                    hidden_states_from, save_npz, wrong_category, make_distractor,
                    make_contradiction, make_paraphrase, make_falsehood, make_negstep,
                    DATA, RESULTS)

FAMILY = os.environ.get("PERT_FAMILY", "wrong")   # wrong | distractor | contradiction
OUT = os.path.join(RESULTS, "perturbed" if FAMILY == "wrong" else f"perturbed_{FAMILY}")
GOLD_ALIGNED = os.path.join(RESULTS, "aligned_gold")
SKIP_GOLD_STATES = os.environ.get("SKIP_GOLD_STATES", "0") == "1"
POINTS = ["early", "mid", "late"]

def make_corruption(family, question, steps, si, pi):
    if family == "wrong":
        return wrong_category(question, steps, steps[si], idx=pi)
    if family == "distractor":
        return make_distractor(question, steps, idx=pi)
    if family == "contradiction":
        return make_contradiction(steps, si)
    if family == "paraphrase":
        return make_paraphrase(steps, si, idx=pi)
    if family == "falsehood":
        return make_falsehood(question, steps, steps[si], idx=pi)
    if family == "negstep":
        return make_negstep(steps, si)
    if family.startswith("neghop"):
        from common import make_neghop
        return make_neghop(steps, si, int(family.replace("neghop", "")))
    raise ValueError(family)

def entity_sentences(steps):
    """Indices of sentences whose subject is the entity (capitalized single name)."""
    if not steps:
        return []
    ent = steps[0].split()[0]
    return [i for i, s in enumerate(steps) if s.split()[0] == ent]

def injection_points(steps):
    """Map early/mid/late -> sentence index among intermediate derived facts."""
    ents = entity_sentences(steps)
    inter = ents[1:-1]  # drop premise restatement and final conclusion
    if len(inter) < 3:
        return None
    return {"early": inter[0], "mid": inter[len(inter) // 2], "late": inter[-1]}

def main():
    os.makedirs(os.path.join(OUT, "hs"), exist_ok=True)
    os.makedirs(GOLD_ALIGNED, exist_ok=True)
    tok, model = load_model()
    rows = [json.loads(l) for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))]
    rows = [r for r in rows if r["solved"]]
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    t0 = time.time()
    n_ok = 0
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for i, r in enumerate(rows):
            inst = data[r["id"]]
            steps = split_sentences(r["gen_text"])
            pts = injection_points(steps)
            if pts is None:
                fh.write(json.dumps({"id": r["id"], "skip": "too_few_intermediate_steps"}) + "\n")
                continue
            safe = r["id"].replace("/", "_").replace("::", "__")

            # --- aligned gold states, computed once in re-tokenized space ---
            full_ids = make_prompt_ids(tok, inst["question"], inst["target"],
                                       answer_prefix=" " + " ".join(steps))[0]
            boundaries, stable = {}, True
            for name, si in pts.items():
                pref_ids = make_prompt_ids(tok, inst["question"], inst["target"],
                                           answer_prefix=" " + " ".join(steps[:si + 1]))[0]
                if not torch.equal(full_ids[: len(pref_ids)], pref_ids):
                    stable = False
                    break
                boundaries[name] = int(len(pref_ids))
            if not stable:
                fh.write(json.dumps({"id": r["id"], "skip": "tokenization_not_prefix_stable"}) + "\n")
                continue
            if not SKIP_GOLD_STATES:
                ans_start = int(make_prompt_ids(tok, inst["question"], inst["target"]).shape[1])
                hs = hidden_states_from(tok, model, full_ids, ans_start)
                save_npz(os.path.join(GOLD_ALIGNED, safe + ".npz"), hs,
                         id=r["id"], ans_start=ans_start, boundaries=boundaries,
                         gold_len=int(len(full_ids)))

            # --- perturbed rollouts ---
            for pi, (name, si) in enumerate(pts.items()):
                correct = steps[si]
                corrupted = make_corruption(FAMILY, inst["question"], steps, si, pi)
                if corrupted is None:
                    fh.write(json.dumps({"id": r["id"], "point": name,
                                         "skip": "no_corruption_candidate"}) + "\n")
                    continue
                prefix = " " + " ".join(steps[:si] + [corrupted])
                ids = make_prompt_ids(tok, inst["question"], inst["target"], answer_prefix=prefix)
                gen_text, full = greedy(tok, model, ids, max_new=192)
                full_answer = (prefix + " " + gen_text).strip()
                rec_flag = solved(full_answer, inst["target"])
                hs_p = hidden_states_from(tok, model, full, ids.shape[1])
                save_npz(os.path.join(OUT, "hs", f"{safe}__{name}.npz"), hs_p,
                         id=r["id"], point=name, boundary=int(ids.shape[1]))
                fh.write(json.dumps({
                    "id": r["id"], "point": name, "sent_idx": si,
                    "correct_step": correct, "corrupted_step": corrupted,
                    "continuation": gen_text, "recovered": rec_flag,
                    "n_steps_gold": len(steps),
                    "boundary": int(ids.shape[1]),
                }) + "\n")
                fh.flush()
            n_ok += 1
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(rows)}] processed={n_ok} ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"DONE perturb: instances={n_ok} ({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
