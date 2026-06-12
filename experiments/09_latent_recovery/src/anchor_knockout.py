"""Goal-anchor ablation (b): attention knockout on the goal-statement span.

During the perturbed continuation, attention_mask=0 on the prompt tokens spanning
"Prove: <target>." — the model cannot read the goal. If validated re-derivation
survives, recovery is not goal-anchor re-reading.

Families: paraphrase (benign) and negstep (genuine falsehood), mid injection.
Output: results/anchor_knockout/runs.jsonl + summary.
"""
import os, json, re, time
import torch
from common import (load_model, INSTR, FEWSHOT, split_sentences, solved, RESULTS, DATA)
from perturb import injection_points, make_corruption

OUT = os.path.join(RESULTS, "anchor_knockout")
FAMILIES = ["paraphrase", "negstep"]
POINT = "mid"

def prompt_with_goal_span(tok, question, target):
    user = INSTR + FEWSHOT + f"Q: {question} Prove: {target}\nA:"
    msgs = [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    goal_str = f"Prove: {target}"
    cstart = text.rindex(goal_str)          # the test question's goal, not few-shot
    cend = cstart + len(goal_str)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"])
            if a < cend and b > cstart]
    return torch.tensor([ids]), (span[0], span[-1] + 1)

@torch.no_grad()
def greedy_masked(tok, model, ids, span, prefix_ids=None, max_new=192):
    if prefix_ids is not None:
        ids = torch.cat([ids, prefix_ids], dim=1)
    ids = ids.to("cuda:0")
    mask = torch.ones_like(ids)
    mask[0, span[0]: span[1]] = 0
    out = model.generate(ids, attention_mask=mask, max_new_tokens=max_new,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def main():
    os.makedirs(OUT, exist_ok=True)
    tok, model = load_model()
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = [json.loads(l) for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))]
    gold = [g for g in gold if g["solved"]]
    # sanity: masked-goal UNPERTURBED continuation from an early prefix should still
    # complete the proof if goal-reading is not load-bearing; record this too.
    t0 = time.time()
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for i, g in enumerate(gold):
            inst = data[g["id"]]
            steps = split_sentences(g["gen_text"])
            pts = injection_points(steps)
            if pts is None:
                continue
            si = pts[POINT]
            ids, span = prompt_with_goal_span(tok, inst["question"], inst["target"])
            for fam in FAMILIES:
                corrupted = make_corruption(fam, inst["question"], steps, si, 1)
                if corrupted is None:
                    continue
                prefix = " " + " ".join(steps[:si] + [corrupted])
                pre_ids = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
                cont = greedy_masked(tok, model, ids, span, pre_ids)
                full_answer = (prefix + " " + cont).strip()
                fh.write(json.dumps({"id": g["id"], "family": fam, "point": POINT,
                                     "sent_idx": si, "corrupted_step": corrupted,
                                     "continuation": cont,
                                     "final_ok_naive": solved(full_answer, inst["target"]),
                                     }) + "\n")
                fh.flush()
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(gold)}] ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("DONE knockout", flush=True)

if __name__ == "__main__":
    main()
