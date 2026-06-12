"""R1-Distill-Qwen-7B arm: is silence an alignment artifact?

Pre-registered (PLAN.md 2026-06-11): injection into the VISIBLE proof body; think
block stripped before validation; doubt measured separately per channel; primary =
doubt rises vs Qwen2.5-7B-Instruct, validated recovery flat. Private-channel doubt
during perturbed continuations is NOT measurable in this design (continuation is
prefilled past </think>); think-channel stats come from gold rollouts only.

Output: results/r1/{gold.jsonl,runs.jsonl,summary.json}
"""
import os, json, re, time
import torch
from common import (INSTR, FEWSHOT, split_sentences, solved, RESULTS, DATA,
                    make_paraphrase, make_negstep)
from validator import validate_continuation, DOUBT
from perturb import injection_points

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUT = os.path.join(RESULTS, "r1")
MAX_SOLVED = 150
POINT = "mid"

def load_r1():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    return tok, model

def split_think(text):
    if "</think>" in text:
        th, vis = text.split("</think>", 1)
        return th.replace("<think>", "").strip(), vis.strip()
    return "", text.strip()

def gold_ids(tok, question, target):
    user = (INSTR + FEWSHOT + f"Q: {question} Prove: {target}\n"
            "Answer with only the proof steps in the style of the examples.")
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   add_generation_prompt=True,
                                   return_tensors="pt")["input_ids"]

def cont_ids(tok, question, target, think, vis_prefix):
    user = (INSTR + FEWSHOT + f"Q: {question} Prove: {target}\n"
            "Answer with only the proof steps in the style of the examples.")
    ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                  add_generation_prompt=True,
                                  return_tensors="pt")["input_ids"]
    pre = tok(f"{think}\n</think>\n\n{vis_prefix}", return_tensors="pt",
              add_special_tokens=False)["input_ids"]
    return torch.cat([ids, pre], dim=1)

@torch.no_grad()
def greedy(tok, model, ids, max_new):
    out = model.generate(ids.to("cuda:0"), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def main():
    os.makedirs(OUT, exist_ok=True)
    tok, model = load_r1()
    data = [json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))]
    t0 = time.time()

    gold_rows, n_solved = [], 0
    for i, inst in enumerate(data):
        if n_solved >= MAX_SOLVED:
            break
        txt = greedy(tok, model, gold_ids(tok, inst["question"], inst["target"]), 1500)
        think, vis = split_think(txt)
        ok = solved(vis, inst["target"])
        gold_rows.append({"id": inst["id"], "think": think, "visible": vis,
                          "solved": ok, "truncated": "</think>" not in txt,
                          "think_doubt": bool(DOUBT.search(think)),
                          "visible_doubt": bool(DOUBT.search(vis))})
        n_solved += int(ok)
        if (i + 1) % 25 == 0:
            print(f"[{i+1}] solved={n_solved} ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(os.path.join(OUT, "gold.jsonl"), "w") as fh:
        for g in gold_rows:
            fh.write(json.dumps(g) + "\n")
    solved_rows = [g for g in gold_rows if g["solved"]]
    print(f"R1 gold: solved {len(solved_rows)}/{len(gold_rows)} "
          f"truncated={sum(g['truncated'] for g in gold_rows)}", flush=True)

    dmap = {d["id"]: d for d in data}
    out_rows = []
    for g in solved_rows:
        inst = dmap[g["id"]]
        steps = split_sentences(g["visible"])
        pts = injection_points(steps)
        if pts is None:
            continue
        si = pts[POINT]
        entity = steps[0].split()[0]
        for fam, mk in [("paraphrase", lambda: make_paraphrase(steps, si, 1)),
                        ("negstep", lambda: make_negstep(steps, si))]:
            corrupted = mk()
            if corrupted is None:
                continue
            vis_prefix = " ".join(steps[:si] + [corrupted])
            cont = greedy(tok, model, cont_ids(tok, inst["question"], inst["target"],
                                               g["think"], vis_prefix), 900)
            _, cont_vis = split_think(cont)   # in case it re-opens a think block
            v = validate_continuation(inst["question"], steps[:si], corrupted,
                                      cont_vis, inst["target"], entity)
            out_rows.append({"id": g["id"], "family": fam, "sent_idx": si,
                             "corrupted_step": corrupted, "continuation": cont_vis,
                             "reopened_think": "<think>" in cont, **v})
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    summary = {"_gold": {"solved": len(solved_rows), "attempted": len(gold_rows),
                         "gold_think_doubt_rate": round(
                             sum(g["think_doubt"] for g in solved_rows) / max(1, len(solved_rows)), 4),
                         "gold_visible_doubt_rate": round(
                             sum(g["visible_doubt"] for g in solved_rows) / max(1, len(solved_rows)), 4)}}
    for fam in ["paraphrase", "negstep"]:
        sub = [r for r in out_rows if r["family"] == fam]
        if not sub:
            continue
        n = len(sub)
        summary[fam] = {"n": n,
                        "valid_rederivation": round(sum(r["class"] == "valid_rederivation" for r in sub) / n, 4),
                        "derailed": round(sum(r["class"] == "derailed" for r in sub) / n, 4),
                        "unparsed": round(sum(r["class"] == "unparsed" for r in sub) / n, 4),
                        "ack_visible": round(sum(r["acknowledged"] for r in sub) / n, 4),
                        "reopened_think": round(sum(r["reopened_think"] for r in sub) / n, 4)}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE r1", flush=True)

if __name__ == "__main__":
    main()
