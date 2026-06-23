"""GSM8K perturbation pilot: first realistic-task regime cell.

Gold: greedy CoT on GSM8K test items, numeric validation. Injection: corrupt the
middle intermediate computation result in the model's own CoT (plausible delta),
continue with no feedback. Recovered = correct final answer. Poisoned = corrupted
value reused downstream.

Output: results/gsm8k/{gold.jsonl,runs.jsonl,summary.json}
"""
import os, json, re, random, time
import torch
from common import load_model, RESULTS

OUT = os.path.join(RESULTS, "gsm8k")
N_GOLD_ATTEMPTS = 250
MAX_SOLVED = 120

FEWSHOT = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes she earned 0.2 * 50 = 10 dollars. The answer is 10.

"""
INSTR = ("Solve the math word problem step by step, writing each computation as "
         "'a op b = c'. End with 'The answer is N.'\n\n")

CALC_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/])\s*(-?[\d,]+(?:\.\d+)?)\s*=\s*(-?[\d,]+(?:\.\d+)?)")
FINAL_RE = re.compile(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", re.I)

def num(s):
    return float(s.replace(",", ""))

def ids_for(tok, q, prefix=None):
    user = INSTR + FEWSHOT + f"Q: {q}\nA:"
    ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                  add_generation_prompt=True, return_tensors="pt")["input_ids"]
    if prefix:
        pre = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([ids, pre], dim=1)
    return ids

@torch.no_grad()
def greedy(tok, model, ids, max_new=320):
    out = model.generate(ids.to("cuda:0"), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def main():
    os.makedirs(OUT, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(0)
    tok, model = load_model()
    t0 = time.time()

    gold, n_solved = [], 0
    for i, ex in enumerate(ds):
        if i >= N_GOLD_ATTEMPTS or n_solved >= MAX_SOLVED:
            break
        true_ans = num(ex["answer"].split("####")[-1].strip())
        txt = greedy(tok, model, ids_for(tok, ex["question"]))
        m = FINAL_RE.search(txt)
        calcs = CALC_RE.findall(txt)
        ok = bool(m) and abs(num(m.group(1)) - true_ans) < 1e-6 and len(calcs) >= 3
        gold.append({"id": f"gsm_{i}", "question": ex["question"], "answer": true_ans,
                     "gen": txt, "solved": ok, "n_calcs": len(calcs)})
        n_solved += int(ok)
        if (i + 1) % 50 == 0:
            print(f"[{i+1}] solved={n_solved} ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(os.path.join(OUT, "gold.jsonl"), "w") as fh:
        for g in gold:
            fh.write(json.dumps(g) + "\n")
    solved = [g for g in gold if g["solved"]]
    print(f"gsm8k gold: solved {len(solved)}/{len(gold)}", flush=True)

    rows = []
    for g in solved:
        calcs = list(CALC_RE.finditer(g["gen"]))
        if len(calcs) < 3:
            continue
        mid = calcs[len(calcs) // 2 - 1] if len(calcs) > 3 else calcs[1]
        c_val = num(mid.group(4))
        delta = rng.choice([2, 3, 5, 10, -2, -3])
        wrong = c_val + delta
        wrong_s = str(int(wrong)) if wrong == int(wrong) else f"{wrong:.2f}"
        prefix = " " + g["gen"][: mid.start(4)] + wrong_s + "."
        cont = greedy(tok, model, ids_for(tok, g["question"], prefix=prefix), max_new=256)
        m = FINAL_RE.search(cont)
        final = num(m.group(1)) if m else None
        reused = wrong_s in cont or f"{wrong_s} " in cont
        ops = CALC_RE.findall(cont)
        op_uses = any(abs(num(a) - wrong) < 1e-6 or abs(num(b) - wrong) < 1e-6
                      for a, _, b, _ in ops)
        rows.append({"id": g["id"], "true_val": c_val, "wrong_val": wrong,
                     "final": final, "recovered": final == g["answer"],
                     "poisoned": bool(op_uses or reused), "continuation": cont})
    with open(os.path.join(OUT, "runs.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    summary = {"n": n,
               "recovered_rate": round(sum(r["recovered"] for r in rows) / n, 4),
               "poisoned_rate": round(sum(r["poisoned"] for r in rows) / n, 4),
               "_gold": {"solved": len(solved), "attempted": len(gold)}}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE gsm8k", flush=True)

if __name__ == "__main__":
    main()
