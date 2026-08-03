"""GSM8K falsifiability gradient sweep (early/mid/late × k=1..3).

Reuses the existing gold runs if present; otherwise regenerates up to
N_GOLD_ATTEMPTS problems and keeps those with valid CoTs. For each solved
instance, we corrupt the numeric result of a selected calculation step and let
the model continue. The parameter k denotes how many calculation steps remain
*after* the corrupted step; we only keep combinations where at least k-1 steps
follow (so k=1 is the standard negstep, k>1 pushes the corruption earlier).

Outputs:
 - results/gsm8k_gold.jsonl (cached gold)
 - results/gsm8k_gradient/runs.jsonl (all perturbations)
 - results/gsm8k_gradient/summary.json (metrics grouped by point,k)
"""
import os
import json
import re
import random
import time
from collections import defaultdict

import torch
from datasets import load_dataset

from common import load_model, RESULTS

OUT_DIR = os.path.join(RESULTS, "gsm8k_gradient")
GOLD_PATH = os.path.join(RESULTS, "gsm8k_gold.jsonl")
N_GOLD_ATTEMPTS = 400
MAX_SOLVED = 150

INSTR = ("Solve the math word problem step by step, writing each computation as "
         "'a op b = c'. End with 'The answer is N.'\n\n")
FEWSHOT = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes she earned 0.2 * 50 = 10 dollars. The answer is 10.

"""
CALC_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/])\s*(-?[\d,]+(?:\.\d+)?)\s*=\s*(-?[\d,]+(?:\.\d+)?)")
FINAL_RE = re.compile(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", re.I)

POINT_FUNCS = {
    "early": lambda calcs: 0,
    "mid": lambda calcs: max(0, len(calcs) // 2 - 1),
    "late": lambda calcs: max(0, len(calcs) - 3),
}


def parse_answer(ans_text: str) -> float:
    return float(ans_text.replace(",", ""))


def prompt_ids(tok, question: str, prefix: str | None = None):
    user = INSTR + FEWSHOT + f"Q: {question}\nA:"
    ids = tok.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True, return_tensors="pt"
    )["input_ids"]
    if prefix:
        pre = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([ids, pre], dim=1)
    return ids


@torch.no_grad()
def greedy(tok, model, ids, max_new=320):
    out = model.generate(ids.to("cuda:0"), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def ensure_gold(tok, model):
    if os.path.exists(GOLD_PATH):
        gold = [json.loads(l) for l in open(GOLD_PATH)]
        print(f"Loaded cached gold ({len(gold)} instances).", flush=True)
        return gold
    ds = load_dataset("openai/gsm8k", "main", split="test")
    gold, solved = [], 0
    rng = random.Random(0)
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i >= N_GOLD_ATTEMPTS or solved >= MAX_SOLVED:
            break
        target = parse_answer(ex["answer"].split("####")[-1].strip())
        txt = greedy(tok, model, prompt_ids(tok, ex["question"]))
        m = FINAL_RE.search(txt)
        calcs = CALC_RE.findall(txt)
        ok = bool(m) and abs(parse_answer(m.group(1)) - target) < 1e-6 and len(calcs) >= 4
        gold.append({
            "id": f"gsm_{i}",
            "question": ex["question"],
            "answer": target,
            "gen": txt,
            "solved": ok,
            "n_calcs": len(calcs)
        })
        solved += int(ok)
        if (i + 1) % 50 == 0:
            print(f"[{i+1}] solved={solved} ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(GOLD_PATH, "w") as fh:
        for g in gold:
            fh.write(json.dumps(g) + "\n")
    print(f"Generated gold: solved {solved}/{len(gold)}", flush=True)
    return gold


def select_calc_indices(calcs, point):
    if not calcs:
        return None
    idx = POINT_FUNCS[point](calcs)
    idx = min(max(idx, 0), len(calcs) - 2)  # leave at least two steps after for late cases
    return idx


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok, model = load_model()
    gold = ensure_gold(tok, model)
    solved = [g for g in gold if g["solved"]]
    print(f"Gradient sweep using {len(solved)} solved GSM8K instances.", flush=True)

    rng = random.Random(123)
    runs = []
    for g in solved:
        calcs = list(CALC_RE.finditer(g["gen"]))
        if len(calcs) < 4:
            continue
        for point in ("early", "mid", "late"):
            si = select_calc_indices(calcs, point)
            if si is None:
                continue
            remaining = len(calcs) - si - 1
            for k in (1, 2, 3):
                if remaining < k - 1:
                    continue
                calc = calcs[si]
                true_val = parse_answer(calc.group(4))
                delta = rng.choice([2, 3, 5, -2, -3])
                wrong_val = true_val + delta
                wrong_str = str(int(wrong_val)) if wrong_val == int(wrong_val) else f"{wrong_val:.2f}"
                prefix = " " + g["gen"][: calc.start(4)] + wrong_str + "."
                cont = greedy(tok, model, prompt_ids(tok, g["question"], prefix=prefix), max_new=256)
                m = FINAL_RE.search(cont)
                final = parse_answer(m.group(1)) if m else None
                calc_uses = CALC_RE.findall(cont)
                reused = any(
                    abs(parse_answer(a) - wrong_val) < 1e-6 or
                    abs(parse_answer(b) - wrong_val) < 1e-6
                    for a, _, b, _ in calc_uses
                ) or wrong_str in cont
                runs.append({
                    "id": g["id"],
                    "point": point,
                    "k": k,
                    "step_index": si,
                    "total_steps": len(calcs),
                    "remaining": remaining,
                    "true_val": true_val,
                    "wrong_val": wrong_val,
                    "final": final,
                    "recovered": bool(final is not None and abs(final - g["answer"]) < 1e-6),
                    "poisoned": reused,
                    "continuation": cont
                })
    runs_path = os.path.join(OUT_DIR, "runs.jsonl")
    with open(runs_path, "w") as fh:
        for r in runs:
            fh.write(json.dumps(r) + "\n")
    summary = defaultdict(lambda: defaultdict(int))
    for r in runs:
        key = (r["point"], r["k"])
        summary[key]["n"] += 1
        summary[key]["recovered"] += int(r["recovered"])
        summary[key]["poisoned"] += int(r["poisoned"])
    summary_out = {}
    for (point, k), stats in summary.items():
        n = stats["n"]
        summary_out.setdefault(point, {})[f"k={k}"] = {
            "n": n,
            "recovered_rate": round(stats["recovered"] / n, 4),
            "poisoned_rate": round(stats["poisoned"] / n, 4)
        }
    summary_out["_meta"] = {
        "gold_solved": len(solved),
        "gold_attempted": len(gold)
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary_out, fh, indent=2)
    print(json.dumps(summary_out, indent=2), flush=True)
    print(f"DONE gsm8k gradient: {len(runs)} perturbations", flush=True)


if __name__ == "__main__":
    main()
