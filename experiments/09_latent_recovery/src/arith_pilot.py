"""Brittle-regime pilot: chained arithmetic with the same injection protocol.

Task: "Start with 7. Add 12. Multiply by 3. ..." Gold CoT = running values.
There is NO redundancy: a corrupted intermediate value admits no alternative
derivation path — recovery requires ignoring the stated value and recomputing.

Stages (single script, sequential): gold collection -> injection (corrupt one
intermediate running value, plausible magnitude) -> exact validation.
Behavioral only (no hidden-state caching) for the pilot.

Outputs: results/arith/{gold.jsonl, perturbed.jsonl, summary.json}
"""
import os, json, re, random, time
import torch
from common import load_model, RESULTS

N_PROBLEMS = 100
K_OPS = 6
OUT = os.path.join(RESULTS, "arith")

FEWSHOT = """Q: Start with 4. Add 9. Multiply by 2. Subtract 6. Add 11. What is the result?
A: 4 + 9 = 13. 13 * 2 = 26. 26 - 6 = 20. 20 + 11 = 31. The result is 31.

Q: Start with 12. Subtract 5. Multiply by 3. Add 8. Subtract 14. What is the result?
A: 12 - 5 = 7. 7 * 3 = 21. 21 + 8 = 29. 29 - 14 = 15. The result is 15.

"""
INSTR = ("Solve the chained arithmetic problem step by step, exactly in the style of "
         "the examples. End with 'The result is N.'\n\n")

def gen_problems(rng):
    probs = []
    for i in range(N_PROBLEMS):
        start = rng.randint(2, 20)
        ops, vals = [], [start]
        v = start
        for _ in range(K_OPS):
            op = rng.choice(["add", "sub", "mul"])
            if op == "mul" and abs(v) < 200:
                k = rng.randint(2, 4); v = v * k; ops.append(f"Multiply by {k}.")
            elif op == "sub":
                k = rng.randint(2, 19); v = v - k; ops.append(f"Subtract {k}.")
            else:
                k = rng.randint(2, 19); v = v + k; ops.append(f"Add {k}.")
            vals.append(v)
        text = f"Start with {start}. " + " ".join(ops) + " What is the result?"
        probs.append({"id": f"arith_{i}", "question": text, "values": vals,
                      "answer": vals[-1]})
    return probs

def prompt_ids(tok, q, prefix=None):
    user = INSTR + FEWSHOT + f"Q: {q}\nA:"
    ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                  add_generation_prompt=True, return_tensors="pt")["input_ids"]
    if prefix:
        pre = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([ids, pre], dim=1)
    return ids

@torch.no_grad()
def greedy(tok, model, ids, max_new=128):
    out = model.generate(ids.to("cuda:0"), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

STEP_RE = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=\s*(-?\d+)")
FINAL_RE = re.compile(r"result is\s*(-?\d+)", re.I)

def parse_steps(text):
    return [(int(a), op, int(b), int(c)) for a, op, b, c in STEP_RE.findall(text)]

def step_ok(a, op, b, c):
    return c == (a + b if op == "+" else a - b if op == "-" else a * b)

def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(0)
    probs = gen_problems(rng)
    tok, model = load_model()

    # ---- gold ----
    gold = []
    for p in probs:
        txt = greedy(tok, model, prompt_ids(tok, p["question"]))
        m = FINAL_RE.search(txt)
        solved = bool(m) and int(m.group(1)) == p["answer"]
        steps = parse_steps(txt)
        chain_ok = all(step_ok(*s) for s in steps) and len(steps) == K_OPS
        gold.append({**p, "gen": txt, "solved": solved and chain_ok})
    with open(os.path.join(OUT, "gold.jsonl"), "w") as fh:
        for g in gold:
            fh.write(json.dumps(g) + "\n")
    solved = [g for g in gold if g["solved"]]
    print(f"gold: solved {len(solved)}/{len(gold)}", flush=True)

    # ---- inject: corrupt the result of step idx (early/mid/late) ----
    out_rows = []
    for g in solved:
        steps = parse_steps(g["gen"])
        sent_ends = [m.end() for m in re.finditer(r"\.\s", g["gen"])]
        for name, idx in [("early", 1), ("mid", K_OPS // 2), ("late", K_OPS - 2)]:
            a, op, b, c = steps[idx]
            wrong = c + rng.choice([-3, -2, 2, 3, 10, -10])
            # prefix = gold text up to and including step idx, with corrupted value
            pat = f"{a} {op} {b} = {c}."
            pos = g["gen"].find(pat)
            if pos < 0:
                continue
            prefix = " " + g["gen"][:pos] + f"{a} {op} {b} = {wrong}."
            cont = greedy(tok, model, prompt_ids(tok, g["question"], prefix=prefix))
            m = FINAL_RE.search(cont)
            final = int(m.group(1)) if m else None
            cont_steps = parse_steps(cont)
            # poisoned: next step uses the corrupted value as its left operand
            poisoned = bool(cont_steps) and cont_steps[0][0] == wrong
            # recovered: correct final answer (requires ignoring the wrong value)
            out_rows.append({"id": g["id"], "point": name, "true_val": c,
                             "wrong_val": wrong, "final": final,
                             "recovered": final == g["answer"],
                             "poisoned_next_step": poisoned,
                             "continuation": cont})
    with open(os.path.join(OUT, "perturbed.jsonl"), "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    summary = {}
    for name in ["early", "mid", "late"]:
        sub = [r for r in out_rows if r["point"] == name]
        if not sub:
            continue
        summary[name] = {
            "n": len(sub),
            "recovered_rate": round(sum(r["recovered"] for r in sub) / len(sub), 4),
            "poisoned_next_step_rate": round(sum(r["poisoned_next_step"] for r in sub) / len(sub), 4),
        }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE arith", flush=True)

if __name__ == "__main__":
    main()
