"""LLM-judge doubt classifier: Qwen2.5-32B judges whether a continuation expresses
doubt/hesitation about earlier reasoning. Validated against lexical labels; corrected
per-family doubt rates reported.

Sample: all runs from {wrong, paraphrase, negstep, falsehood, contradiction} at mid.
Output: results/doubt_judge.json
"""
import os, json, re, time
import torch
from common import RESULTS

JUDGE = "Qwen/Qwen2.5-32B-Instruct"
FAMS = [("wrong", "perturbed"), ("paraphrase", "perturbed_paraphrase"),
        ("negstep", "perturbed_negstep"), ("falsehood", "perturbed_falsehood"),
        ("contradiction", "perturbed_contradiction")]
DOUBT = re.compile(r"\b(wait|however|but|actually|mistake|error|contradict|incorrect|cannot|inconsistent)\b", re.I)

PROMPT = """You will see the continuation of a step-by-step logic proof. Decide whether the text EXPRESSES DOUBT, HESITATION, SELF-CORRECTION, or flags an inconsistency about any earlier step (e.g., questioning a premise, noting a contradiction, revising a claim). Mechanical proof steps with no commentary = NO.

Continuation:
---
{cont}
---
Answer with exactly one word: YES or NO."""

@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(JUDGE)
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    cohort = {json.loads(l)["id"] for l in open(os.path.join(RESULTS, "validated.jsonl"))}
    out = {}
    t0 = time.time()
    for fam, d in FAMS:
        rpath = os.path.join(RESULTS, d, "runs.jsonl")
        if not os.path.exists(rpath):
            continue
        runs = [json.loads(l) for l in open(rpath) if "skip" not in json.loads(l)]
        runs = [r for r in runs if r["point"] == "mid" and r["id"] in cohort]
        agree = j_yes = l_yes = both = 0
        rows = []
        for r in runs:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(cont=r["continuation"][:1500])}],
                add_generation_prompt=True, return_tensors="pt")["input_ids"].to("cuda:0")
            o = model.generate(ids, max_new_tokens=4, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            ans = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip().upper()
            judge = ans.startswith("YES")
            lex = bool(DOUBT.search(r["continuation"]))
            agree += int(judge == lex)
            j_yes += int(judge); l_yes += int(lex); both += int(judge and lex)
            rows.append({"id": r["id"], "judge": judge, "lex": lex})
        n = len(runs)
        out[fam] = {"n": n, "judge_doubt_rate": round(j_yes / n, 4),
                    "lexical_doubt_rate": round(l_yes / n, 4),
                    "agreement": round(agree / n, 4),
                    "judge_and_lex": both}
        print(f"{fam}: {out[fam]} ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(os.path.join(RESULTS, "doubt_judge.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("DONE judge", flush=True)

if __name__ == "__main__":
    main()
