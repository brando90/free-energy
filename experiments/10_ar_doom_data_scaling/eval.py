"""Evaluate base or LoRA adapter with vLLM (greedy) on held-out planted-error programs.
Perturbed: depths --depths, N programs each; control (true line 10): depths 1,3,5.
Output: JSON with per-depth propagation / correction / other rates and control false-flag rate."""
import argparse, json
from collections import Counter
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from common import gen_program, build_example, user_message, plant, grade

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--adapter", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--depths", default="1,2,3,4,5,6,8")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
llm = LLM(a.model, enable_lora=bool(a.adapter), max_lora_rank=16, max_model_len=3072,
          gpu_memory_utilization=0.85, dtype="bfloat16")
tok = llm.get_tokenizer()
lr = LoRARequest("ad", 1, a.adapter) if a.adapter else None
sp = SamplingParams(temperature=0.0, max_tokens=500)
jobs = []
for d in [int(x) for x in a.depths.split(",")]:
    for i in range(a.n):
        jobs.append((d, i, "perturbed"))
for d in (1, 3, 5):
    for i in range(a.n):
        jobs.append((d, i, "control"))
prompts, meta = [], []
for d, i, kind in jobs:
    seed = i  # eval seeds 0..n-1 (train seeds start at 1,000,000)
    p = gen_program(d, seed)
    pre, _ = build_example(p, kind, seed)
    prompt = tok.apply_chat_template([{"role": "user", "content": user_message(p)}], add_generation_prompt=True, tokenize=False) + pre
    prompts.append(prompt); meta.append((d, kind, p, plant(p, seed) if kind == "perturbed" else None))
outs = llm.generate(prompts, sp, lora_request=lr)
res = {}; samples = []
for (d, kind, p, pv), o in zip(meta, outs):
    txt = o.outputs[0].text
    g = grade(p, pv, txt, kind)
    res.setdefault((kind, d), Counter())[g["label"]] += 1
    res[(kind, d)]["rest_ok"] += int(g["rest_ok"])
    if len(samples) < 6 and kind == "perturbed": samples.append({"depth": d, "text": txt[:400], "g": g})
summary = {}
for (kind, d), c in sorted(res.items()):
    n = sum(v for k, v in c.items() if k != "rest_ok")
    summary["%s_d%d" % (kind, d)] = {"n": n, **{k: v for k, v in c.items()},
        **{"rate_" + k: v / n for k, v in c.items() if k != "rest_ok"}}
json.dump({"adapter": a.adapter, "model": a.model, "summary": summary, "samples": samples}, open(a.out, "w"), indent=1)
for k, v in summary.items():
    print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("rate_")})
