"""Trace-level real-detection test: for each adapter, the SAME 100 programs per depth (eval seeds 0..99)
are run twice, once with the planted wrong line 10 (perturbed) and once with the true line 10 (control).
Per-item re-check flags are saved. A model that detects errors re-checks the perturbed trace and not the
control; a blanket re-checker flags both. Output: per-item JSON + summary."""
import argparse, json, re
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from common import gen_program, build_example, user_message, plant, grade, true_values, SITE

ap = argparse.ArgumentParser()
ap.add_argument("--adapters", required=True)   # comma list of run dirs ('base' = untuned)
ap.add_argument("--outdir", required=True)
ap.add_argument("--depths", default="1,2,3,4,5")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
llm = LLM("Qwen/Qwen2.5-7B-Instruct", enable_lora=True, max_lora_rank=16, max_model_len=3072,
          gpu_memory_utilization=0.85, dtype="bfloat16")
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=500)
depths = [int(x) for x in a.depths.split(",")]
progs = {(d, i): gen_program(d, i) for d in depths for i in range(a.n)}
for k, ad in enumerate(a.adapters.split(",")):
    name = ad.rstrip("/").split("/")[-1]
    lr = None if ad == "base" else LoRARequest(name, k + 1, ad)
    prompts, meta = [], []
    for (d, i), p in progs.items():
        for kind in ("perturbed", "control"):
            pre, _ = build_example(p, kind, i)
            prompts.append(tok.apply_chat_template([{"role": "user", "content": user_message(p)}],
                           add_generation_prompt=True, tokenize=False) + pre)
            meta.append((d, i, kind, p, plant(p, i) if kind == "perturbed" else None))
    outs = llm.generate(prompts, sp, lora_request=lr)
    items = []
    for (d, i, kind, p, pv), o in zip(meta, outs):
        txt = o.outputs[0].text
        g = grade(p, pv, txt, kind)
        tv = true_values(p)[SITE]
        first = txt.strip().splitlines()[0] if txt.strip() else ""
        # does the first re-check line name the true value (and, for perturbed, the planted one)?
        names_true = bool(re.search(r"= %d\b" % tv, first)) and "re-check" in first.lower()
        items.append({"depth": d, "i": i, "kind": kind, "recheck": g["recheck"], "label": g["label"],
                      "rest_ok": g["rest_ok"], "names_true": names_true, "first": first[:160]})
    json.dump({"adapter": ad, "items": items}, open("%s/%s.json" % (a.outdir, name), "w"))
    print("DONE", name, flush=True)
