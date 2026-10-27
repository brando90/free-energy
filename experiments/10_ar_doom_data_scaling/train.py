"""LoRA SFT (plain PyTorch loop) on planted-error data. Hyperparameters = paper Table 4:
LoRA r16/a32/drop0.05 on q,k,v,o,gate,up,down; AdamW lr 1e-4 (5 warmup updates, cosine to 0),
2 x 8 accumulation = 16 examples/update, 3 epochs, grad clip 1.0, bf16, grad checkpointing,
max len 1536. Loss only on target tokens (+ end-of-turn)."""
import argparse, json, math, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=3)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open(a.data)]
MAXLEN = 1536
ex = []
for r in rows:
    p = tok.apply_chat_template([{"role": "user", "content": r["user"]}], add_generation_prompt=True, tokenize=False) + r["prefix"]
    pi = tok(p, add_special_tokens=False)["input_ids"]
    ti = tok(r["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (pi + ti)[:MAXLEN]
    assert len(pi) + len(ti) <= MAXLEN, "truncation"
    ex.append((ids, [-100] * len(pi) + ti))
print("examples", len(ex), "mean len", sum(len(e[0]) for e in ex) / len(ex),
      "mean target", sum(sum(1 for t in e[1] if t != -100) for e in ex) / len(ex), flush=True)

model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0})
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
MB, ACC = 2, 8
n_upd = math.ceil(len(ex) * a.epochs / (MB * ACC))
def lr_at(u):
    if u < 5: return 1e-4 * (u + 1) / 5
    return 1e-4 * 0.5 * (1 + math.cos(math.pi * (u - 5) / max(1, n_upd - 5)))
order = []
for e in range(a.epochs):
    idx = list(range(len(ex))); random.shuffle(idx); order += idx
model.train(); t0 = time.time(); u = 0; pos = 0; losses = []
while pos < len(order):
    batch = order[pos:pos + MB * ACC]; pos += MB * ACC
    ntok = sum(sum(1 for t in ex[i][1] if t != -100) for i in batch)
    tot = 0.0
    for k in range(0, len(batch), MB):
        mb = [ex[i] for i in batch[k:k + MB]]
        m = max(len(x[0]) for x in mb)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (m - len(x[0])) for x in mb]).cuda()
        lab = torch.tensor([x[1] + [-100] * (m - len(x[1])) for x in mb]).cuda()
        att = torch.tensor([[1] * len(x[0]) + [0] * (m - len(x[0])) for x in mb]).cuda()
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                                 ignore_index=-100, reduction="sum") / ntok
        loss.backward(); tot += loss.item()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    for g in opt.param_groups: g["lr"] = lr_at(u)
    opt.step(); opt.zero_grad(set_to_none=True); u += 1; losses.append(tot)
    if u % 10 == 0 or u == n_upd:
        print("upd %d/%d loss %.4f %.0fs" % (u, n_upd, tot, time.time() - t0), flush=True)
model.save_pretrained(a.out)
json.dump({"n_examples": len(ex), "updates": n_upd, "losses": losses, "seconds": time.time() - t0,
           "model": a.model, "seed": a.seed}, open(a.out + "/train_log.json", "w"))
print("saved", a.out)
