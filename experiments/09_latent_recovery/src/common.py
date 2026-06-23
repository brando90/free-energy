"""Shared model/prompt/parsing utilities for the latent-recovery pilot."""
import os, json, re
import numpy as np
import torch

MODEL = os.environ.get("LR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_N_LAYERS = {"Qwen/Qwen2.5-1.5B-Instruct": 28, "Qwen/Qwen2.5-7B-Instruct": 28,
             "Qwen/Qwen2.5-32B-Instruct": 64,
             "allenai/OLMo-2-1124-7B-Instruct": 32,
             "mistralai/Mistral-7B-Instruct-v0.2": 32,
             "NousResearch/Meta-Llama-3.1-8B-Instruct": 32}.get(MODEL, 28)
LAYERS = [0] + [round(_N_LAYERS * f) for f in (0.25, 0.5, 0.75, 1.0)]  # emb..final
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, os.environ.get("LR_RESULTS", "results"))

FEWSHOT = """Q: Every yumpus is a dumpus. Dumpuses are tumpuses. Tumpuses are not bright. Sam is a yumpus. Prove: Sam is not bright.
A: Sam is a yumpus. Every yumpus is a dumpus. Sam is a dumpus. Dumpuses are tumpuses. Sam is a tumpus. Tumpuses are not bright. Sam is not bright.

Q: Each gorpus is a sterpus. Sterpuses are red. Every borpus is a gorpus. Alex is a borpus. Prove: Alex is red.
A: Alex is a borpus. Every borpus is a gorpus. Alex is a gorpus. Each gorpus is a sterpus. Alex is a sterpus. Sterpuses are red. Alex is red.

"""

INSTR = ("You will be given facts and rules about fictional creatures, then asked to prove a statement. "
         "Answer with only the proof: a sequence of statements, one deduction at a time, in the exact "
         "style of the examples. End with the statement to be proven.\n\n")
if os.environ.get("LR_VERIFY_PROMPT") == "1":
    INSTR = INSTR.rstrip() + (" Before using any stated fact, verify that it follows from "
             "the given rules and premises; if a statement conflicts with them, point out the "
             "conflict and continue from the correct facts.\n\n")

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    return tok, model

def make_prompt_ids(tok, question, target, answer_prefix=None):
    """Chat-formatted prompt; optionally pre-fill the assistant turn with answer_prefix."""
    user = INSTR + FEWSHOT + f"Q: {question} Prove: {target}\nA:"
    msgs = [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")["input_ids"]
    if answer_prefix:
        pre = tok(answer_prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([ids, pre], dim=1)
    return ids

def split_sentences(text):
    text = text.strip()
    parts = [p.strip() for p in re.split(r"(?<=\.)\s+", text) if p.strip()]
    return parts

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def solved(gen_text, target):
    sents = split_sentences(gen_text)
    return len(sents) > 0 and norm(sents[-1]) == norm(target)

@torch.no_grad()
def greedy(tok, model, ids, max_new=256):
    out = model.generate(ids.to("cuda:0"), max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), out[0]

@torch.no_grad()
def sample_n(tok, model, ids, n=8, temp=0.8, max_new=256):
    out = model.generate(ids.to("cuda:0"), max_new_tokens=max_new, do_sample=True,
                         temperature=temp, num_return_sequences=n,
                         pad_token_id=tok.eos_token_id)
    texts = [tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]
    return texts, out

@torch.no_grad()
def hidden_states_from(tok, model, full_ids, from_pos):
    """One forward pass; return {layer: [T_after, d] fp16} for positions >= from_pos."""
    out = model(input_ids=full_ids.unsqueeze(0).to("cuda:0"), output_hidden_states=True)
    hs = {}
    for L in LAYERS:
        hs[str(L)] = out.hidden_states[L][0, from_pos:, :].to(torch.float16).cpu().numpy()
    return hs

def save_npz(path, hs, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{f"layer_{k}": v for k, v in hs.items()},
                        meta=json.dumps(meta))

def make_distractor(question, gold_cot, idx=0):
    """Irrelevant-but-plausible rule sentence: off-path category + question adjective,
    guaranteed not to be an actual rule of this question."""
    cats = {(_c[:-2] if _c.endswith("es") else _c)
            for _c in re.findall(r"\b([a-z]+pus(?:es)?)\b", question)}
    on_path = set(re.findall(r"\b([a-z]+pus)\b", " ".join(gold_cot).lower()))
    off = sorted(cats - on_path) or sorted(cats)
    adjs = sorted(set(re.findall(r"\bis (?:not )?([a-z]+)\.", question)) - cats -
                  {"a", "an"})
    if not off or not adjs:
        return None
    for shift in range(len(off) * len(adjs)):
        c = off[(idx + shift) % len(off)]
        a = adjs[(idx + shift) % len(adjs)]
        cand = f"Every {c} is {a}."
        if cand.lower() not in question.lower():
            return cand
    return None

def make_paraphrase(steps, si, idx=0):
    """Benign control: same semantic content, different surface form."""
    markers = ["Therefore, ", "It follows that ", "Thus, "]
    s = steps[si].rstrip(".")
    return f"{markers[idx % len(markers)]}{s[0].lower() + s[1:] if s.split()[0] in ('Every','Each','All') else s}."

def make_falsehood(question, gold_cot, correct_step, idx=0):
    """Like wrong_category but GUARANTEED false: candidate categories filtered to
    those NOT entailed for the entity by the question's premises under rule closure."""
    from validator import parse_world, parse_fact, closure, derivable
    rules, facts, _ = parse_world(question)
    reach = closure(rules)
    m = re.match(r"^(\S+) is ", correct_step)
    if not m:
        return None
    entity = m.group(1)
    premises = {p for e, p in facts if e == entity}
    cats = {(_c[:-2] if _c.endswith("es") else _c)
            for _c in re.findall(r"\b([a-z]+pus(?:es)?)\b", question)}
    false_cats = sorted(c for c in cats
                        if not derivable(("cat", c), premises, reach))
    if not false_cats:
        return None
    repl = false_cats[idx % len(false_cats)]
    cand = f"{entity} is {'an ' if repl[0] in 'aeiou' else 'a '}{repl}."
    f = parse_fact(cand, entity)
    assert f and not derivable(f[1], premises, reach)
    return cand

def make_neghop(steps, si, k):
    """Dose-response: negate the entity fact the model will derive k hops from the
    injection point (k=1 == negstep). Falsifying evidence is k inference hops away."""
    ent = steps[0].split()[0]
    ents = [i for i, s in enumerate(steps) if s.split()[0] == ent]
    if si not in ents:
        return None
    j = ents.index(si)
    if j + (k - 1) >= len(ents) - 1:   # don't target the final conclusion
        return None
    return make_negstep(steps, ents[j + (k - 1)])

def make_negstep(steps, si):
    """Guaranteed-false, selection-free: negate the correct step itself. The gold
    step is entailed-true, so its negation is false; nothing in the prefix asserts
    it yet, so the falsehood is consistent with stated context."""
    s = steps[si]
    m = re.match(r"^(\S+) is (a|an) (\w+)\.$", s)
    if m:
        return f"{m.group(1)} is not {m.group(2)} {m.group(3)}."
    m = re.match(r"^(\S+) is (not )?(\w+)\.$", s)
    if m:
        return (f"{m.group(1)} is {m.group(3)}." if m.group(2)
                else f"{m.group(1)} is not {m.group(3)}.")
    return None

def make_contradiction(steps, prefix_end):
    """Negate the most recently established entity fact in the prefix."""
    for s in reversed(steps[:prefix_end]):
        m = re.match(r"^(\S+) is (a|an) (\w+)\.$", s)
        if m:
            return f"{m.group(1)} is not {m.group(2)} {m.group(3)}."
        m = re.match(r"^(\S+) is (not )?(\w+)\.$", s)
        if m:
            return (f"{m.group(1)} is {m.group(3)}." if m.group(2)
                    else f"{m.group(1)} is not {m.group(3)}.")
    return None

def wrong_category(question, gold_cot, correct_step, idx=0):
    """Pick a category word from the question that is plausible but off the gold path."""
    cats = set(re.findall(r"\b([a-z]+pus(?:es)?)\b", question))
    cats = {c[:-2] if c.endswith("es") else c for c in cats}
    gold_text = " ".join(gold_cot).lower()
    # prefer categories never derived for the entity in the gold chain
    on_path = set(re.findall(r"\b([a-z]+pus)\b", gold_text))
    off = sorted(cats - on_path) or sorted(cats)
    m = re.match(r"^(\S+) is (an? |not )?(.+?)\.?$", correct_step)
    if not m or not off:
        return None
    entity, art = m.group(1), m.group(2) or "a "
    repl = off[idx % len(off)]
    if norm(f"{entity} is {art}{repl}") == norm(correct_step):
        return None
    return f"{entity} is {'an ' if repl[0] in 'aeiou' else 'a '}{repl}."
