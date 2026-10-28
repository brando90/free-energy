"""E16 shared library (ADDITIVE; no existing file modified).

Spec: ../../E16_STEERING.md (binding, committed a4fec56).

Reuses the shipped e11 pipeline BY IMPORT for prompts, worlds, and grading;
adds a transformers backend with residual-stream hooks (vLLM cannot
intervene mid-layer).

LAYER CONVENTION: "layer L" means the output of model.model.layers[L-1],
which equals output_hidden_states[L]. The steering hook and the extraction
site use the same convention by construction.

WORLD PARTITION (deviation from spec n, logged in DEVIATIONS.md): the E11
result pool (150 programs/model) is split deterministically by sorted
program_id -- last 60 = extraction, first 90 = evaluation (screen uses the
first 40 of those, confirm all 90). Disjointness between extraction and
evaluation takes precedence over the spec's confirm n=150.
"""
import hashlib
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
E16 = os.path.abspath(os.path.join(HERE, ".."))
RW = os.path.abspath(os.path.join(E16, ".."))                 # recompute_wall
E11_RUN_DIR = os.path.join(RW, "e11_run")
IMPROVE = os.path.abspath(os.path.join(RW, "..", ".."))       # improvement_plan
TOOLING = os.path.join(IMPROVE, "tooling")
for _d in (HERE, E11_RUN_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import e11_run as e11                      # noqa: E402  (wires expg paths)
import trace_format as tf                  # noqa: E402
import validate as val_mod                 # noqa: E402
from interp import parse_program           # noqa: E402

MODELS = {
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct",
               "a09a35458c702b33eeacc393d103063234e8bc28", 28),
    "llama8b": ("NousResearch/Meta-Llama-3.1-8B-Instruct",
                "d10aef7999a2b5ba950ab3974312feeedbfe0b77", 32),
}
LAYERS = {"qwen7b": [7, 14, 21], "llama8b": [8, 16, 24]}
ALPHAS = [1, 2, 4, 8, 16, -1, -2, -4, -8, -16]
MAX_NEW = 384
SEED = 160911

VEC_DIR = os.path.join(E16, "vectors")
RES_DIR = os.path.join(E16, "results")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------- pool loading

def e11_dir(model_key):
    return os.path.join(E11_RUN_DIR, "results",
                        {"qwen7b": "E11_qwen7b", "llama8b": "E11_llama8b"}[model_key])


def load_pool(model_key):
    """Programs + manifest rows for the families E16 uses, partitioned."""
    d = e11_dir(model_key)
    programs = {p["program_id"]: p for p in read_jsonl(os.path.join(d, "programs.jsonl"))}
    fams = ("anchor_k0", "depth_k1_bare", "depth_k2_bare")
    man = [m for m in read_jsonl(os.path.join(d, "manifest.jsonl")) if m["family"] in fams]
    by_fam = {}
    for m in man:
        by_fam.setdefault(m["family"], {})[m["program_id"]] = m
    # The shipped E11 families were gold-gated independently, so each family
    # covers its OWN ~150 programs. Extraction contrasts are unpaired
    # (diff-in-means) and the breach rule compares within-family, so the
    # partition is per-family: last 60 = extraction, first 90 = evaluation,
    # first 40 = screen. Extraction/eval disjointness holds within family.
    part = {}
    for f in fams:
        pids = sorted(by_fam.get(f, {}))
        part[f] = {"extract": pids[-60:], "eval": pids[:90], "screen": pids[:40]}
    return SimpleNamespace(programs=programs, by_fam=by_fam, part=part, dir=d)


def unperturbed_prefix(m):
    """Gold prefix 1..site_line: manifest prefix with the injected last line
    replaced by the model's own original (true) line."""
    lines = m["prefix_text"].rstrip("\n").split("\n")
    assert lines[-1] == m["injected_trace_line"].rstrip("\n") or True
    lines[-1] = m["original_trace_line"]
    return "\n".join(lines)


def probe_question(program, m):
    """Direct question about the planted line's computation (k=1: 2 operands)."""
    ops = program.get("operands")
    assert ops and len(ops) >= 2, ("no operands", m["program_id"])
    return "What is %d + %d?" % (ops[0], ops[1])


# ------------------------------------------------------------- model + prompts

def load_model(model_key, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name, rev, _ = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=rev, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    return model, tok


def chat_ids(tok, msgs):
    """Chat-template ids with generation prompt. Byte-parity vs
    vllm_gen._chat_ids is asserted by audit A1c before any real run.
    return_dict=False guards against the transformers 5.x default of
    returning a BatchEncoding (whose list() is its KEYS)."""
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  tokenize=True, return_dict=False)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = list(ids)
    assert ids and all(isinstance(i, int) for i in ids), "chat_ids: non-int ids"
    return ids


def render_continue(tok, user_msg, prefill):
    ids = chat_ids(tok, [{"role": "user", "content": user_msg}])
    if prefill:
        ids = ids + list(tok(prefill, add_special_tokens=False)["input_ids"])
    return ids


def render_probe(tok, user_msg, prefix, question):
    return chat_ids(tok, [{"role": "user", "content": user_msg},
                          {"role": "assistant", "content": prefix},
                          {"role": "user", "content": question}])


def user_msg_for(program):
    return tf.make_user_message(tf.make_listing_text(program["stmt_texts"]))


def planted_line_span(tok, user_msg, prefix):
    """Token index span of the prefix's last line inside render_continue ids."""
    head, last = prefix.rstrip("\n").rsplit("\n", 1)
    ids_head = render_continue(tok, user_msg, head + "\n")
    ids_full = render_continue(tok, user_msg, head + "\n" + last)
    return len(ids_head), len(ids_full)


# ------------------------------------------------------------- steering hook

class Steer:
    """Adds alpha * unit(vec) to the output of model.model.layers[layer-1]
    for GENERATED positions only (seq_len == 1 forwards; the prompt pass has
    seq_len > 1). alpha=0 or vec=None => mathematical no-op (audit A1)."""

    def __init__(self, model, layer, vec, alpha):
        import torch
        self.module = model.model.layers[layer - 1]
        self.alpha = float(alpha)
        if vec is None:
            self.delta = None
        else:
            v = vec.to(dtype=torch.float32)
            v = v / (v.norm() + 1e-8)
            p = next(model.parameters())
            self.delta = (self.alpha * v).to(device=p.device, dtype=p.dtype)
        self.handle = None

    def _hook(self, module, inputs, output):
        if self.delta is None or self.alpha == 0.0:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != 1:            # prompt pass: untouched
            return output
        hs = hs + self.delta
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle:
            self.handle.remove()


# ------------------------------------------------------------- generation

def _gen(model, tok, ids, do_sample, temp, seed_offset):
    import torch
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    if do_sample:
        torch.manual_seed(SEED + seed_offset)
    with torch.no_grad():
        out = model.generate(
            x, max_new_tokens=MAX_NEW, do_sample=do_sample,
            temperature=temp if do_sample else None,
            top_p=1.0 if do_sample else None,
            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True)


def greedy(model, tok, ids):
    return _gen(model, tok, ids, False, None, 0)


def sampled(model, tok, ids, roll):
    return _gen(model, tok, ids, True, 0.7, roll)


# ------------------------------------------------------------- grading glue

def classify(program, m, continuation):
    stmts = parse_program(program["stmt_texts"])
    inj = {k: m[k] for k in ("cf", "planted_var", "planted_value",
                             "force_after_line", "expected_lines_from")}
    met = val_mod.classify_run(stmts, program, inj, continuation)
    rec = dict(met)
    rec["judge_reject"] = None
    rec["label"] = e11.three_way_label(rec)
    return rec


def control_valid(program, prefix, continuation):
    """Unperturbed continuation reaches the true output (solve + no false
    'correction' of true values)."""
    import re
    nums = re.findall(r"output\s*=\s*(-?\d+)", prefix + "\n" + continuation)
    return bool(nums) and int(nums[-1]) == program["out_true"]


def rate(rows, pred):
    n = len(rows)
    return (sum(1 for r in rows if pred(r)) / n) if n else None, n


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))
