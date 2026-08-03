"""Offline vLLM generation backend for the latent-recovery harness.

Reproduces the HF-transformers generation semantics of src/common.py /
src/expc_polarity_control.py:

  * chat template applied to a single user message
    (INSTR + FEWSHOT + "Q: {question} Prove: {target}\nA:"),
    add_generation_prompt=True
  * assistant turn pre-filled with an answer prefix, tokenized with
    add_special_tokens=False and concatenated AT THE TOKEN-ID LEVEL
    (this is what common.make_prompt_ids does -- we feed vLLM
    prompt_token_ids, so the prompt is byte-for-byte identical to the
    HF path by construction; a string-render cross-check is available
    via --verify-prompt)
  * greedy decoding (temperature=0), max_new_tokens cap, stop on the
    model generation_config's eos ids (HF model.generate default),
    decode with skip_special_tokens=True

Public API:
    generate_continuations(model_id, rows, max_new_tokens=192, revision=None)

  Each row is a dict with ONE of:
    "prompt_token_ids": [int, ...]          # exact ids (preferred)
    "prompt": "..."                          # fully rendered string ending
                                             # mid-assistant-turn
    "messages": [...], "assistant_prefill": "..."   # rendered like common.py
  Returns a list of dicts (same order): {"text", "token_ids", "finish_reason"}.

CLI:
    python vllm_gen.py --replay raw_generations.jsonl --manifest manifest.jsonl \
        --run-metadata run_metadata.json --n 20 --model Qwen/Qwen2.5-7B-Instruct \
        --revision <sha> [--validator-src /path/to/src] [--out replay_report.json]
    python vllm_gen.py --benchmark --manifest manifest.jsonl --n 100 --model ... \
        --run-metadata run_metadata.json
    python vllm_gen.py --verify-prompt --manifest manifest.jsonl --model ... \
        --run-metadata run_metadata.json
"""
import argparse
import json
import os
import sys
import time

# Defaults copied verbatim from src/common.py (cross-checked against the
# "prompt" block of run_metadata.json at replay time).
INSTR = ("You will be given facts and rules about fictional creatures, then asked to prove a statement. "
         "Answer with only the proof: a sequence of statements, one deduction at a time, in the exact "
         "style of the examples. End with the statement to be proven.\n\n")
FEWSHOT = """Q: Every yumpus is a dumpus. Dumpuses are tumpuses. Tumpuses are not bright. Sam is a yumpus. Prove: Sam is not bright.
A: Sam is a yumpus. Every yumpus is a dumpus. Sam is a dumpus. Dumpuses are tumpuses. Sam is a tumpus. Tumpuses are not bright. Sam is not bright.

Q: Each gorpus is a sterpus. Sterpuses are red. Every borpus is a gorpus. Alex is a borpus. Prove: Alex is red.
A: Alex is a borpus. Every borpus is a gorpus. Alex is a gorpus. Each gorpus is a sterpus. Alex is a sterpus. Sterpuses are red. Alex is red.

"""


def get_tokenizer(model_id, revision=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def user_message(question, target, instruction=INSTR, fewshot=FEWSHOT):
    return instruction + fewshot + f"Q: {question} Prove: {target}\nA:"


def _chat_ids(tok, msgs):
    """apply_chat_template -> flat id list (handles v4 list / v5 BatchEncoding)."""
    out = tok.apply_chat_template(msgs, add_generation_prompt=True)
    if hasattr(out, "data") and "input_ids" in getattr(out, "data", {}):
        out = out["input_ids"]
    elif isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def render_prompt_ids(tok, question, target, answer_prefix=None,
                      instruction=INSTR, fewshot=FEWSHOT):
    """Exact replica of common.make_prompt_ids (token-id concat)."""
    msgs = [{"role": "user", "content": user_message(question, target, instruction, fewshot)}]
    ids = _chat_ids(tok, msgs)
    if answer_prefix:
        ids = ids + list(tok(answer_prefix, add_special_tokens=False)["input_ids"])
    return ids


def render_prompt_string(tok, question, target, answer_prefix=None,
                         instruction=INSTR, fewshot=FEWSHOT):
    """String-render path (add_generation_prompt + manual concat)."""
    msgs = [{"role": "user", "content": user_message(question, target, instruction, fewshot)}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s + (answer_prefix or "")


def expc_answer_prefix(manifest_row):
    """Exact replica of the EXPC generate stage prefix construction."""
    return " " + " ".join(manifest_row["prefix_steps"] + [manifest_row["injected_statement"]])


def eos_token_ids(model_id, tok, revision=None):
    """Union of tokenizer eos + generation_config eos ids (HF generate default)."""
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(model_id, revision=revision)
        eos = gc.eos_token_id
        if eos is not None:
            for e in (eos if isinstance(eos, (list, tuple)) else [eos]):
                ids.add(int(e))
    except Exception:
        pass
    return sorted(ids)


_LLM_CACHE = {}


def _get_llm(model_id, revision=None):
    key = (model_id, revision)
    if key not in _LLM_CACHE:
        import os
        from vllm import LLM
        # Judge-stage-only overrides via env; generation leaves these UNSET so its
        # behavior is unchanged. On <100GB-VRAM nodes (e.g. 80GB A100 vs 143GB H200)
        # the 32B judge needs a smaller max_model_len to leave room for KV cache.
        _extra = {}
        _mml = os.environ.get("EXPG_LLM_MAX_MODEL_LEN")
        if _mml:
            _extra["max_model_len"] = int(_mml)
        _gmu = os.environ.get("EXPG_LLM_GPU_MEM_UTIL")
        _extra["gpu_memory_utilization"] = float(_gmu) if _gmu else 0.85
        _LLM_CACHE[key] = LLM(
            model=model_id,
            revision=revision,
            tokenizer_revision=revision,
            dtype="bfloat16",
            seed=0,
            enable_prefix_caching=False,  # determinism: no cross-request state
            **_extra,
        )
    return _LLM_CACHE[key]


def generate_continuations(model_id, rows, max_new_tokens=192, revision=None):
    """Greedy continuations for a batch of independent prompts.

    rows: list of dicts, each with "prompt_token_ids" OR "prompt" OR
          ("messages" [+ "assistant_prefill"]).
    Returns list of {"text", "token_ids", "finish_reason"} in input order.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    tok = get_tokenizer(model_id, revision)
    prompts = []
    for r in rows:
        if "prompt_token_ids" in r:
            prompts.append(TokensPrompt(prompt_token_ids=list(r["prompt_token_ids"])))
        elif "prompt" in r:
            prompts.append(r["prompt"])
        elif "messages" in r:
            ids = _chat_ids(tok, r["messages"])
            pre = r.get("assistant_prefill")
            if pre:
                ids = ids + list(tok(pre, add_special_tokens=False)["input_ids"])
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        else:
            raise ValueError("row needs prompt_token_ids, prompt, or messages")

    sp = SamplingParams(
        temperature=0.0,                    # greedy == HF do_sample=False
        max_tokens=max_new_tokens,          # == HF max_new_tokens
        stop_token_ids=eos_token_ids(model_id, tok, revision),
        skip_special_tokens=True,           # == HF decode(skip_special_tokens=True)
    )
    llm = _get_llm(model_id, revision)
    outs = llm.generate(prompts, sp)        # returns in input order
    results = []
    for o in outs:
        c = o.outputs[0]
        results.append({
            "text": c.text,
            "token_ids": list(c.token_ids),
            "finish_reason": c.finish_reason,
        })
    return results


# ---------------------------------------------------------------- CLI helpers

def read_jsonl(path, limit=None):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_prompt_block(run_metadata_path):
    instruction, fewshot = INSTR, FEWSHOT
    if run_metadata_path and os.path.exists(run_metadata_path):
        meta = json.load(open(run_metadata_path))
        p = meta.get("prompt", {})
        instruction = p.get("instruction", instruction)
        fewshot = p.get("fewshot", fewshot)
    return instruction, fewshot


def replay_rows(args):
    """Pick N replayable rows (model-generated, non-failed) + manifest join."""
    manifest = {r["run_id"]: r for r in read_jsonl(args.manifest)}
    picked = []
    for r in read_jsonl(args.replay):
        if r.get("failed_generation") or r.get("generated_by_model") is False:
            continue
        if "decoding" not in r or r["run_id"] not in manifest:
            continue
        picked.append((r, manifest[r["run_id"]]))
        if len(picked) >= args.n:
            break
    return picked


def cmd_verify_prompt(args):
    tok = get_tokenizer(args.model, args.revision)
    instruction, fewshot = load_prompt_block(args.run_metadata)
    m = next(iter(read_jsonl(args.manifest, limit=1)))
    pre = expc_answer_prefix(m)
    ids = render_prompt_ids(tok, m["question"], m["target"], pre, instruction, fewshot)
    s = render_prompt_string(tok, m["question"], m["target"], pre, instruction, fewshot)
    dec = tok.decode(ids)
    reenc = tok(s, add_special_tokens=False)["input_ids"]
    print(json.dumps({
        "run_id": m["run_id"],
        "ids_path_len": len(ids),
        "string_path_len": len(reenc),
        "decoded_ids_equals_string": dec == s,
        "reencoded_string_equals_ids": list(reenc) == list(ids),
        "first_id_divergence": next((i for i, (a, b) in enumerate(zip(ids, reenc)) if a != b),
                                    None if len(ids) == len(reenc) else min(len(ids), len(reenc))),
        "prompt_tail_decoded": dec[-160:],
    }, indent=2))


def cmd_replay(args):
    tok = get_tokenizer(args.model, args.revision)
    instruction, fewshot = load_prompt_block(args.run_metadata)
    picked = replay_rows(args)
    rows, refs = [], []
    for raw, m in picked:
        pre = expc_answer_prefix(m)
        ids = render_prompt_ids(tok, m["question"], m["target"], pre, instruction, fewshot)
        mx = raw.get("decoding", {}).get("max_new_tokens", args.max_new_tokens)
        rows.append({"prompt_token_ids": ids, "_max": mx})
        refs.append((raw, m))
    max_new = max(r["_max"] for r in rows)
    if args.sequential:  # batch-1 submission: isolates batching from kernel numerics
        outs = []
        for r in rows:
            outs += generate_continuations(args.model, [r], max_new_tokens=max_new,
                                           revision=args.revision)
    else:
        outs = generate_continuations(args.model, rows, max_new_tokens=max_new,
                                      revision=args.revision)
    eos = set(eos_token_ids(args.model, tok, args.revision))
    for o in outs:  # drop trailing stop token: HF continuation text excludes it
        while o["token_ids"] and o["token_ids"][-1] in eos:
            o["token_ids"] = o["token_ids"][:-1]

    validator = None
    if args.validator_src:
        sys.path.insert(0, args.validator_src)
        import validator as _v
        validator = _v

    n_exact, flips, report_rows = 0, 0, []
    for (raw, m), out in zip(refs, outs):
        stored, new = raw["continuation"], out["text"]
        exact = stored == new
        n_exact += exact
        stored_ids = tok(stored, add_special_tokens=False)["input_ids"]
        if exact:
            div = None
        else:
            div = next((i for i, (a, b) in enumerate(zip(stored_ids, out["token_ids"]))
                        if a != b), None)
            if div is None and len(stored_ids) != len(out["token_ids"]):
                div = min(len(stored_ids), len(out["token_ids"]))
        rec = {"run_id": raw["run_id"], "condition": raw["condition"],
               "exact_match": exact, "first_divergence_token": div,
               "stored_len_tokens": len(stored_ids),
               "new_len_tokens": len(out["token_ids"]),
               "finish_reason": out["finish_reason"]}
        if validator is not None:
            v_old = validator.validate_continuation(
                m["question"], m["prefix_steps"], m["injected_statement"],
                stored, m["target"], m["entity"])
            v_new = validator.validate_continuation(
                m["question"], m["prefix_steps"], m["injected_statement"],
                new, m["target"], m["entity"])
            c_old = v_old.get("class") if isinstance(v_old, dict) else v_old
            c_new = v_new.get("class") if isinstance(v_new, dict) else v_new
            rec["class_stored"], rec["class_new"] = c_old, c_new
            rec["class_flip"] = c_old != c_new
            flips += rec["class_flip"]
        if not exact:
            rec["stored_continuation"] = stored
            rec["new_continuation"] = new
        report_rows.append(rec)

    summary = {"n": len(report_rows), "exact_match": n_exact,
               "exact_match_rate": n_exact / max(1, len(report_rows)),
               "class_flips": flips if validator is not None else None,
               "model": args.model, "revision": args.revision,
               "sequential": bool(args.sequential),
               "rows": report_rows}
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps({k: summary[k] for k in
                      ("n", "exact_match", "exact_match_rate", "class_flips")}, indent=2))


def cmd_benchmark(args):
    tok = get_tokenizer(args.model, args.revision)
    instruction, fewshot = load_prompt_block(args.run_metadata)
    ms = read_jsonl(args.manifest, limit=args.n)
    rows = [{"prompt_token_ids": render_prompt_ids(
        tok, m["question"], m["target"], expc_answer_prefix(m), instruction, fewshot)}
        for m in ms]
    _get_llm(args.model, args.revision)          # exclude engine startup
    t0 = time.time()
    outs = generate_continuations(args.model, rows, max_new_tokens=args.max_new_tokens,
                                  revision=args.revision)
    dt = time.time() - t0
    gen_tok = sum(len(o["token_ids"]) for o in outs)
    prompt_tok = sum(len(r["prompt_token_ids"]) for r in rows)
    print(json.dumps({
        "n_prompts": len(rows), "wall_seconds": round(dt, 2),
        "generated_tokens": gen_tok, "prompt_tokens": prompt_tok,
        "gen_tokens_per_sec": round(gen_tok / dt, 1),
        "prompts_per_sec": round(len(rows) / dt, 2),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--run-metadata", default=None)
    ap.add_argument("--replay", default=None, help="raw_generations.jsonl to replay")
    ap.add_argument("--sequential", action="store_true",
                    help="replay: submit prompts one at a time (batch-1)")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--verify-prompt", action="store_true")
    ap.add_argument("--validator-src", default=None,
                    help="path to src/ (read-only) for validator class comparison")
    ap.add_argument("--out", default=None, help="write full replay report JSON here")
    args = ap.parse_args()
    if args.verify_prompt:
        cmd_verify_prompt(args)
    elif args.replay:
        cmd_replay(args)
    elif args.benchmark:
        cmd_benchmark(args)
    else:
        ap.error("choose one of --replay / --benchmark / --verify-prompt")


if __name__ == "__main__":
    main()
