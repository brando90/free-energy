# TLDR: Score a trained EBT checkpoint on a VB split — completion-only CE (B1-comparable) + full-seq CE.
"""Evaluate an official-EBT checkpoint on the VeriBench split with B1's masking protocol.

Run from anywhere; needs the patched EBT checkout (apply_ebt_integration.py) on disk.
Computes token-weighted mean NLL (nats) over: (a) completion tokens only (prompt
masked — primary, comparable to B1's completion-only CE), (b) the full sequence.
Matches EBT training semantics by excluding eos-as-pad targets (their CE uses
ignore_index = eos); the per-doc effect is ~1 token in ~2k, noted in results_b2.md.

NOTE: no torch.no_grad() — EBT inference *requires* autograd (energy-gradient MCMC).

Usage:
    python eval_b2.py --ckpt <file-or-dir> [--ebt-dir ../../EBT] [--split test]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROMPT_TMPL = (
    "/- Translate the following Python program into a Lean 4 formalization with theorems. -/\n\n"
    "-- PYTHON SOURCE:\n{py}\n\n-- LEAN 4:\n"
)


def pick_ckpt(path: Path) -> Path:
    if path.is_file():
        return path
    cands = sorted(path.rglob("*.ckpt"))
    if not cands:
        raise SystemExit(f"no .ckpt under {path}")
    scored = [(float(c.stem.split("valid_loss=")[-1]), c) for c in cands if "valid_loss=" in c.stem]
    return min(scored)[1] if scored else max(cands, key=lambda c: c.stat().st_mtime)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ebt-dir", default=str(Path(__file__).resolve().parent / ".." / ".." / "EBT"))
    p.add_argument("--splits-dir", default=str(Path(__file__).resolve().parent.parent / "08_vb_train_val_test" / "splits"))
    p.add_argument("--split", default="test")
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ebt_dir = Path(args.ebt_dir).resolve()
    sys.path.insert(0, str(ebt_dir))
    from base_model_trainer import ModelTrainer  # noqa: E402

    ckpt = pick_ckpt(Path(args.ckpt).resolve())
    module = ModelTrainer.load_from_checkpoint(str(ckpt), map_location="cuda")
    model = module.model.cuda().eval()
    tok_name = module.hparams.tokenizer
    from transformers import AutoTokenizer  # noqa: E402
    tokenizer = AutoTokenizer.from_pretrained(tok_name, clean_up_tokenization_spaces=False)
    eos = tokenizer.eos_token_id
    max_len = args.context_length + 1

    rows = [json.loads(l) for l in open(Path(args.splits_dir) / f"{args.split}.jsonl") if json.loads(l).get("py_code")]
    comp_nll = comp_tok = full_nll = full_tok = 0.0
    for i, r in enumerate(rows):
        prompt_ids = tokenizer(PROMPT_TMPL.format(py=r["py_code"]), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(r["lean_text"], add_special_tokens=False)["input_ids"] + [eos]
        if len(target_ids) > max_len:
            target_ids, prompt_ids = target_ids[:max_len], []
        elif len(prompt_ids) + len(target_ids) > max_len:
            prompt_ids = prompt_ids[-(max_len - len(target_ids)):]
        ids = torch.tensor([prompt_ids + target_ids], device="cuda")
        x, targets = ids[:, :-1], ids[:, 1:]
        dists, _ = model(x, learning=False, return_raw_logits=False, no_randomness=True)
        logp = dists[-1].view(1, x.shape[1], -1)  # final MCMC step, log-softmaxed
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        keep = targets != eos  # match EBT training: eos==pad is ignore_index
        comp_start = max(len(prompt_ids) - 1, 0)  # first predicted completion token
        comp_mask = torch.zeros_like(keep)
        comp_mask[:, comp_start:] = True
        comp_mask &= keep
        comp_nll += nll[comp_mask].sum().item()
        comp_tok += comp_mask.sum().item()
        full_nll += nll[keep].sum().item()
        full_tok += keep.sum().item()
        del dists, logp, nll
        if (i + 1) % 20 == 0:
            print(f"[eval_b2] {i+1}/{len(rows)} docs | running completion CE {comp_nll/max(comp_tok,1):.4f}")

    res = {
        "mode": "b2_eval", "ckpt": str(ckpt), "split": args.split, "n_docs": len(rows),
        "context_length": args.context_length, "tokenizer": tok_name,
        "completion_ce": comp_nll / max(comp_tok, 1), "completion_ppl": math.exp(comp_nll / max(comp_tok, 1)),
        "completion_tokens": int(comp_tok),
        "full_seq_ce": full_nll / max(full_tok, 1), "full_seq_tokens": int(full_tok),
        "eos_targets_excluded": True,
    }
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "results" / f"b2_eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[eval_b2] {args.split}: completion CE {res['completion_ce']:.4f} nats (ppl {res['completion_ppl']:.2f}) | "
          f"full-seq CE {res['full_seq_ce']:.4f} | wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
