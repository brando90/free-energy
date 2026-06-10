# TLDR: B1 baseline — SFT a small LM on VB-train (py->lean), early-stop on val CE, report completion-only test CE.
"""VeriBench CE baseline B1: LM + SFT.

Trains a causal LM on (py_code -> lean_text) pairs from
experiments/08_vb_train_val_test/splits/, with completion-only cross-entropy
(prompt tokens masked) as the primary metric and full-sequence CE secondary.
CE is the token-weighted mean NLL in nats over the split; ppl = exp(CE).

Modes:
    zeroshot  - eval the un-finetuned model on val+test (reference numbers)
    overfit   - single-batch overfit sanity check (CE -> ~0)
    train     - one config (lr, seed): train w/ early stopping on val CE,
                restore best-val weights, eval test once, write JSON
    aggregate - read results/*.json -> results_b1.md

Examples:
    python train_b1.py --mode zeroshot
    python train_b1.py --mode overfit --lr 5e-5
    python train_b1.py --mode train --lr 2e-5 --seed 0
    python train_b1.py --mode aggregate
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROMPT_TMPL = (
    "/- Translate the following Python program into a Lean 4 formalization with theorems. -/\n\n"
    "-- PYTHON SOURCE:\n{py}\n\n-- LEAN 4:\n"
)


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]  # -100 on prompt positions


class PairDataset(Dataset):
    """py->lean pairs, tokenized once. Prompt is left-truncated to keep the target whole."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int):
        self.examples: list[Example] = []
        self.n_prompt_trunc = 0
        self.n_target_trunc = 0
        self.n_skipped_unpaired = 0
        for r in rows:
            if not r.get("py_code"):
                self.n_skipped_unpaired += 1
                continue
            prompt_ids = tokenizer(PROMPT_TMPL.format(py=r["py_code"]), add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(r["lean_text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            if len(target_ids) > max_len:
                target_ids = target_ids[:max_len]
                self.n_target_trunc += 1
                prompt_ids = []
            elif len(prompt_ids) + len(target_ids) > max_len:
                prompt_ids = prompt_ids[-(max_len - len(target_ids)):]
                self.n_prompt_trunc += 1
            self.examples.append(
                Example(input_ids=prompt_ids + target_ids, labels=[-100] * len(prompt_ids) + target_ids)
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def stats(self) -> dict:
        return {
            "n_examples": len(self.examples),
            "n_skipped_unpaired": self.n_skipped_unpaired,
            "n_prompt_truncated": self.n_prompt_trunc,
            "n_target_truncated": self.n_target_trunc,
            "total_target_tokens": sum(sum(l != -100 for l in e.labels) for e in self.examples),
        }


def collate(batch: list[Example], pad_id: int):
    width = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        labels[i, :n] = torch.tensor(e.labels)
        mask[i, :n] = 1
    return ids, labels, mask


@torch.no_grad()
def evaluate(model, loader, device, full_sequence: bool = False) -> tuple[float, int]:
    """Token-weighted mean NLL (nats). full_sequence=True scores all non-pad tokens, not just the target."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for ids, labels, mask in loader:
        ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
        if full_sequence:
            labels = ids.masked_fill(mask == 0, -100)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        total_nll += F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        ).item()
        total_tokens += (shift_labels != -100).sum().item()
    return total_nll / max(total_tokens, 1), total_tokens


def load_rows(data_dir: Path, split: str) -> list[dict]:
    return [json.loads(l) for l in (data_dir / f"{split}.jsonl").open()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(args, splits=("train", "val", "test")):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_dir = Path(args.data_dir).expanduser().resolve()
    sets = {s: PairDataset(load_rows(data_dir, s), tokenizer, args.max_seq_len) for s in splits}
    model = AutoModelForCausalLM.from_pretrained(args.model).float()  # fp32 master weights regardless of transformers version
    model.to(args.device)
    return tokenizer, sets, model


def make_loader(ds, tokenizer, bs, shuffle=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=gen if shuffle else None,
                      collate_fn=lambda b: collate(b, tokenizer.pad_token_id))


def run_zeroshot(args) -> dict:
    tokenizer, sets, model = build(args)
    out = {"mode": "zeroshot", "model": args.model, "max_seq_len": args.max_seq_len,
           "data_stats": {k: v.stats() for k, v in sets.items()}}
    for split in ("val", "test"):
        loader = make_loader(sets[split], tokenizer, args.eval_batch_size)
        ce, ntok = evaluate(model, loader, args.device)
        ce_full, _ = evaluate(model, loader, args.device, full_sequence=True)
        out[f"{split}_ce"] = ce
        out[f"{split}_ppl"] = math.exp(ce)
        out[f"{split}_ce_full_seq"] = ce_full
        out[f"{split}_target_tokens"] = ntok
        print(f"[zeroshot] {split}: completion CE {ce:.4f} nats (ppl {math.exp(ce):.1f}) | full-seq CE {ce_full:.4f}")
    return out


def run_overfit(args) -> dict:
    set_seed(args.seed)
    tokenizer, sets, model = build(args, splits=("train",))
    batch = [sets["train"][i] for i in range(args.batch_size)]
    ids, labels, mask = collate(batch, tokenizer.pad_token_id)
    ids, labels, mask = ids.to(args.device), labels.to(args.device), mask.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.train()
    first = last = None
    for step in range(args.overfit_steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
        if step % 20 == 0 or step == args.overfit_steps - 1:
            print(f"[overfit] step {step:4d} CE {loss.item():.4f}")
    ok = last < 0.05
    print(f"[overfit] CE {first:.3f} -> {last:.4f} | {'PASS' if ok else 'FAIL'} (threshold 0.05)")
    return {"mode": "overfit", "lr": args.lr, "steps": args.overfit_steps,
            "ce_first": first, "ce_last": last, "pass": ok}


def run_train(args) -> dict:
    set_seed(args.seed)
    tokenizer, sets, model = build(args)
    train_loader = make_loader(sets["train"], tokenizer, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(sets["val"], tokenizer, args.eval_batch_size)
    test_loader = make_loader(sets["test"], tokenizer, args.eval_batch_size)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    n_params = sum(p.numel() for p in model.parameters())

    best = {"val_ce": float("inf"), "epoch": -1, "state": None}
    curves, tokens_seen, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        running, running_tok = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (ids, labels, mask) in enumerate(train_loader):
            ids, labels, mask = ids.to(args.device), labels.to(args.device), mask.to(args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            ntok = (shift_labels != -100).sum()
            loss_sum = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                       shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
            (loss_sum / ntok / args.grad_accum).backward()
            running += loss_sum.item()
            running_tok += ntok.item()
            tokens_seen += mask.sum().item()
            if (i + 1) % args.grad_accum == 0 or i == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        val_ce, _ = evaluate(model, val_loader, args.device)
        train_ce = running / max(running_tok, 1)
        curves.append({"epoch": epoch, "train_ce": train_ce, "val_ce": val_ce, "lr": sched.get_last_lr()[0]})
        print(f"[train lr={args.lr} seed={args.seed}] epoch {epoch}: train CE {train_ce:.4f} | val CE {val_ce:.4f}")
        if val_ce < best["val_ce"] - 1e-4:
            best = {"val_ce": val_ce, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= args.patience:
            print(f"[train] early stop at epoch {epoch} (best val CE {best['val_ce']:.4f} @ epoch {best['epoch']})")
            break

    assert best["state"] is not None, "no best checkpoint recorded"
    model.load_state_dict(best["state"])
    model.to(args.device)
    test_ce, test_tok = evaluate(model, test_loader, args.device)
    test_ce_full, _ = evaluate(model, test_loader, args.device, full_sequence=True)
    result = {
        "mode": "train", "model": args.model, "n_params": n_params, "lr": args.lr, "seed": args.seed,
        "max_seq_len": args.max_seq_len, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum, "epochs_max": args.epochs,
        "epochs_ran": len(curves), "best_epoch": best["epoch"], "patience": args.patience,
        "val_ce": best["val_ce"], "test_ce": test_ce, "test_ppl": math.exp(test_ce),
        "test_ce_full_seq": test_ce_full, "test_target_tokens": test_tok,
        "tokens_seen": tokens_seen, "flops_est_6ND": 6 * n_params * tokens_seen,
        "runtime_s": round(time.time() - t0, 1), "curves": curves,
        "data_stats": {k: v.stats() for k, v in sets.items()},
        "tokenizer": args.model, "weight_decay": 0.0, "warmup_ratio": 0.03, "lr_schedule": "cosine",
        "grad_clip": 1.0, "precision": "bf16 autocast, fp32 master weights",
    }
    print(f"[train] DONE lr={args.lr} seed={args.seed}: val CE {best['val_ce']:.4f} | "
          f"test CE {test_ce:.4f} (ppl {math.exp(test_ce):.1f}) | full-seq {test_ce_full:.4f}")
    return result


def run_aggregate(args) -> dict:
    res_dir = Path(args.results_dir)
    runs = [json.loads(p.read_text()) for p in sorted(res_dir.glob("*.json"))]
    zero = next((r for r in runs if r["mode"] == "zeroshot"), None)
    trains = [r for r in runs if r["mode"] == "train"]
    sweep = [r for r in trains if r["seed"] == args.seed_sweep]
    by_lr = sorted(sweep, key=lambda r: r["val_ce"])
    best_lr = by_lr[0]["lr"] if by_lr else None
    finals = [r for r in trains if r["lr"] == best_lr]
    lines = [
        "# B1 results — LM + SFT on VeriBench train/val/test",
        "",
        f"**TLDR:** {finals[0]['model'] if finals else '?'} full-SFT on VB-train (707 tasks): "
        f"test completion-CE {np.mean([r['test_ce'] for r in finals]):.4f} ± {np.std([r['test_ce'] for r in finals]):.4f} nats "
        f"(ppl {np.mean([r['test_ppl'] for r in finals]):.2f}) over {len(finals)} seeds vs zero-shot "
        f"{zero['test_ce'] if zero else float('nan'):.4f} nats (ppl {zero['test_ppl'] if zero else float('nan'):.1f}). "
        if finals else "**TLDR:** incomplete — missing final runs.",
        "",
        "Completion-only CE (prompt tokens masked) in nats; ppl = exp(CE). Full protocol: `PROMPT.md`.",
        "",
        "## LR sweep (seed 0, early stop on val CE)",
        "",
        "| LR | epochs ran | best epoch | val CE | test CE* |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(sweep, key=lambda r: r["lr"]):
        lines.append(f"| {r['lr']:g} | {r['epochs_ran']} | {r['best_epoch']} | {r['val_ce']:.4f} | {r['test_ce']:.4f} |")
    lines += ["", "*test CE shown for completeness; best LR was selected on val only.", ""]
    if finals:
        f0 = finals[0]
        lines += [
            f"## Final: best LR = {best_lr:g} × {len(finals)} seeds",
            "",
            "| model | params | LR | sched | eff. batch | max epochs | best epoch (per seed) | tokens seen (mean) | "
            "val CE (mean±std) | **test CE (mean±std)** | test ppl | full-seq test CE | zero-shot test CE |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            "| {model} | {params:.0f}M | {lr:g} | cosine, wu 3% | {eb} | {me} | {be} | {tok:.2e} | "
            "{vm:.4f}±{vs:.4f} | **{tm:.4f}±{ts:.4f}** | {pm:.2f} | {fm:.4f} | {z:.4f} |".format(
                model=f0["model"], params=f0["n_params"] / 1e6, lr=best_lr, eb=f0["effective_batch"],
                me=f0["epochs_max"], be=",".join(str(r["best_epoch"]) for r in finals),
                tok=float(np.mean([r["tokens_seen"] for r in finals])),
                vm=np.mean([r["val_ce"] for r in finals]), vs=np.std([r["val_ce"] for r in finals]),
                tm=np.mean([r["test_ce"] for r in finals]), ts=np.std([r["test_ce"] for r in finals]),
                pm=np.mean([r["test_ppl"] for r in finals]),
                fm=np.mean([r["test_ce_full_seq"] for r in finals]),
                z=zero["test_ce"] if zero else float("nan"),
            ),
            "",
            f"FLOPs estimate (6·N·D): {np.mean([r['flops_est_6ND'] for r in finals]):.2e} per final run. "
            f"Truncation at max_seq_len {f0['max_seq_len']}: "
            f"{f0['data_stats']['train']['n_prompt_truncated']} prompt-truncated, "
            f"{f0['data_stats']['train']['n_target_truncated']} target-truncated of "
            f"{f0['data_stats']['train']['n_examples']} train examples; "
            f"{f0['data_stats']['train']['n_skipped_unpaired']}/1/1 unpaired rows skipped (train/val/test).",
        ]
    lines += [
        "",
        "## Repro",
        "```bash",
        "python train_b1.py --mode zeroshot",
        "python train_b1.py --mode overfit --lr 5e-5",
        "for lr in 1e-5 2e-5 5e-5; do python train_b1.py --mode train --lr $lr --seed 0; done",
        f"for s in 0 1 2; do python train_b1.py --mode train --lr {best_lr:g} --seed $s; done" if best_lr else "",
        "python train_b1.py --mode aggregate",
        "```",
    ]
    out_path = res_dir.parent / "results_b1.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {out_path}")
    return {"mode": "aggregate", "best_lr": best_lr, "n_final_seeds": len(finals)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["zeroshot", "overfit", "train", "aggregate"])
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent / "08_vb_train_val_test" / "splits"))
    p.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--overfit-steps", type=int, default=150)
    p.add_argument("--seed-sweep", type=int, default=0, help="seed used for the LR sweep (aggregate mode)")
    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode != "aggregate":
        assert args.device == "cuda", "refusing to run on CPU — set CUDA_VISIBLE_DEVICES"

    result = {"zeroshot": run_zeroshot, "overfit": run_overfit, "train": run_train, "aggregate": run_aggregate}[args.mode](args)

    if args.mode in ("zeroshot", "train"):
        res_dir = Path(args.results_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        name = "zeroshot.json" if args.mode == "zeroshot" else f"b1_lr{args.lr:g}_seed{args.seed}.json"
        (res_dir / name).write_text(json.dumps(result, indent=2) + "\n")
        print(f"[main] wrote {res_dir / name}")


if __name__ == "__main__":
    main()
