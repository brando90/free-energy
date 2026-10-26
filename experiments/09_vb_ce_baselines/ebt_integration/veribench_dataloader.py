# TLDR: VeriBench dataset for the official EBT harness — same py->lean formatting as B1, pretokenized per doc.
"""VeriBench dataloader for the official EBT codebase (alexiglad/EBT).

Drop into EBT/data/nlp/ (use apply_ebt_integration.py). Follows the explicit-split
pattern of gsm8k_dataloader.py. Documents are formatted EXACTLY like B1
(train_b1.py PROMPT_TMPL + lean_text + eos) so cross-entropies are comparable;
only `paired` rows are used (same 702/86/96 docs as B1's conditional CE).

hparams.dataset_dir must point at .../experiments/08_vb_train_val_test/splits.
Requires --pretokenize_dataset (items are token dicts; their NLP_HF_Collator pads).
"""
import json
import os

from torch.utils.data import Dataset
from transformers import AutoTokenizer

PROMPT_TMPL = (
    "/- Translate the following Python program into a Lean 4 formalization with theorems. -/\n\n"
    "-- PYTHON SOURCE:\n{py}\n\n-- LEAN 4:\n"
)


class VeribenchDataset(Dataset):
    def __init__(self, hparams, split):
        if hparams.execution_mode != "pretrain":
            raise ValueError("VeribenchDataset: only pretrain execution_mode is supported (paper-recipe fallback).")
        if not hparams.pretokenize_dataset:
            raise ValueError("VeribenchDataset: pass --pretokenize_dataset (collator expects token dicts).")
        self.max_length = hparams.context_length + 1
        path = os.path.join(hparams.dataset_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"VeriBench split not found: {path} (set --dataset_dir to .../08_vb_train_val_test/splits)")
        tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.items = []
        n_prompt_trunc = n_target_trunc = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("py_code"):
                    continue  # keep the doc set identical to B1's conditional-CE set
                # same truncation policy as B1 (train_b1.py): left-truncate the prompt to keep the target whole
                prompt_ids = tokenizer(PROMPT_TMPL.format(py=r["py_code"]), add_special_tokens=False)["input_ids"]
                target_ids = tokenizer(r["lean_text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
                if len(target_ids) > self.max_length:
                    target_ids = target_ids[: self.max_length]
                    n_target_trunc += 1
                    prompt_ids = []
                elif len(prompt_ids) + len(target_ids) > self.max_length:
                    budget = self.max_length - len(target_ids)  # guard: [-0:] would keep the whole prompt
                    prompt_ids = prompt_ids[-budget:] if budget > 0 else []
                    n_prompt_trunc += 1
                ids = prompt_ids + target_ids
                self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        print(f"[veribench] split={split}: {len(self.items)} docs (paired only), "
              f"{n_prompt_trunc} prompt-truncated / {n_target_trunc} target-truncated at {self.max_length} tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
