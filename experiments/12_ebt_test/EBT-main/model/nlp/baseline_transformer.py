import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

from transformers import AutoTokenizer

import math
import random
import os
from model.model_utils import *
from data.nlp.synthetic_chain_dataset import SyntheticChainTokenizer


class Baseline_Transformer_NLP(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        if self.hparams.tokenizer == "synthetic_chain" or self.hparams.dataset_name == "synthetic_chain":
            tokenizer = SyntheticChainTokenizer.default()
            self.tokenizer_pad_token_id = tokenizer.pad_token_id
            self.vocab_size = tokenizer.vocab_size
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
            self.tokenizer_pad_token_id = tokenizer.eos_token_id # is token 0, was right padding things
            self.vocab_size = len(tokenizer) # self.vocab_size = tokenizer.vocab_size caused errors since is smaller than len(tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        self.tokenizer = tokenizer
        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)

        self.transformer = setup_transformer(self.hparams)
        
        self.output = nn.Linear(self.hparams.embedding_dim, self.vocab_size, bias = False)
        init_whole_model_weights(self.output, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        if getattr(self.hparams, "tie_token_embeddings", False):
            self.output.weight = self.embeddings.weight
        
        self.finished_warming_up = False
        

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False): # accepts input_ids as input
        embeddings = self.embeddings(x) # x here is input_ids
        
        predicted_embeddings = self.transformer(embeddings, start_pos = start_pos, learning = learning) # BS, S, D
        predicted_logits = self.output(predicted_embeddings) #BS, S, vocab_size
        if return_raw_logits:
            return predicted_logits
        else:
            predicted_distribution = self.log_softmax(predicted_logits).reshape(-1, self.vocab_size) # BS*S, V; reshape since preds for nll should be 2d
            return predicted_distribution
        

    def forward_loss_wrapper(self, x, phase="train"):
        all_input_ids = x['input_ids'].squeeze(dim=1)
        input_ids = all_input_ids[:, :-1] # x['input_ids'] shape is BS, S+1, only input first S--remove next tokens

        predicted_distribution = self(input_ids)
        
        if "labels" in x:
            next_token_indices = x["labels"].squeeze(dim=1)[:, :-1]
            ignore_index = -100
        else:
            next_token_indices = all_input_ids[:, 1:] # squeeze was to remove 1 on 2nd dim
            ignore_index = self.tokenizer_pad_token_id
        if self.hparams.execution_mode == "finetune" and self.hparams.dataset_name not in ["synthetic_chain", "ruletaker"]: # Only tokens after "[[Answer]]: " will be calculated in finetune
            next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        labels_2d = next_token_indices
        next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D
        
        cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=ignore_index)
        ppl_loss = torch.exp(cce_loss).detach()
        logits_3d = predicted_distribution.reshape(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
        accuracy_dict = self._target_accuracy(logits_3d, labels_2d, x)

        log_dict = {
            'loss': cce_loss,
            'perplexity': ppl_loss,
            **accuracy_dict,
        }
        return log_dict

    def _target_accuracy(self, logits, labels, batch):
        if "target_metric_mask" in batch:
            valid_mask = batch["target_metric_mask"].squeeze(dim=1)[:, :-1].bool()
        else:
            valid_mask = labels != self.tokenizer_pad_token_id
        if valid_mask.shape != labels.shape:
            valid_mask = labels != -100
        safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        preds = logits.argmax(dim=-1)
        correct = (preds == safe_labels) & valid_mask
        token_count = valid_mask.sum().clamp_min(1).to(dtype=torch.float32)
        token_accuracy = correct.sum().to(dtype=torch.float32) / token_count
        exact = (correct | ~valid_mask).flatten(1).all(dim=1)
        has_any = valid_mask.flatten(1).any(dim=1)
        exact_accuracy = ((exact & has_any).to(dtype=torch.float32)).mean()
        return {
            "final_token_accuracy": token_accuracy.detach(),
            "final_exact_accuracy": exact_accuracy.detach(),
            "teacher_forcing_token_accuracy": token_accuracy.detach(),
            "teacher_forcing_exact_accuracy": exact_accuracy.detach(),
        }
