import os
import sys
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # append parent dir

from data.nlp.pajama_dataloader import RedPajamaDataset
from data.nlp.collator import NLP_HF_Collator
from model.nlp.baseline_transformer import Baseline_Transformer_NLP
from model.nlp.ebt import EBT_NLP

########################################################################################################################

# NOTE IMPORTANT this code is not to reproduce results because it doesnt have all features (LR scheduler, correct wd on all params, etc); it is just a proof of concept for how things work. results are far from exact; it is recommended to use the whole codebase 

########################################################################################################################

class ModelWrapper(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(hparams)

        model_cls = {
            "baseline_transformer": Baseline_Transformer_NLP,
            "ebt": EBT_NLP,
        }[self.hparams.model_name]

        self.model = model_cls(self.hparams)
        self.dataset = RedPajamaDataset(self.hparams)
        self.collate_fn = NLP_HF_Collator(self.hparams)

    def training_step(self, batch, batch_idx):
        metrics = self.model.forward_loss_wrapper(batch, "train")
        loss = metrics["loss"]
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        workers = torch.cuda.device_count() * self.hparams.num_workers_per_gpu
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size_per_device,
            shuffle=True,
            num_workers=workers,
            collate_fn=self.collate_fn,
        )

def main():
    hparams = dict(
        # optimisation
        lr=1e-3,
        batch_size_per_device=32,
        num_workers_per_gpu=12,
        max_steps=100000,
        # data
        dataset_dir="",
        dataset_name="pajama",
        context_length=256,
        pretokenize_dataset=True,
        tokenizer="EleutherAI/gpt-neox-20b",
        # model choice
        model_name="ebt",  # "baseline_transformer" or "ebt"
        # model size
        embedding_dim=384,
        num_transformer_blocks=6,
        multiheaded_attention_heads=6,
        ffn_dim_multiplier=1,
        weight_initialization_method="xavier",
        weight_initialization_gain=1.0,
        # misc
        execution_mode="pretrain",
        debug_unused_parameters=False
    )

    ebt_params = dict( #NOTE 
        mcmc_step_size=500.0,
        mcmc_step_size_lr_multiplier=1500.0, 
        mcmc_num_steps=2,
        ebt_type="time_embed",
        normalize_initial_condition=True,
        denoising_initial_condition="random_noise",
        mcmc_step_size_learnable=True,
        no_mcmc_detach=False,
        # only up to these first ones are actually used, and really only the first four of them are real hparams (the others can stay as they are so normalize, learnable, and condition dont need to be tuned). time_embed can also almost always stay and mcmc_num_steps = 2 is very safe. keeping mcmc_step_size_lr_multiplier = 3x mcmc_step_size is safe and what works well so the most important and arguably only really neccesary to tune hparam is mcmc_step_size

        # below are just to make existing code run well and happy :) they are not used at all. you can try them out if you ever want to add fancier hparams but are not recommended for getting started
        ebt_norm="rms",
        ebt_act_func="silu",
        dyt_alpha_init=0.5,
        mcmc_replay_buffer=False,
        gaussian_random_noise_scaling=1.0,
        normalize_initial_condition_only_first_step=False,
        randomize_mcmc_step_size_scale=1.0,
        randomize_mcmc_num_steps=0,
        randomize_mcmc_num_steps_min=0,
        randomize_mcmc_num_steps_final_landscape=False,
        langevin_dynamics_noise=0.0,
        langevin_dynamics_noise_learnable=False,
        vocab_to_embed_uses_prob_dist=False,
        num_modality_processing_mlp_layers=1,
        truncate_mcmc=False,
        clamp_futures_grad=False,
        clamp_futures_grad_max_change=9.0,
        absolute_clamp=0.0,
        clamp_max_after_warm_up=0.0,
        sharpen_predicted_distribution=0.0,
        reconstruction_coeff=1.0,
        contrastive_loss=False,
        contrastive_loss_coeff=0.0005,
        soften_target_prob_dist=0.0,
    )
    hparams.update(ebt_params)

    model = ModelWrapper(hparams)
    logger = WandbLogger(
        name="minimal_wrapper_run", project="nlp_pretrain_minimal", entity=""
    )

    trainer = pl.Trainer(
        max_steps=hparams["max_steps"],
        devices=-1,
        logger=logger,
        max_epochs=1,
        enable_model_summary=True,
        enable_checkpointing=False,
    )
    trainer.fit(model)




if __name__ == "__main__":
    main()
