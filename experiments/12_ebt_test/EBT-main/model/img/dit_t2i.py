import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
import traceback
from torchvision.transforms import functional as TF
import torchvision.models as models
import math
import random
from diffusers import AutoencoderKL

from model.model_utils import *
from model.diffusion import create_diffusion
from model.diffusion.gaussian_diffusion import mean_flat



class Diffusion_Transformer_IMG_T2I(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size, device = self.device, use_ema=True) # dtype 32 by default, for here is actually a VAE
        self.text_encoder, self.text_encoder_dim = get_clip_text_encoder(self.hparams.clip_text_encoder_size)
        self.text_embeddings_projector = nn.Linear(self.text_encoder_dim, self.hparams.embedding_dim, bias=True)

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.diffusion_transformer = setup_diffusion_transformer(self.hparams)
        self.diffusion = create_diffusion(timestep_respacing="", diffusion_steps=self.hparams.diffusion_steps)

        self.reset_image_encoder_decoder = False # this is bc of a bug with the vae requiring reloading to use... i think its some interact with PL
        
        self.finished_warming_up = False

    def forward(self, x, t, y): # x here is image_embeddings, y here is caption_embeddings. t is timestep
        projected_caption_embeddings = self.text_embeddings_projector(y)
        return self.diffusion_transformer(x, t, y=projected_caption_embeddings)
        
    
    def forward_loss_wrapper(self, x, phase="train"):
        images = x['image']
        captions = x['caption']
        image_embeddings = get_encoded_images(images, self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization = True)
        caption_embeddings = get_text_embeddings(self.text_encoder, captions)
        t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device)

        model_kwargs = dict(y=caption_embeddings)
        
        log_dict = self.diffusion.training_losses(self.forward, image_embeddings, t, model_kwargs) # logs keys vb, mse, loss -- change names
        key_map = {"vb": "vb_loss", "mse": "mse_conditioned_loss"}
        log_dict = {key_map.get(k, k): (v.mean().detach() if k != 'loss' else v.mean()) for k, v in log_dict.items()}

        is_update_step = (self.trainer.fit_loop.epoch_loop.batch_progress.current.completed + 1) % self.trainer.accumulate_grad_batches == 0 # so dont have issues with gradient accumulation
        if self.trainer.global_step % self.hparams.log_image_every_n_steps == 0 and is_update_step and phase == "train":
            if not self.reset_image_encoder_decoder: # this is done to prevent bug where loading ckpt image encoder doesnt work well, not sure why ckpt image decoder doesnt load well, maybe related to HF or PL
                    self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size, use_ema=True).to(self.device)
                    self.image_encoder.eval()
                    for param in self.image_encoder.parameters():
                        param.requires_grad = False
                    self.reset_image_encoder_decoder = True
            
            with torch.no_grad():
                # just denoise single sample, if want many can remove the [0].unsqueeze(0) and other indexing. dont have here since takes too long
                model_kwargs = dict(y=caption_embeddings[0].unsqueeze(0))
                image_embeddings_shape = image_embeddings[0].unsqueeze(0).shape
                z = torch.randn(*image_embeddings_shape, device=self.device)
                samples = self.diffusion.p_sample_loop(
                    self.forward, image_embeddings_shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device
                )

                mse_raw_loss = torch.mean(mean_flat((image_embeddings[0].unsqueeze(0) - samples) ** 2)).detach()
                log_dict['mse_raw_loss'] = mse_raw_loss # this is loss with no conditioning (i.e. unlike the mse loss above where you just predict x_{t-1}), this is from just sampling from prior

                samples_for_decoding = torch.cat((samples[0].unsqueeze(0), image_embeddings[0].unsqueeze(0)), dim = 0) # only use first index of generated samples for loss calc
                decoded_images = self.image_encoder.decode(samples_for_decoding.reshape(-1, *image_embeddings_shape[1:]) / 0.18215).sample  # constant following SDXL VAE https://github.com/CompVis/stable-diffusion
                decoded_images = denormalize(decoded_images.unsqueeze(0), self.hparams.dataset_name, self.device, self.hparams.custom_image_normalization, True).squeeze(0)

                log_dict['generated_image'] = decoded_images[0]
                log_dict['original_image'] = decoded_images[1]

        return log_dict
