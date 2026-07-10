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



class Diffusion_Transformer_IMG_Denoise(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.diffusion_transformer = setup_diffusion_transformer(self.hparams)
        self.diffusion = create_diffusion(timestep_respacing="", diffusion_steps=int(self.hparams.diffusion_steps / self.hparams.denoise_images_noise_level)) # divide this so can use diffusion module with partial steps
        
        self.finished_warming_up = False

    def forward(self, x, t, y = None): # x here is image_embeddings, y here is caption_embeddings. t is timestep
        return self.diffusion_transformer(x, t, y)
        
    
    def forward_loss_wrapper(self, x, phase="train"):
        real_images = x['image']
        # noised_images = real_images + torch.randn_like(real_images.detach(), device=real_images.device) * self.hparams.denoise_images_noise_level
        # t = torch.randint(0, self.diffusion.num_timesteps, (real_images.shape[0],), device=self.device)
        t = torch.randint(0, self.hparams.diffusion_steps, (real_images.shape[0],), device=self.device)
        # noised_images = self.diffusion.q_sample(real_images, t)

        model_kwargs = dict(y=None)
        
        log_dict = self.diffusion.training_losses(self.forward, real_images, t, model_kwargs) # logs keys vb, mse, loss -- changed the names
        key_map = {"vb": "vb_loss", "mse": "mse_conditioned_loss"}
        log_dict = {key_map.get(k, k): (v.mean().detach() if k != 'loss' else v.mean()) for k, v in log_dict.items()}

        is_update_step = (self.trainer.fit_loop.epoch_loop.batch_progress.current.completed + 1) % self.trainer.accumulate_grad_batches == 0 # so dont have issues with gradient accumulation
        if self.trainer.global_step % self.hparams.log_image_every_n_steps == 0 and is_update_step and phase == "train":
        
            with torch.no_grad():
                # just denoise single sample, if want many can remove the [0].unsqueeze(0) and other indexing. dont have here since takes too long
                model_kwargs = dict(y=None)
                real_images_shape = real_images[0].unsqueeze(0).shape
                # noised_images = (real_images + torch.randn_like(real_images.detach(), device=real_images.device) * self.hparams.denoise_images_noise_level)[0].unsqueeze(0)
                t = torch.tensor(self.hparams.diffusion_steps-1, device=self.device).expand(real_images_shape[0])[0] # makes so is max step for noising
                noised_images = self.diffusion.q_sample(real_images, t)[0].unsqueeze(0)
                if self.hparams.use_deterministic_reverse:
                    samples = self.diffusion.ddim_sample_loop(
                        self.forward, real_images_shape, noise=noised_images, model_kwargs=model_kwargs, progress=True, device=self.device, eta=0.0, timestep_start=t
                    )
                else:
                    samples = self.diffusion.p_sample_loop(
                        self.forward, real_images_shape, noised_images, model_kwargs=model_kwargs, progress=True, device=self.device, timestep_start=t
                    )
                
                mse_raw_loss = torch.mean(mean_flat((real_images[0].unsqueeze(0) - samples) ** 2)).detach()
                log_dict['mse_raw_loss'] = mse_raw_loss # this is loss with no conditioning (i.e. unlike the mse loss above where you just predict x_{t-1}), this is from just sampling from prior

                predicted_samples = torch.cat((samples[0].unsqueeze(0), real_images[0].unsqueeze(0), noised_images[0].unsqueeze(0)), dim = 0) # only use first index of generated samples for loss calc
                decoded_images = denormalize(predicted_samples.unsqueeze(0), self.hparams.dataset_name, self.device, self.hparams.custom_image_normalization, True).squeeze(0)

                log_dict['denoised_image'] = decoded_images[0]
                log_dict['original_image'] = decoded_images[1]
                log_dict['noised_image'] = decoded_images[2]

        return log_dict
