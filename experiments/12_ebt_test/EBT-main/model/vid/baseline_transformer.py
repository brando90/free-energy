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
from model.model_utils import *


class Baseline_Transformer_VID(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        self.reconstruction_criterion = nn.SmoothL1Loss(beta=1.0)

        self.image_encoder = load_image_encoder(self.hparams.backbone_type, self.hparams.vit_backbone_size) # dtype 32 by default; also image_encoder here can refer to an entire VAE
        if self.hparams.backbone_type == "vae" and self.hparams.embedding_dim != 3136: # hardcoded original dim of vae from sd-vae
            self.encoder_down_projection = nn.Linear(3136, self.hparams.embedding_dim, bias=not self.hparams.weight_tie_vae_proj) # if are weight tying to make experiments equivalent then cannot use bias since would need different biases
            self.encoder_up_projection = nn.Linear(self.hparams.embedding_dim, 3136, bias=not self.hparams.weight_tie_vae_proj)
            init_whole_model_weights(self.encoder_down_projection, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
            init_whole_model_weights(self.encoder_up_projection, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
            self.encoder_dim = 3136
            if self.hparams.weight_tie_vae_proj:
                self.encoder_up_projection.weight = nn.Parameter(self.encoder_down_projection.weight.transpose(0, 1))

        else:
            self.encoder_down_projection = nn.Identity() # dont downproject, encoder dim = embed dim
            self.encoder_up_projection = nn.Identity() # dont upproject, encoder dim = embed dim
            self.encoder_dim = self.hparams.embedding_dim

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.transformer = setup_transformer(self.hparams)
        
        self.output = nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim, bias = False)
        init_whole_model_weights(self.output, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.finished_warming_up = False

    def forward(self, original_embeddings, start_pos = 0, learning = True): # accepts embeddings as input, predicts next embeddings. assumes embeddings is of shape BS, S, D
        # original here is to disdinguish between the original space of the encoder and the embedding space of the trans, these may be the same in some cases (see logic in init, sometimes i use nn.Identity)
        embeddings = self.encoder_down_projection(original_embeddings)
        predicted_embeddings = self.transformer(embeddings, start_pos = start_pos, learning = learning)
        predicted_embeddings = self.encoder_up_projection(self.output(predicted_embeddings))
        
        return predicted_embeddings
    
    def forward_loss_wrapper(self, x, phase="train"):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        x = x.reshape(-1, *x.shape[2:]) # BS*(S+1), C, W, H
        embeddings = get_encoded_images(x, self.hparams.backbone_type, self.image_encoder, sdxl_vae_standardization = self.hparams.sdxl_vae_standardization) 
        embeddings = embeddings.reshape(batch_size, seq_length, self.encoder_dim) # BS, S+1, D
        input_embeddings = embeddings[:, :-1] # only input first S; so BS, S, D

        predicted_embeddings = self(input_embeddings)

        next_embeddings = embeddings[:, 1:] # compare to last S (as make pred of next element); so BS, S, D
        loss = self.reconstruction_criterion(predicted_embeddings, next_embeddings) # predicted_embeddings has grad and is BS, S, D; next_embeddings is same shape and does not have grad

        log_dict = {
            'loss': loss,
        }
        return log_dict