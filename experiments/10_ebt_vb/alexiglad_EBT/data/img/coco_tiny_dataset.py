from datasets import load_dataset
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image

clip_model_name_mapping = {
    "small": "openai/clip-vit-base-patch32",
    "base": "openai/clip-vit-base-patch16",
    "large": "openai/clip-vit-large-patch14",
    "xl": "openai/clip-vit-large-patch14-336"
}

class COCOTinyDataset(Dataset):
    def __init__(self, hparams, split, transform): 
        self.hparams = hparams 
        hf_token = os.getenv('HF_TOKEN')
        hf_home = os.getenv('HF_HOME')
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
        self.dataset = load_dataset("howard-hou/COCO-Text", cache_dir=dataset_dir, token=hf_token)[split]
        self.transform = transform
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name_mapping[hparams.clip_text_encoder_size]) # assumes using clip tokenizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        caption = self.dataset[idx]['caption'][0] # just get first caption

        if image.mode == 'L':
            image = Image.fromarray(np.repeat(np.array(image)[:, :, np.newaxis], 3, axis=2))

        image = self.transform(image)
        caption = self.tokenizer(caption, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt") # has keys ['input_ids', 'attention_mask']
        caption = {k: v.squeeze(0) for k, v in caption.items()}
            
        return {"image": image, "caption": caption}



