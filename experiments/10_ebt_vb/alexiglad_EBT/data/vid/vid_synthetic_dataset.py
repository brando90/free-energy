import torch
from torch.utils.data import Dataset


class VIDSyntheticDataset(Dataset):
    def __init__(self, hparams, size=10000000):
        self.size = size
        self.context_length = hparams.context_length
        self.image_dims = hparams.image_dims
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        random_embeddings = torch.randn(self.context_length, 3, self.image_dims[0], self.image_dims[1])
        return random_embeddings