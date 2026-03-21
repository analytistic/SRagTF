import torch
class BaseCollator:
    def __call__(self, batch):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch], dim=0).to(dtype=torch.float32) for key in keys}
    
from .base_dataset import BaseDataset
from .milan_datasets import MilanDataset
from .mobiletraffic_datasets import MobileTrafficDataset

__all__ = [
    "BaseCollator",
    "BaseDataset",
    "MilanDataset",
    "MobileTrafficDataset",
]