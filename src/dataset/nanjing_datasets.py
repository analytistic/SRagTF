import os
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  
import json
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Callable
import inspect
from datasets import load_dataset




class NanJingDataset(BaseDataset):
    def __init__(
        self,
        datasets: str,
        data_path: str,
        seq_len: int = 24,      
        pred_len: int = 12,
        scale: bool = True,
        processor: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            datasets=datasets,
            data_path=data_path,
            **kwargs
        )
        self.mode = kwargs.get("mode", "train")
        self.post_init(processor)
        
    def _load_data(self):
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data

        
        

    def post_init(self, processor):
        data = self._load_data()

        

        
        low_idx = low_idx_list[["train", "test", "eval"].index(self.mode)]
        up_idx = up_idx_list[["train", "test", "eval"].index(self.mode)]
        train_timeseries = timeseries.iloc[low_idx_list[0]:up_idx_list[0]]
        scaler = None
        if self.scale:
            scaler = StandardScaler()
            scaler.fit(train_timeseries.values)
        self.timeseries = timeseries.iloc[low_idx:up_idx]
        self.other_features = {
            "cell_lng": cell_lng,
            "cell_lat": cell_lat,
        }
        self.timestamp = timeseries.index[low_idx:up_idx]      
        if processor is not None:
            if inspect.isclass(processor):
                self.processor = processor(
                    scale=self.scale,
                    transform=scaler,
                )
            else:
                self.processor = processor
        else:
            self.processor = None
        
 
    def __len__(self):
        return len(self.timeseries) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        inputs = self.timeseries.iloc[index:index+self.seq_len]
        labels = self.timeseries.iloc[index+self.seq_len:index+self.seq_len+self.pred_len]
        if self.processor is not None:
            data = self.processor(
                timeseries=inputs,
                labels=labels,
                scale=self.scale,
                return_tensors="pt"
            )
            return dict(
                timeseries=data["timeseries"],
                labels=data["labels"],
            )
        else:
            return dict(
                timeseries=torch.tensor(inputs.values, dtype=torch.float),
                labels=torch.tensor(labels.values, dtype=torch.float),
            )

if __name__ == "__main__":
    from src.model.ST_Linear.processing_ST_Linear import ST_LinearProcessor
    dataset = NanJingDataset(
        datasets="NanJing",
        data_path="data/nanjing/24_TrafficNC_short64.json",
        num_cells=100,
        data_key="call",
        seq_len=64,      
        pred_len=64,
        processor=ST_LinearProcessor,
    )
    instance = dataset[0]