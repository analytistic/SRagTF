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
from ..model.utils.scaler import ScalerType




class MobileTrafficDataset(BaseDataset):
    def __init__(
        self,
        datasets: str, 
        data_path: str,
        seq_len: int = 24,      
        pred_len: int = 12,
        scale: bool = True,
        scaler_type: ScalerType | None = None,
        processor: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            datasets=datasets,
            data_path=data_path,
            **kwargs
        )
        self.mode = kwargs.get("mode", "train")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.scaler_type = scaler_type
        self.post_init(processor)
        
    def _load_data(self):
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data

        
        

    def post_init(self, processor):
        data = self._load_data()


        data_dict = {
            "train": ('X_train','train'),
            "eval": ('X_val','val'),
            "test": ('X_test','test'),
        }


        train_timeseries = np.array(data[data_dict["train"][0]][0])
        clip_datas_train = np.percentile(train_timeseries, 99.99)
        train_timeseries = np.clip(train_timeseries, 0, clip_datas_train) / clip_datas_train

        if processor is not None:
            if inspect.isclass(processor):
                self.processor = processor(
                    scale=self.scale,
                    scaler_type=self.scaler_type,
                )
                self.processor.fit(train_timeseries.reshape(-1, 1))
            else:
                self.processor = processor
        else:
            self.processor = None

        
        self.timeseries = np.array(data[data_dict[self.mode][0]][0])
        self.timeseries = np.clip(self.timeseries, 0, clip_datas_train) / clip_datas_train
        self.timeseries = self.timeseries.reshape(self.timeseries.shape[0], self.timeseries.shape[1], self.timeseries.shape[2]*self.timeseries.shape[3])
        self.timestamp = np.array(data['timestamps'][data_dict[self.mode][1]])

        
 
    def __len__(self):
        return len(self.timeseries)

    def __getitem__(self, index):
        inputs = self.timeseries[index][:self.seq_len]
        labels = self.timeseries[index][self.seq_len:self.seq_len+self.pred_len]
        if self.processor is not None:
            data = self.processor(
                timeseries=inputs.reshape(-1, 1),
                labels=labels,
                scale=self.scale,
                return_tensors="pt"
            )
            return dict(
                timeseries=data['timeseries'].reshape(inputs.shape),
                labels=data["labels"].reshape(labels.shape),
            )
        else:
            return dict(
                timeseries=torch.tensor(inputs, dtype=torch.float),
                labels=torch.tensor(labels, dtype=torch.float),
            )
