import os
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler  
import h5py 
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Callable
import inspect
from ..model.utils.scaler import ScalerType



class MilanDataset(BaseDataset):
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
        self.num_cells = kwargs.get("num_cells", 100)
        assert self.num_cells ** 0.5 == int(self.num_cells ** 0.5), "num_cells must be a perfect square"
        self.data_key = kwargs.get("data_key", "call")
        self.mode = kwargs.get("mode", "train")
        self.train_ratio = kwargs.get("train_ratio", 0.7)
        self.test_ratio = kwargs.get("test_ratio", 0.2)
        assert self.train_ratio + self.test_ratio <= 1.0, "train_ratio and test_ratio must sum to less than or equal to 1.0"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.scaler_type = scaler_type
        self.post_init(processor)

    def _load_data(self):
        f = h5py.File(self.data_path, 'r')
        idx = f['idx'][()]
        cell = f['cell'][()]
        lng = f['lng'][()]
        lat = f['lat'][()]
        data = f[self.data_key][()][:, cell - 1]
        df = pd.DataFrame(data, index=pd.to_datetime(idx.ravel(), unit='s'), columns=cell)
        selected_cells = sorted(np.random.choice(list(cell), self.num_cells, replace=False))
        df = df[selected_cells]
        selected_cells_idx = np.where(np.isin(list(cell), selected_cells))
        cell_lng = lng[selected_cells_idx]
        cell_lat = lat[selected_cells_idx]
        df.fillna(0, inplace=True)
        return df, cell_lng, cell_lat
        

    def post_init(self, processor):
        timeseries, cell_lng, cell_lat = self._load_data()
        all_length = len(timeseries)
        num_train = int(all_length * self.train_ratio)
        num_test = int(all_length * self.test_ratio)
        num_eval = all_length - num_train - num_test
        low_idx_list = [0, num_train - self.seq_len, all_length - num_test - self.seq_len]
        up_idx_list = [num_train, num_train + num_eval, all_length]
        low_idx = low_idx_list[["train", "test", "eval"].index(self.mode)]
        up_idx = up_idx_list[["train", "test", "eval"].index(self.mode)]
        train_timeseries = timeseries.iloc[low_idx_list[0]:up_idx_list[0]]

        if processor is not None:
            if inspect.isclass(processor):
                self.processor = processor(
                    scale=self.scale,
                    scaler_type=self.scaler_type,
                )
                self.processor.fit(train_timeseries.values)
            else:
                self.processor = processor
        else:
            self.processor = None

        self.timeseries = timeseries.iloc[low_idx:up_idx]
        self.other_features = {
            "cell_lng": cell_lng,
            "cell_lat": cell_lat,
        }
        self.timestamp = timeseries.index[low_idx:up_idx]       


        
 
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
            return {key: value for key, value in data.items()}
        else:
            return dict(
                timeseries=torch.tensor(inputs.values, dtype=torch.float),
                labels=torch.tensor(labels.values, dtype=torch.float),
            )

if __name__ == "__main__":
    from src.model.ST_Linear.processing_ST_Linear import ST_LinearProcessor
    dataset = MilanDataset(
        datasets="Milan",
        data_path="data/milano.h5",
        num_cells=100,
        data_key="call",
        seq_len=432,      
        pred_len=144,
        processor=ST_LinearProcessor,
    )
    instance = dataset[0]