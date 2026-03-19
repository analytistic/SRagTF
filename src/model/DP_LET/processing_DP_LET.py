from transformers import FeatureExtractionMixin
from typing import Optional, Tuple, Dict, Any, List, Callable
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers.feature_extraction_utils import BatchFeature
from pandas import DataFrame
import torch

class DP_LETProcessor(FeatureExtractionMixin):
    def __init__(self, 
                 transform: None,
                 scale: bool,
                 mean: np.ndarray | List | float | None = None, 
                 std: np.ndarray | List | float | None = None, 
                 var: np.ndarray | List | float | None = None
                 ):

        if transform is not None:
            self.transform = transform
            mean = np.array([getattr(transform, "mean_", 0.0)])
            std = np.array([getattr(transform, "scale_", 1.0)])
            var = np.array([getattr(transform, "var_", 1.0)])
        else:
            mean = np.array(mean) if mean is not None else np.array([0.0])
            std = np.array(std) if std is not None else np.array([1.0])
            var = np.array(var) if var is not None else np.array([1.0])
            self.transform = StandardScaler()
            self.transform.mean_ = mean
            self.transform.scale_ = std
            self.transform.var_ = var
        super().__init__(scale=scale, mean=mean, std=std, var=var)
        
        self.scale = scale
    
    def __call__(self, 
                 timeseries: DataFrame | np.ndarray | List, 
                 ex_features: DataFrame | np.ndarray | List | None = None,
                 labels: DataFrame | np.ndarray | List | None = None,
                 scale: Optional[bool] = None, 
                 return_tensors: str = 'pt', 
                 **kwargs) -> BatchFeature:
        outputs = {}
        scale = scale if scale is not None else self.scale
        
        timestamp = timeseries.index if isinstance(timeseries, DataFrame) else None
        timeseries = timeseries.values if isinstance(timeseries, DataFrame) else np.array(timeseries)
        timeseries = self.transform.transform(timeseries) if scale else timeseries
        labels = labels.values if isinstance(labels, DataFrame) else np.array(labels) if labels is not None else None
        labels = self.transform.transform(labels) if scale and labels is not None  else labels
        outputs["timeseries"] = timeseries
        outputs["timestamp"] = timestamp
        outputs["labels"] = labels

        return BatchFeature(
            data=outputs,
            tensor_type=return_tensors,
        )
    
    def to_dict(self):
        output = super().to_dict()
        output.pop("transform", None)

        for key in ["mean", "std", "var"]:
            if isinstance(output.get(key), np.ndarray):
                output[key] = output[key].tolist()
        return output