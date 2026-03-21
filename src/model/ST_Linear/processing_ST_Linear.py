from transformers import FeatureExtractionMixin
from typing import Optional, Tuple, Dict, Any, List, Callable
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers.feature_extraction_utils import BatchFeature
from pandas import DataFrame
import torch
from ..utils.scaler import ScalerType, StandaryScaler, MinMaxScaler, BaseScaler



class ST_LinearProcessor(FeatureExtractionMixin):
    def __init__(self, 
                 scale: bool,
                 scaler_type: Optional[ScalerType] = None,
                 **kwargs
                 ):
        super().__init__(scale=scale, scaler_type=scaler_type, **kwargs)
        
        self.scale = scale
        self.scaler = self._build_scaler(scaler_type, **kwargs) if scale else None

    
    def _build_scaler(self, scaler_type: ScalerType | None, **kwargs) -> BaseScaler:
        if scaler_type == ScalerType.STANDARY:
            return StandaryScaler(**kwargs)
        elif scaler_type == ScalerType.MINMAX:
            return MinMaxScaler(**kwargs)
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
    def fit(self, inputs: np.ndarray):
        if self.scaler is not None:
            self.scaler.fit(inputs)
        else:
            raise ValueError("No scaler defined to fit the data.")
    
    def transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(inputs)
        else:
            raise ValueError("No scaler defined to transform the data.")
        
    def inverse_transform(self, inputs: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.inverse_transform(inputs)
        else:
            raise ValueError("No scaler defined to inverse transform the data.")

    
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
        labels = labels.values if isinstance(labels, DataFrame) else np.array(labels) if labels is not None else None
        if self.scaler is not None and self.scale:
            timeseries = self.scaler.transform(timeseries) if scale else timeseries
            labels = self.scaler.transform(labels) if scale and labels is not None else labels

        outputs["timeseries"] = timeseries
        outputs["timestamp"] = timestamp
        outputs["labels"] = labels

        return BatchFeature(
            data=outputs,
            tensor_type=return_tensors,
        )
    
    def to_dict(self):
        output = super().to_dict()
        scaler = output.pop("scaler", None)
        if scaler is not None:
            for key, value in scaler.__dict__.items():
                if isinstance(value, np.ndarray):
                    output[key] = value.tolist()
                elif isinstance(value, (list, int, float, str, bool, type(None))):
                    output[key] = value
        return output