import numpy as np
from transformers import EvalPrediction
from transformers import FeatureExtractionMixin
import torch



def compute_mae(preds, labels):
    return np.mean(np.abs(preds - labels))

def compute_mse(preds, labels):
    return np.mean((preds - labels) ** 2)

def compute_rmse(preds, labels):
    return np.sqrt(compute_mse(preds, labels))

def compute_r2(preds, labels):
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    return 1 - (ss_res / ss_tot)

def compute_mase(preds, labels, w=1):
    n = len(labels)
    mae = compute_mae(preds, labels)
    naive_mae = np.sum(np.abs(labels[w:] - labels[:-w])) * (n / w)
    return mae / (naive_mae + np.finfo(naive_mae.dtype).tiny)

def compute_mape(preds, labels):
    return np.mean(np.abs((labels - preds) / (labels + np.finfo(labels.dtype).tiny)))

def compute_smape(preds, labels):
    return np.mean(np.abs(labels - preds) / np.sum(np.abs(labels) + np.abs(preds)) * 2)

def compute_rmsle(preds, labels):
    # RMSLE clip
    preds = np.clip(preds, 0.0, None)
    labels = np.clip(labels, 0.0, None)
    return np.sqrt(np.mean((np.log1p(preds) - np.log1p(labels)) ** 2))

def compute_owa(preds, labels, w=1):
    pass




class MetricsComputer:
    def __init__(self, processor: FeatureExtractionMixin | None, period: int=1):
        self.processor = processor
        self.period = period

    def __call__(self, eval_pred: EvalPrediction):
        outputs = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        results = {
            "mae": compute_mae(outputs, labels),
            "mse": compute_mse(outputs, labels),
            "rmse": compute_rmse(outputs, labels),
            "r2": compute_r2(outputs, labels),
            # "mase": compute_mase(outputs, labels, w=self.period),
            "mape": compute_mape(outputs, labels),
            "smape": compute_smape(outputs, labels),
            "rmsle": compute_rmsle(outputs, labels)
        }
        return results