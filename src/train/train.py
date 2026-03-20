from src.train.trainer.base_trainer import BaseTrainer
from src.train.utils.arguments import TrainingArguments, ModelArguments, DataArguments
from src.train.metrics.eval import MetricsComputer
from src.data.milan_datasets import MilanDataset
from src.data import BaseCollator
from src.model import *

import argparse
from transformers import set_seed
import torch
from transformers.utils import logging
logger = logging.get_logger(__name__)

model_dict = {
    'ST_Linear': (ST_LinearForTrafficPrediction, ST_LinearConfig, ST_LinearProcessor),
    'DP_LET': (DP_LET_Predictor, DP_LETConfig, DP_LETProcessor),
    'iTransformer': (iTransformer, iTransformerConfig, iTransformerProcessor),
    'PatchTST': (PatchTST, PatchTSTConfig, PatchTSTProcessor),
    'TimeMixer': (TimeMixer, TimeMixerConfig, TimeMixerProcessor)
}



def train(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Train a model.")
        parser.add_argument("--config_path", type=str, help="Path to the config file (TOML format).")
        args = parser.parse_args()
        config_path = args.config_path

    # Load arguments from the config file
    data_args = DataArguments.from_toml(config_path)
    model_args = ModelArguments.from_toml(config_path)
    training_args = TrainingArguments.from_toml(config_path)
    set_seed(training_args.seed)


    # Initialize dataset and trainer
    train_dataset = MilanDataset(
        data_path=data_args.data_path,
        datasets=data_args.datasets,
        seq_len=model_args.config.get("seq_len", 512),
        pred_len=model_args.config.get("pred_len", 128),
        scale=data_args.scale,
        num_cells=getattr(data_args, "num_cells", 100),
        processor=model_dict[model_args.pretrained_model_name_or_path][2],
        data_key=getattr(data_args, "data_key", "call"),
        mode="train",
    )
    eval_dataset = MilanDataset(
        data_path=data_args.data_path,
        datasets=data_args.datasets,
        seq_len=model_args.config.get("seq_len", 512),
        pred_len=model_args.config.get("pred_len", 128),
        scale=data_args.scale,
        num_cells=getattr(data_args, "num_cells", 100),
        processor=model_dict[model_args.pretrained_model_name_or_path][2],
        data_key=getattr(data_args, "data_key", "call"),
        mode="eval",
    )
    test_dataset = MilanDataset(
        data_path=data_args.data_path,
        datasets=data_args.datasets,
        seq_len=model_args.config.get("seq_len", 512),
        pred_len=model_args.config.get("pred_len", 128),
        scale=data_args.scale,
        num_cells=getattr(data_args, "num_cells", 100),
        processor=model_dict[model_args.pretrained_model_name_or_path][2],
        data_key=getattr(data_args, "data_key", "call"),
        mode="test",
    )
    config = model_dict[model_args.pretrained_model_name_or_path][1].from_dict(model_args.config)
    model = model_dict[model_args.pretrained_model_name_or_path][0](config)
    model.to(dtype=torch.float32)

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=BaseCollator(),
        compute_metrics=MetricsComputer(train_dataset.processor),
        processing_class=train_dataset.processor
    )

    trainer.train()

if __name__ == "__main__":
    config_path = '/Users/alex/project/SR_Linear/src/train/config/ST_Linear.toml'

    train(config_path)
