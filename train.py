import argparse
import importlib
import os
import random
import sys
import torch

import numpy as np
import pandas as pd
import yaml
from addict import Dict
from pandas.core.reshape.concat import concat
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.plugins import DDPPlugin, NativeMixedPrecisionPlugin

import wandb
from dataset import DataModule
from trainer import ReguressionModule, ClassificationModule
import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(args):
    seed_everything(2021)
    with open(f"configs/{args.config}.yml", "r+") as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    config = Dict(yml)

    if args.fold_num is not None:
        config.globals.fold_num = args.fold_num
    
    print(config.globals.fold_num)
    name = f"{args.config}/fold{config.globals.fold_num}"

    logger = CSVLogger(save_dir=f"./result/{args.config}", name=f"fold{config.globals.fold_num}")
    wandb.init(
        dir="../dataset/kaggle_ventilator",
        group=f"{args.config}",
        name=name, 
        project='kaggle_ventilator',
    )

    wandb_logger = WandbLogger(
        group=f"{args.config}",
        name=name, 
        project='kaggle_ventilator',
        entity='ikki1111'
    )

    wandb.save("train.py")
    wandb.save("trainer.py")
    wandb.save("models/model.py")
    wandb.save("dataset.py")
    wandb.save(f"configs/{args.config}.yml")


    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./result/{name}",
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.3f}-{val_score:.3f}',
        mode='min',
        save_top_k=5,
        save_weights_only=True,
    )

    swa_callback = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=1e-6)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode='min'
    )

    gpus = 1 if len(config.globals.gpus) == 1 else -1

    trainer = Trainer(
        max_epochs=config.globals.max_epoch,
        gpus=[0],
        auto_select_gpus=False,
        progress_bar_refresh_rate=10,
        gradient_clip_val=1000,
        num_sanity_val_steps=2,
        amp_backend="native",
        amp_level="02",
        precision=16 if config.globals.enable_amp else 32,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        stochastic_weight_avg=config.globals.swa,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=[logger, wandb_logger],
    )

    if config.globals.task == "classification":
        print("CLS Trainer")
        model = ClassificationModule(config)
    else:
        print("Regression Trainer")
        model = ReguressionModule(config)

        
    datamodule = DataModule(config)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True
    )

    parser.add_argument(
        "--fold_num",
        "-fn",
        type=int,
        default=None
    )
    args = parser.parse_args()

    # os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    main(args)