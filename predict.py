import argparse
import gc
import os
import warnings

import numpy as np
import pandas as pd
import torch.nn as nn
import yaml
from addict import Dict
from pytorch_lightning import seed_everything
from torch.utils import data
from tqdm import tqdm

from dataset import *
from dataset import DataModule
from models import *

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

device = torch.device("cuda")
BATCH_SIZE = 128

def metric(preds, y, u_out):
    """
    Metric for the problem, as I understood it.
    """
    mask = 1 - u_out
    mae = mask * np.abs(y - preds)
    mae = mae.sum() / mask.sum()
    
    return mae


class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric
    """
    def __call__(self, preds, y, u_out):
        mask = 1 - u_out

        mae = torch.abs(mask * (y - preds))
        mae = torch.sum(mae) / torch.sum(mask)

        return mae


def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    return model


def prediction(model, loader, device, mode="val"):
    model = model.cuda()
    model.eval()

    preds = []
    loss_fn = VentilatorLoss()
    metric1 = []
    metric2 = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            cate_seq_x = batch["cate_seq_x"].cuda()
            cont_seq_x = batch["cont_seq_x"].cuda()
  
   
            logits = model(cate_seq_x, cont_seq_x)
            if mode == "val":
                metric1.append(loss_fn(logits.detach().cpu(), batch["targets"], batch["u_out"]).mean().detach())

            y_pred = logits.view(-1).detach().cpu().numpy()
            preds.append(y_pred)

    preds = np.concatenate(preds)
    if mode != "val":
        return preds
    else:
        return np.array(preds), np.mean(metric1)


def main(args):
    seed_everything(2021)
    with open(f"configs/{args.config}.yml", "r+") as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    config = Dict(yml)

    print(config.pred.path)
    predicts = []
    train_ids = []
    train_oof = []

    for i in range(len(config.pred.path)):
        config.globals.fold_num = i
        print("fold: ",config.globals.fold_num)
        datamodule = DataModule(config)
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        val_df = datamodule.get_datagrame("valid")
        test_df = datamodule.get_datagrame("test")

        print("load")
        model = globals()[config.Model.cls](config.Model.params)
        model = load_pytorch_model(config.pred.path[i], model)

        test_preds = prediction(model, test_loader, device, "test")
        val_preds, score = prediction(model, val_loader, device, "val")

        print(f"Fold_{config.globals.fold_num}: ", score)

        train_ids.extend(val_df["id"].values)
        train_oof.extend(val_preds)
        
        predicts.append(test_preds)

        del datamodule, model
        gc.collect()

    
    result = pd.DataFrame({
        "id": test_df["id"].values,
        "pressure": np.asarray(predicts).mean(axis=0).flatten()
    })
    result.to_csv(f"./result/sub/submission_{args.config}_mean.csv", index=None)

    result = pd.DataFrame({
        "id": test_df["id"].values,
        "pressure": np.asarray(predicts).mean(axis=0).flatten()
    })
    result.to_csv(f"./result/sub/submission_{args.config}_median.csv", index=None)

    result_oof = pd.DataFrame({
        "id": train_ids,
        "pressure": train_oof
    })
    result_oof.to_csv(f"./result/sub/{args.config}_oof.csv", index=None)

    # pp
    train_df = pd.read_csv("./data/train.csv")
    unique_pressures = train_df["pressure"].unique()
    sorted_pressures = np.sort(unique_pressures)
    total_pressures_len = len(sorted_pressures)

    PRESSURE_MIN = -1.895744294564641
    PRESSURE_MAX = 64.8209917386395
    PRESSURE_STEP = 0.07030214545121005
    
    sub_df = pd.read_csv(f"./result/sub/submission_{args.config}_mean.csv")
    sub_df["pressure"] = np.round((sub_df["pressure"] - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
    sub_df.pressure = np.clip(sub_df.pressure, PRESSURE_MIN, PRESSURE_MAX)
    sub_df.to_csv(f"./result/sub/submission_{args.config}_mean_pp.csv", index=None)
    
    sub_df = pd.read_csv(f"./result/sub/submission_{args.config}_median.csv")
    sub_df["pressure"] = np.round((sub_df["pressure"] - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
    sub_df.pressure = np.clip(sub_df.pressure, PRESSURE_MIN, PRESSURE_MAX)
    sub_df.to_csv(f"./result/sub/submission_{args.config}_median_pp.csv", index=None)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True
    )

    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
        default='0'
    )
    parser.add_argument(
        "--valid",
        "-v",
        action='store_true'
    )
    parser.add_argument(
        "--proba",
        action='store_true'
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

    main(args)
