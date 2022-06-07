import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from models import *
from optimizer import get_optimizers


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def metric(preds, y, u_out):
    """
    Metric for the problem, as I understood it.
    """
    mask = 1 - u_out
    mae = mask * np.abs(y - preds)
    mae = mae.sum() / mask.sum()
    
    return mae


class VentilatorLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.loss_w = weights

    def forward(self, preds, y, u_out):

        mask = u_out
        mask[u_out == 1] = self.loss_w[0]
        mask[u_out == 0] = self.loss_w[1]
       

        mae = torch.abs(mask * (y - preds))
        mae = torch.sum(mae)

        return mae


def cls_loss_fn( y_pred, y_true):
    loss = nn.CrossEntropyLoss()(y_pred.reshape(-1, 950), y_true.reshape(-1))
    return loss


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


class ReguressionModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = globals()[self.config.Model.cls](config.Model.params)
        if config.globals.load is not None:
            print("load: ", config.globals.load)
            self.model = load_pytorch_model(config.globals.load, self.model)

        if self.config.loss.type == "VentilatorLoss":
            self.loss = VentilatorLoss(self.config.loss.weights)
        elif self.config.loss.type == "L1Loss":
            self.loss = torch.nn.L1Loss()
        elif self.config.loss.type == "HuberLoss":
            self.loss = torch.nn.HuberLoss()


    def training_step(self, batch, batch_idx):
        loss_mask = batch["u_out"] == 0

        x1 = batch["cate_seq_x"]
        x2 = batch["cont_seq_x"]

        logits = self.model(x1, x2)

        if self.config.loss.type != "VentilatorLoss":
            loss = 2. * self.loss(logits[loss_mask], batch["targets"][loss_mask]) + self.loss(logits[loss_mask == 0], batch["targets"][loss_mask == 0])
            loss = loss / 2
        else:
            loss = self.loss(
                logits,
                batch["targets"],
                batch["u_out"]
            ).mean()

        return loss

    def validation_step(self, batch, batch_idx):
        loss_mask = batch["u_out"] == 0

        x1 = batch["cate_seq_x"]
        x2 = batch["cont_seq_x"]

        logits = self.model(x1, x2)

        if self.config.loss.type != "VentilatorLoss":
            loss = 2. * self.loss(logits[loss_mask], batch["targets"][loss_mask]) + self.loss(logits[loss_mask == 0], batch["targets"][loss_mask == 0])
            loss = loss / 2
        else:
            loss = self.loss(
                logits,
                batch["targets"],
                batch["u_out"]
            ).mean()

        output = OrderedDict({
            "targets": batch["targets"].detach().cpu().numpy(), 
            "preds": logits.detach().cpu().numpy(), 
            "u_out": batch["u_out"].detach().cpu().numpy(),
            "loss": loss.detach()
        })
        return output


    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["val_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = np.concatenate([o["targets"] for o in outputs])
        preds = np.concatenate([o["preds"] for o in outputs])
        u_out = np.concatenate([o["u_out"] for o in outputs])

        score = metric(preds, targets, u_out)

        d["val_score"] = score
        self.log_dict(d, prog_bar=True)


    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(
            model=self.model, 
            config=self.config,
        )
        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step', # or 'epoch'
                          'frequency': 1}

        return [optimizer], [scheduler_dict]


class ClassificationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = globals()[self.config.Model.cls](config.Model.params)
        if config.globals.load is not None:
            print("load: ", config.globals.load)
            self.model = load_pytorch_model(config.globals.load, self.model)

        self.loss = cls_loss_fn
        self.target_dic_inv = load_dict("./data/target_dic_inv.pkl")
        self.target_dic = load_dict("./data/target_dic.pkl")

    def training_step(self, batch, batch_idx):
        
        x1 = batch["cate_seq_x"]
        x2 = batch["cont_seq_x"]

        
        logits = self.model(x1, x2)

        loss = self.loss(logits, batch["targets"])

        return loss

    def validation_step(self, batch, batch_idx):
        x1 = batch["cate_seq_x"]
        x2 = batch["cont_seq_x"]

        predicts = []
        logits = self.model(x1, x2)
        out = torch.tensor([[self.target_dic_inv[j.item()] for j in i] for i in logits.argmax(2)])
        rev_trgets = torch.tensor([[self.target_dic_inv[j.item()] for j in i] for i in batch["targets"]])
        loss = self.loss(logits, batch["targets"])

        predicts.append(out)
        output = OrderedDict({
            "targets": rev_trgets.detach().cpu().numpy(), 
            "preds": out.detach().cpu().numpy(), 
            "u_out": batch["u_out"].detach().cpu().numpy(),
            "loss": loss.detach()
        })
        return output


    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["val_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = np.concatenate([o["targets"] for o in outputs])
        preds = np.concatenate([o["preds"] for o in outputs])
        u_out = np.concatenate([o["u_out"] for o in outputs])

        score = metric(preds, targets, u_out)

        d["val_score"] = score
        self.log_dict(d, prog_bar=True)


    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(
            model=self.model, 
            config=self.config,
        )
        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step', # or 'epoch'
                          'frequency': 1}

        return [optimizer], [scheduler_dict]
