
   
import math

import torch
from timm.optim.lookahead import Lookahead
from torch import optim as optim
# Third party libraries
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from warmup_scheduler import GradualWarmupScheduler
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from madgrad import MADGRAD

class WarmRestart(lr_scheduler.CosineAnnealingLR):
    """This class implements Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.
    Set the learning rate of each parameter group using a cosine annealing schedule,
    When last_epoch=-1, sets initial lr as lr.
    This can't support scheduler.step(epoch). please keep epoch=None.
    """

    def __init__(self, optimizer, T_max=10, T_mult=2, eta_min=0, last_epoch=-1):
        """implements SGDR
        Parameters:
        ----------
        T_max : int
            Maximum number of epochs.
        T_mult : int
            Multiplicative factor of T_max.
        eta_min : int
            Minimum learning rate. Default: 0.
        last_epoch : int
            The index of last epoch. Default: -1.
        """
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


def warm_restart(scheduler, T_mult=2):
    """warm restart policy
    Parameters:
    ----------
    T_mult: int
        default is 2, Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.
    Examples:
    --------
    >>> # some other operations(note the order of operations)
    >>> scheduler.step()
    >>> scheduler = warm_restart(scheduler, T_mult=2)
    >>> optimizer.step()
    """
    if scheduler.last_epoch == scheduler.T_max:
        scheduler.last_epoch = -1
        scheduler.T_max *= T_mult
    return scheduler


def get_optimizers(model, config):
    opt_lower = config.Optimizer.type.lower()
    param_lists = []
    for name, param in model.named_parameters():
        if "cqt_kernels" in name:
            param_lists.append({"params": param, "lr": 3e-6})
        else:
            param_lists.append({"params": param, "lr": config.Optimizer.params.lr})

    opt_look_ahed = config.Optimizer.lookahead
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=config.Optimizer.params.lr, momentum=config.Optimizer.params.momentum, weight_decay=config.Optimizer.params.weight_decay, nesterov=True)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config.Optimizer.params.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.Optimizer.params.weight_decay)
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.Optimizer.params.lr, weight_decay=config.Optimizer.params.weight_decay, eps=config.Optimizer.params.opt_eps)
    elif opt_lower == 'madgrad':
        optimizer = MADGRAD(model.parameters(), lr=config.Optimizer.params.lr, weight_decay=config.Optimizer.params.weight_decay)
    else:
        assert False and "Invalid optimizer"
        raise ValueError


    if opt_look_ahed:
        optimizer = Lookahead(optimizer, alpha=0.5, k=5)

    if config.Scheduler.type == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            eta_min=config.Scheduler.eta_min,
            T_0=config.Scheduler.T_0,
            T_mult=config.Scheduler.T_multi,
            last_epoch=-1
            )

    elif config.Scheduler.type == "WarmRestart":
        scheduler = WarmRestart(optimizer, T_max=config.Scheduler.T_max, T_mult=config.Scheduler.T_multi, eta_min=config.Scheduler.eta_min)
    elif config.Scheduler.type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.globals.max_epoch,
            eta_min=config.Scheduler.params.min_lr,
            last_epoch=-1
        )

    elif config.Scheduler.type == "LinearWarmupCosineAnnealingLR":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=config.Scheduler.params.warmup_epochs, 
            max_epochs=config.globals.max_epoch,
            eta_min=config.Scheduler.params.min_lr,
            warmup_start_lr= config.Scheduler.params.warmup_start_lr,
            last_epoch=-1
        )

    elif config.Scheduler.type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min",
            factor=0.1,
            patience=15,
            min_lr=1e-6
        )
    elif config.Scheduler.type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.Scheduler.params.gamma)
    else:
        raise Exception(f"Not Implimented: {config.Scheduler.type}")

    return optimizer, scheduler
    