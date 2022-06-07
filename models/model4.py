from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from .model import _weights_init



class Net4(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.net_params = net_params
        self.hidden_size = net_params.hidden_size
        self.embed_size = net_params.embed_size

        self.seq_emb = nn.Sequential(
            nn.Linear(len(net_params.cont_seq_cols), self.embed_size),
            nn.LayerNorm(self.embed_size),
            nn.ReLU(),

        )

        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers=net_params.num_layers,
                            batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size ),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.net_params.num_classes),
        )

        self.reg = True if self.net_params.num_classes == 1 else False

        self.apply(_weights_init) 

    def forward(self, cate_seq_x, cont_seq_x):
        bs = cont_seq_x.size(0)

        seq_emb = self.seq_emb(cont_seq_x)
        seq_emb, _ = self.lstm(seq_emb)

        if self.reg:
            output = self.head(seq_emb).view(bs, -1)
        else:
            output = self.head(seq_emb)

        return output
