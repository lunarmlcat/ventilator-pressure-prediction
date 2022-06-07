from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable


class Net11(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.net_params = net_params
        self.hidden_size = net_params.hidden_size
        self.embed_size = net_params.embed_size
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)
        self.rc_emb = nn.Embedding(9, 4, padding_idx=0)

        self.seq_emb = nn.Sequential(
            nn.Linear(8 + len(net_params.cont_seq_cols), 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )


        self.reg = True if self.net_params.num_classes == 1 else False


    def forward(self, cate_seq_x, cont_seq_x):
        bs = cont_seq_x.size(0)
        r_emb = self.r_emb(cate_seq_x[:,:,0].long()).view(bs, 80, -1)
        c_emb = self.c_emb(cate_seq_x[:,:,1].long()).view(bs, 80, -1)
        rc_emb = self.rc_emb(cate_seq_x[:,:,2].long()).view(bs, 80, -1)

        seq_x = torch.cat((r_emb, c_emb, rc_emb, cont_seq_x), 2)
        seq_emb = self.seq_emb(seq_x)

        if self.reg:
            output = seq_emb.view(bs, -1)
        else:
            output = seq_emb

        return output