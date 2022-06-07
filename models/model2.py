from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from .model import _weights_init


class Net2(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.net_params = net_params
        self.hidden_size = net_params.hidden_size
        self.embed_size = net_params.embed_size
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)
        self.rc_dot_emb = nn.Embedding(8, 4, padding_idx=0)
        self.rc_sum_emb = nn.Embedding(8, 4, padding_idx=0)

        self.u_out_emb = nn.Embedding(2, 2, padding_idx=0)
        self.u_out_l1_emb = nn.Embedding(2, 2, padding_idx=0)
        self.u_out_l2_emb = nn.Embedding(2, 2, padding_idx=0)
        self.u_out_l3_emb = nn.Embedding(2, 2, padding_idx=0)
        self.u_out_l4_emb = nn.Embedding(2, 2, padding_idx=0)


        self.seq_emb = nn.Sequential(
            nn.Linear(22 + len(net_params.cont_seq_cols), self.embed_size),
            nn.LayerNorm(self.embed_size),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size*2),
            nn.LayerNorm(self.embed_size*2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.embed_size*2, self.hidden_size, num_layers=4,
                            batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.net_params.num_classes),
        )

        self.reg = True if self.net_params.num_classes == 1 else False

        self.apply(_weights_init) 

    def forward(self, cate_seq_x, cont_seq_x):
        bs = cont_seq_x.size(0)
        r_emb = self.r_emb(cate_seq_x[:,:,0].long()).view(bs, 80, -1)
        c_emb = self.c_emb(cate_seq_x[:,:,1].long()).view(bs, 80, -1)
        rc_dot_emb = self.rc_dot_emb(cate_seq_x[:,:,2].long()).view(bs, 80, -1)
        rc_sum_emb = self.rc_sum_emb(cate_seq_x[:,:,3].long()).view(bs, 80, -1)

        uo_emb = self.u_out_emb(cate_seq_x[:,:,4]).view(bs, 80, -1)
        ul1_emb = self.u_out_l1_emb(cate_seq_x[:,:,5]).view(bs, 80, -1)
        ul2_emb = self.u_out_l2_emb(cate_seq_x[:,:,6]).view(bs, 80, -1)
        ul3_emb = self.u_out_l3_emb(cate_seq_x[:,:,7]).view(bs, 80, -1)
        ul4_emb = self.u_out_l4_emb(cate_seq_x[:,:,8]).view(bs, 80, -1)


        seq_x = torch.cat((r_emb, c_emb, rc_dot_emb, rc_sum_emb,  uo_emb, ul1_emb, ul2_emb, ul3_emb, ul4_emb,  cont_seq_x), 2)
        seq_emb = self.seq_emb(seq_x)
        seq_emb, _ = self.lstm(seq_emb)
        if self.reg:
            output = self.head(seq_emb).view(bs, -1)
        else:
            output = self.head(seq_emb)

        return output
