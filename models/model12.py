
from __future__ import annotations
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from .model import _weights_init


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class Net12(nn.Module):
    
    def __init__(self, net_params):
        super(Net12, self).__init__()
        # This embedding method from: https://www.kaggle.com/theoviel/deep-learning-starter-simple-lstm
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)
        self.seq_emb = nn.Sequential(
            nn.Linear(4 + len(net_params.cont_seq_cols), net_params.hidden_size),
            nn.LayerNorm(net_params.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pos_encoder = PositionalEncoding(d_model=net_params.hidden_size, dropout=0.2)
        encoder_layers = nn.TransformerEncoderLayer(d_model=net_params.hidden_size, nhead=8, dim_feedforward=2048, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.head = nn.Linear(net_params.hidden_size, 1)
        
        # Encoder
        initrange = 0.1
        self.r_emb.weight.data.uniform_(-initrange, initrange)
        self.c_emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, cate_seq_x, cont_seq_x):
        bs = cate_seq_x.shape[0]
        r_emb = self.r_emb(cate_seq_x[:,:,0].long()).view(bs, 80, -1)
        c_emb = self.c_emb(cate_seq_x[:,:,1].long()).view(bs, 80, -1)
        seq_x = torch.cat((r_emb, c_emb, cont_seq_x), 2)
        h = self.seq_emb(seq_x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        regr = self.head(h)
                    
        return regr.view(bs, -1)
    