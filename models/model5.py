from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from .model import _weights_init, SimplifiedWaveBlock, WaveBlock



class Net5(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dims = net_params.hidden_dims
        wave_layers = net_params.wave_layers
        wave_block = net_params.wave_block
        kernel_size = net_params.kernel_size

        num_blocks = len(hidden_dims)

        if wave_block == 'simplified':
            wave_block = SimplifiedWaveBlock
        else:
            wave_block = WaveBlock

        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)


        self.features = nn.Sequential()
        self.features.add_module(
            'waveblock0', wave_block(
                wave_layers[0], 
                4 + len(net_params.cont_seq_cols), 
                hidden_dims[0], 
                kernel_size,
                ))
        self.features.add_module(
            'bn0', nn.BatchNorm1d(hidden_dims[0]))
        for i in range(num_blocks-1):
            self.features.add_module(
                f'waveblock{i+1}', wave_block(
                    wave_layers[i+1], 
                    hidden_dims[i], 
                    hidden_dims[i+1], 
                    kernel_size,
                    ))
            self.features.add_module(
                f'bn{i+1}', nn.BatchNorm1d(hidden_dims[i+1]))

        self.lstm = nn.Sequential(
            nn.LSTM(hidden_dims[-1], hidden_dims[-1], batch_first=True, num_layers=4, bidirectional=True)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1] * 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] * 2, 1),
        )

        self.apply(_weights_init)

    def forward(self, cate_seq_x, cont_seq_x):
        bs = cont_seq_x.size(0)
        r_emb = self.r_emb(cate_seq_x[:,:,0]).view(bs, 80, -1)
        c_emb = self.c_emb(cate_seq_x[:,:,1]).view(bs, 80, -1)
        seq_x = torch.cat((r_emb, c_emb, cont_seq_x), 2)

        x = seq_x.permute(0, 2, 1)
        x = self.features(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.head(x)
        return x.view(bs, -1)