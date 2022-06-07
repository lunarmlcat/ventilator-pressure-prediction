from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable
from .model import _weights_init, GaussianNoise, KerasLikeLSTM



class Net10(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.net_params = net_params
        self.hidden_size = net_params.hidden_size
        self.dense_dim = net_params.dense_dim
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)
        self.rc_emb = nn.Embedding(9, 4, padding_idx=0)

        self.mlp = nn.Sequential(
            nn.Linear(4 + len(net_params.cont_seq_cols), self.dense_dim // 2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(self.dense_dim // 2, self.dense_dim),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(self.dense_dim, self.dense_dim,
                            dropout=0., batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.dense_dim * 2, self.dense_dim//2,
                            dropout=0., batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(self.dense_dim//2 * 2, self.dense_dim//4,
                            dropout=0., batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(self.dense_dim//4 * 2, self.dense_dim//8,
                            dropout=0., batch_first=True, bidirectional=True)
                            
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size//8 * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size//8 * 2, 1),
        )

        self.reg = True if self.net_params.num_classes == 1 else False

        self.apply(_weights_init) 

    def forward(self, cate_seq_x, cont_seq_x):
        bs = cont_seq_x.size(0)
        r_emb = self.r_emb(cate_seq_x[:,:,0].long()).view(bs, 80, -1)
        c_emb = self.c_emb(cate_seq_x[:,:,1].long()).view(bs, 80, -1)
        # rc_emb = self.rc_emb(cate_seq_x[:,:,2].long()).view(bs, 80, -1)

        seq_x = torch.cat((r_emb, c_emb, cont_seq_x), 2)
        seq_emb = self.mlp(seq_x)
        seq_emb, _ = self.lstm1(seq_emb)
        seq_emb, _ = self.lstm2(seq_emb)
        seq_emb, _ = self.lstm3(seq_emb)
        seq_emb, _ = self.lstm4(seq_emb)

        if self.reg:
            output = self.head(seq_emb).view(bs, -1)
        else:
            output = self.head(seq_emb)

        return output


if __name__ == "__main__":
    import yaml
    from addict import Dict

    with open(f"configs/{sys.argv[1]}.yml", "r")  as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = Dict(yml)

    print("num_features: ", len(cfg.Model.params.cont_seq_cols))
    model = globals()[cfg.Model.cls](
        net_params=cfg.Model.params
    )

    model.eval()
    batch_size = 4

    x1 = torch.randint(0, 3, (batch_size, 80, 3)).long()
    x2 = torch.randn(batch_size, 80, len(cfg.Model.params.cont_seq_cols))


    logits = model(x1, x2)
    print(logits.shape)