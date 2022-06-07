from __future__ import annotations

import sys
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

from .cnn1d import SimplifiedWaveBlock, WaveBlock


def _weights_init(m):
    if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate, padding_mode='replicate'))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate, padding_mode='replicate'))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SEModule, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        s = F.adaptive_avg_pool1d(x, 1)
        s = self.conv(s)
        x *= torch.sigmoid(s)
        return x


class KerasLikeLSTM(nn.Module):
    def __init__(self, input_dim, lstm_dim, act_fn):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(lstm_dim*2, lstm_dim)
        self.act_fn = act_fn


    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.out(x)
        x = self.act_fn(x)
        return x


class my_round_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ScaleLayer(nn.Module):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        self.min = -1.895744294564641
        self.max = 64.8209917386395
        self.step = 0.07030214545121005
        self.my_round_func = my_round_func()

    def forward(self, inputs):
        steps = inputs.add(-self.min).divide(self.step)
        int_steps = self.my_round_func.apply(steps)
        rescaled_steps = int_steps.multiply(self.step).add(self.min)
        clipped = torch.clamp(rescaled_steps, self.min, self.max)
        return clipped


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).cuda()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 



class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


# class Net6(nn.Module):
#     def __init__(self, net_params):
#         super().__init__()
#         dropout_rate = 0.1

#         inch = 8 + len(net_params.cont_seq_cols)
#         kernel_size = net_params.kernel_size
#         self.r_emb = nn.Embedding(3, 2, padding_idx=0)
#         self.c_emb = nn.Embedding(3, 2, padding_idx=0)
#         self.rc_emb = nn.Embedding(9, 4, padding_idx=0)

#         self.conv1d_1 = nn.Conv1d(inch, 32, kernel_size=1, stride=1, dilation=1, padding=0, padding_mode='replicate')
#         self.batch_norm_conv_1 = nn.BatchNorm1d(32)
#         self.dropout_conv_1 = nn.Dropout(dropout_rate)

#         self.conv1d_2 = nn.Conv1d(inch+16+32+64+128, 32, kernel_size=1, stride=1, dilation=1, padding=0, padding_mode='replicate')
#         self.batch_norm_conv_2 = nn.BatchNorm1d(32)
#         self.dropout_conv_2 = nn.Dropout(dropout_rate)

#         self.wave_block1 = Wave_Block(32, 16, 12, kernel_size)
#         self.wave_block2 = Wave_Block(inch+16, 32, 8, kernel_size)
#         self.wave_block3 = Wave_Block(inch+16+32, 64, 4, kernel_size)
#         self.wave_block4 = Wave_Block(inch+16+32+64, 128, 1, kernel_size)

#         self.se_module1 = SEModule(16)
#         self.se_module2 = SEModule(32)
#         self.se_module3 = SEModule(64)
#         self.se_module4 = SEModule(128)        

#         self.batch_norm_1 = nn.BatchNorm1d(16)
#         self.batch_norm_2 = nn.BatchNorm1d(32)
#         self.batch_norm_3 = nn.BatchNorm1d(64)
#         self.batch_norm_4 = nn.BatchNorm1d(128)

#         self.dropout_1 = nn.Dropout(dropout_rate)
#         self.dropout_2 = nn.Dropout(dropout_rate)
#         self.dropout_3 = nn.Dropout(dropout_rate)
#         self.dropout_4 = nn.Dropout(dropout_rate)

#         self.lstm = nn.LSTM(inch+16+32+64, 32, 1, batch_first=True, bidirectional=True)
#         self.fc0 = nn.Linear(64, 32)
#         self.fc1 = nn.Linear(32, 1)
#         self.fc2 = nn.Linear(32, 1)

#     def forward(self, x1, x2):

#         bs = x1.size(0)
#         r_emb = self.r_emb(x1[:,:,0]).view(bs, 80, -1)
#         c_emb = self.c_emb(x1[:,:,1]).view(bs, 80, -1)
#         rc_emb = self.rc_emb(x1[:,:,2]).view(bs, 80, -1)
#         x = torch.cat((r_emb, c_emb, rc_emb, x2), 2)

#         x = x.permute(0, 2, 1)

#         x0 = self.conv1d_1(x)
#         x0 = F.relu(x0)
#         x0 = self.batch_norm_conv_1(x0)
#         x0 = self.dropout_conv_1(x0)

#         x1 = self.wave_block1(x0)
#         x1 = self.batch_norm_1(x1)
#         x1 = self.dropout_1(x1)
#         x1 = self.se_module1(x1)
#         x2_base = torch.cat([x1, x], dim=1)

#         x2 = self.wave_block2(x2_base)
#         x2 = self.batch_norm_2(x2)
#         x2 = self.dropout_2(x2)
#         x2 = self.se_module2(x2)
#         x3_base = torch.cat([x2_base, x2], dim=1)

#         x3 = self.wave_block3(x3_base)
#         x3 = self.batch_norm_3(x3)
#         x3 = self.dropout_3(x3)
#         x3 = self.se_module3(x3)
#         x4_base = torch.cat([x3_base, x3], dim=1)

#         x4 = self.wave_block4(x4_base)
#         x4 = self.batch_norm_4(x4)
#         x4 = self.dropout_4(x4)
#         x4 = self.se_module4(x4)
#         x5_base = torch.cat([x4_base, x4], dim=1)

#         x5 = self.conv1d_2(x5_base)
#         x5 = F.relu(x5)
#         x5 = self.batch_norm_conv_2(x5)
#         x5 = self.dropout_conv_2(x5)

#         lstm_out, _ = self.lstm(x4_base.permute(0, 2, 1))
#         out1 = self.fc0(lstm_out)
#         out1 = self.fc1(out1)
#         out2 = x5.permute(0, 2, 1)
#         out2 = self.fc2(out2)
#         return (out1 * out2).squeeze(-1)

# class Net3(nn.Module):
#     def __init__(self, net_params):
#         super().__init__()
#         hidden_dims = net_params.hidden_dims
#         wave_layers = net_params.wave_layers
#         wave_block = net_params.wave_block
#         kernel_size = net_params.kernel_size

#         num_blocks = len(hidden_dims)

#         if wave_block == 'simplified':
#             wave_block = SimplifiedWaveBlock
#         else:
#             wave_block = WaveBlock

#         self.r_emb = nn.Embedding(3, 2, padding_idx=0)
#         self.c_emb = nn.Embedding(3, 2, padding_idx=0)


#         self.features = nn.Sequential()
#         self.features.add_module(
#             'waveblock0', wave_block(
#                 wave_layers[0], 
#                 4 + len(net_params.cont_seq_cols), 
#                 hidden_dims[0], 
#                 kernel_size,
#                 ))
#         self.features.add_module(
#             'bn0', nn.BatchNorm1d(hidden_dims[0]))
#         for i in range(num_blocks-1):
#             self.features.add_module(
#                 f'waveblock{i+1}', wave_block(
#                     wave_layers[i+1], 
#                     hidden_dims[i], 
#                     hidden_dims[i+1], 
#                     kernel_size,
#                     ))
#             self.features.add_module(
#                 f'bn{i+1}', nn.BatchNorm1d(hidden_dims[i+1]))

#         self.lstm = nn.Sequential(
#             nn.LSTM(hidden_dims[-1], hidden_dims[-1], batch_first=True, num_layers=4, bidirectional=True)
#         )

#         self.head = nn.Sequential(
#             nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1] * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[-1] * 2, 1),
#         )

#         self.apply(_weights_init)

#     def forward(self, cate_seq_x, cont_seq_x):
#         bs = cont_seq_x.size(0)
#         r_emb = self.r_emb(cate_seq_x[:,:,0]).view(bs, 80, -1)
#         c_emb = self.c_emb(cate_seq_x[:,:,1]).view(bs, 80, -1)
#         seq_x = torch.cat((r_emb, c_emb, cont_seq_x), 2)

#         x = seq_x.permute(0, 2, 1)
#         x = self.features(x)
#         x = x.permute(0, 2, 1)
#         x, _ = self.lstm(x)
#         x = self.head(x)
#         return x.view(bs, -1)


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
