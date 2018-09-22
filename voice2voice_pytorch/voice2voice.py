# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

from .modules import Conv1d, ConvTranspose1d, Embedding, Linear, GradMultiply
from .modules import get_mask_from_lengths, SinusoidalEncoding, Conv1dGLU


def expand_speaker_embed(inputs_btc, speaker_embed=None, tdim=1):
    if speaker_embed is None:
        return None
    # expand speaker embedding for all time steps
    # (B, N) -> (B, T, N)
    ss = speaker_embed.size()
    speaker_embed_btc = speaker_embed.unsqueeze(1).expand(
        ss[0], inputs_btc.size(tdim), ss[-1])
    return speaker_embed_btc

class Encoder(nn.Module):
    def __init__(self, in_dim=513, speaker_style_dim=16,
                 convolutions=((64, 5, 1), (32, 5, 3), (16, 5, 9)), downsample_t = 4,
                 dropout=0.1, apply_grad_scaling=False):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None
        self.apply_grad_scaling = apply_grad_scaling
        self.downsample_t = downsample_t

        # Non causual convolution blocks
        in_channels = 64
        self.convolutions = nn.ModuleList()

        self.speaker_fc1 = Linear(in_dim, in_channels, dropout=dropout)

        self.convolutions = nn.ModuleList()
        std_mul = 1.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(1, speaker_style_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, convolutions[len(convolutions)-1][0], kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, input):
        
        # downsample inputs
        x = input[:, 0::self.downsample_t, :].contiguous()
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.speaker_fc1(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        for f in self.convolutions:
            x = f(x, None) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        x = x[:, int(0.25*x.size(1)):int(0.75*x.size(1)), :]

        x = x.mean(1, True).reshape((x.size(0), x.size(2))).contiguous()

        return x

class Decoder(nn.Module):
    def __init__(self, in_dim=513, speaker_style_dim=16,
                 convolutions=((512, 5, 3),)*4,
                 dropout=0.1, apply_grad_scaling=False):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None
        self.apply_grad_scaling = apply_grad_scaling

        self.speaker_fc1 = Linear(speaker_style_dim, in_dim, dropout=dropout)
        self.speaker_fc2 = Linear(speaker_style_dim, in_dim, dropout=dropout)

        # Non causual convolution blocks
        in_channels = in_dim
        self.convolutions = nn.ModuleList()

        self.convolutions = nn.ModuleList()
        std_mul = 1.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(2, speaker_style_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, in_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, x, style):
        
        # x
        x = F.dropout(x, p=self.dropout, training=self.training)

        # expand speaker style for all time steps
        speaker_style_btc = expand_speaker_embed(x, style)
        if speaker_style_btc is not None:
            speaker_style_btc = F.dropout(speaker_style_btc, p=self.dropout, training=self.training)
            x = x + F.softsign(self.speaker_fc1(speaker_style_btc))

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_style_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        if speaker_style_btc is not None:
            x = x + F.softsign(self.speaker_fc2(speaker_style_btc))

        return torch.sigmoid(x)
