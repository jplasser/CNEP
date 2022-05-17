from collections import OrderedDict
from typing import Tuple, Union

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class NotesDataEncoder(nn.Module):
    def __init__(self, dims=[800], input_dim=700, output_dim=1024, batchnorm=False, actfunc=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        dims = [input_dim] + dims + [output_dim]
        self.dims = dims
        self.layers = len(dims) - 1

        if batchnorm:
            self.encoder = nn.Sequential(
                *[nn.Sequential(# nn.LayerNorm(dims[l_i]),
                                nn.Linear(dims[l_i], dims[l_i + 1]),
                                nn.BatchNorm1d(dims[l_i + 1]),
                                # nn.Dropout(),
                                actfunc()) for l_i in range(self.layers)],
                                nn.LayerNorm(dims[-1])
            )
        else:
            self.encoder = nn.Sequential(
                *[nn.Sequential(# nn.LayerNorm(dims[l_i]),
                                nn.Linear(dims[l_i], dims[l_i + 1]),
                                # nn.Dropout(),
                                actfunc()) for l_i in range(self.layers)],
                                nn.LayerNorm(dims[-1])
            )


    def forward(self, x: torch.Tensor):
        return self.encoder(x)
