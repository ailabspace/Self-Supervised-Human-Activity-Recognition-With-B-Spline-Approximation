import math
import os
import sys

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functorch.dim import Tensor

sys.path.extend(['../../Clusering/'])

import argparse

import torch
import torch.nn as nn

import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model, requires_grad=False)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # TODO: Test changing odd and even to every 5 and 10 with the exception of CLS & SEP
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe[:, 0] = torch.sin(position * div_term)
        #mid = int(max_seq_length/2)

        #pe[:, 1::5] = torch.cos(position * div_term)
        #pe[:, 6::5] = torch.sin(position * div_term)


        self.register_buffer('pe', pe.unsqueeze(0))
        # self.pe = nn.Parameter(pe.unsqueeze(0))

    def forward(self, x):
        return x + (self.pe[:, :x.size(1)])

# class Embedding(nn.Module):
#     def __init__(self, vocab_size, d_model, max_len, n_segments):
#         super().__init__()
#         self.tok_embed = nn.Linear(vocab_size, d_model)
#         self.pos_enc = nn.Linear(max_len, d_model)
#         # self.seg_embed = nn.Linear(n_segments, d_model)