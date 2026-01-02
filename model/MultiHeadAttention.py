import math
import os
import sys
sys.path.extend(['../../Clusering/'])

import argparse

import torch
import torch.nn as nn

import numpy as np

from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Number of dimensions must be divisible by number of heads
        assert d_model % num_heads == 0, "Number of model input must be divisible by number of heads"

        self.model_dim = d_model
        self.num_heads = num_heads
        self.k_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        # print(K.shape)
        # print(Q.shape)
        # print(self.k_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt((self.k_dim))

        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask, -1e9)

        attn_probs = torch.softmax(attn_scores, dim = -1)

        # Output is context in BERT
        output = torch.matmul(attn_probs, V)

        return output

    def split_heads(self, x):
        batch_size, seq_length, model_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.k_dim).transpose(1, 2)

    def combine_head(self, x):
        batch_size, _, seq_length, k_dim = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.model_dim)

    def forward(self, Q, K, V, mask = None):
        Q = self.split_heads(self.query(Q))
        K = self.split_heads(self.key(K))
        V = self.split_heads(self.value(V))


        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.output(self.combine_head(attn_output))

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, seqlen=1):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        x = self.forward_attention(q, k, v)

        x = self.proj(x)

        return x

    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

class SpatioTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, seqlen=1):
        B, N, T, C = x.shape

        # Attention With the Same Segment
        # x_spatial = self.forward_spatial(x.clone().view(B*N, T, C))
        #
        # x_spatial = x_spatial.view(B, N, T, C)
        #
        # x_temporal = self.forward_temporal(x.clone().permute(0, 2, 1, 3).contiguous().view(B * T, N, C))
        #
        # x_temporal = x_temporal.view(B, T, N, C).permute(0, 2, 1, 3).contiguous()

        # return x_spatial + x_temporal

        x = self.forward_temporal(x.permute(0, 2, 1, 3).contiguous().view(B * T, N, C))

        x = x.view(B, T, N, C).permute(0, 2, 1, 3).contiguous()

        x = self.forward_spatial((x.permute(0, 2, 1, 3).contiguous().view(B * T, N, C)))

        x = x.view(B, N, T, C)

        return x
    
    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_spatial(self, x):
        BN, T, C = x.shape
        qkv = self.qkv(x).reshape(BN, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.forward_attention(q, k, v)
        x = self.proj(x)

        return x

    def forward_temporal(self, x):
        BT, N, C = x.shape
        qkv = self.qkv(x).reshape(BT, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.forward_attention(q, k, v)
        x = self.proj(x)

        return x

class STAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.qkv_sp = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_tp = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, seqlen=1):
        B, N, T, C = x.shape

        # Attention With the Same Segment
        # x_spatial = self.forward_spatial(x.clone().view(B*N, T, C))
        #
        # x_spatial = x_spatial.view(B, N, T, C)
        #
        # x_temporal = self.forward_temporal(x.clone().permute(0, 2, 1, 3).contiguous().view(B * T, N, C))
        #
        # x_temporal = x_temporal.view(B, T, N, C).permute(0, 2, 1, 3).contiguous()

        # return x_spatial + x_temporal

        x = self.forward_temporal(x.permute(0, 2, 1, 3).contiguous().view(B * T, N, C))

        x = x.view(B, T, N, C).permute(0, 2, 1, 3).contiguous()

        x = self.forward_spatial((x.permute(0, 2, 1, 3).contiguous().view(B * T, N, C)))

        x = x.view(B, N, T, C)

        return x

    def forward_attention(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_spatial(self, x):
        BN, T, C = x.shape
        qkv = self.qkv_sp(x).reshape(BN, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.forward_attention(q, k, v)
        x = self.proj(x)

        return x

    def forward_temporal(self, x):
        BT, N, C = x.shape
        qkv = self.qkv_tp(x).reshape(BT, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = self.forward_attention(q, k, v)
        x = self.proj(x)

        return x