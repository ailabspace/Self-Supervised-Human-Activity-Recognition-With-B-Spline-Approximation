from .MultiHeadAttention import Attention, SpatioTemporalAttention,STAttention
from .PositionWiseFeedForward import PositionWiseFeedForward

import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = Attention(d_model, num_heads, qkv_bias=True)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ### Post Norm
        # attn_output = self.self_attn(x, x, x, mask = mask)
        # if self.training:
        #     attn_output = self.dropout(attn_output)
        # x = self.norm1(x + attn_output)
        # ff_output = self.feed_forward(x)
        # if self.training:
        #     ff_output = self.dropout(ff_output)
        # x = self.norm2(x + ff_output)

        ### Trying Pre Norm
        norm1_x = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm1_x))
        norm2_x = self.norm2(x)
        x = x + self.dropout(self.feed_forward(norm2_x))
        return x

class EncoderLayerST(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayerST, self).__init__()
        #self.self_attn = SpatioTemporalAttention(d_model, num_heads, qkv_bias=True)
        self.self_attn = STAttention(d_model, num_heads, qkv_bias=True)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ### Post Norm
        # attn_output = self.self_attn(x, x, x, mask = mask)
        # if self.training:
        #     attn_output = self.dropout(attn_output)
        # x = self.norm1(x + attn_output)
        # ff_output = self.feed_forward(x)
        # if self.training:
        #     ff_output = self.dropout(ff_output)
        # x = self.norm2(x + ff_output)

        ### Trying Pre Norm
        norm1_x = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm1_x))
        norm2_x = self.norm2(x)
        x = x + self.dropout(self.feed_forward(norm2_x))
        return x