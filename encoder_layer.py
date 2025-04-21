import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from multihead_attention import MultiHeadAttention
from position_wise_feed_forward_networks import PositionWiseFeedForward
from positional_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    