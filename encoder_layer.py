import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import multihead_attention, position_wise_feed_forward_networks, positional_encoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()