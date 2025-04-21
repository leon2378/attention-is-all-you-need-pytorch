import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import multihead_attention
import position_wise_feed_forward_networks


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        