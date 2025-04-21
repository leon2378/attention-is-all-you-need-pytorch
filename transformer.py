import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

import positional_encoding
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        