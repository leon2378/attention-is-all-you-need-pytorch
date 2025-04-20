import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import multihead_attention

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)