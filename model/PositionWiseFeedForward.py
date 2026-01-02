import math
import os
import sys
sys.path.extend(['../../Clusering/'])

import argparse

import torch
import torch.nn as nn

import numpy as np

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        #self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.gelu(self.fc1(x)))))