import sys

sys.path.extend(['../../Clustering/'])

import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, d_model, num_classes, dropout = 0.):
        super().__init__()
        # self.fc = nn.Linear(d_model, num_classes)
        self.fc = nn.Sequential(nn.BatchNorm1d(d_model, affine = False, eps = 1e-6),
                                nn.Linear(d_model, num_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc(self.dropout(x))