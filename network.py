import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """MLP Model"""
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Sigmoid(),
        )

    def forward(self, t):
        return self.main(t)
