import torch as pt
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import math

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        h = 128
        self.fc = nn.Sequential(
            nn.Linear(180, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.V_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.pi_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 36)
        )
        self.optim = pt.optim.Adam(self.parameters(), lr=1e-3)
        
    
    def forward(self, s:np.ndarray):
        s = pt.tensor(s)
        mask = self.get_mask(s)
        s = s.flatten(-2, -1).float()
        z = self.fc.forward(s)
        v = self.V_head.forward(z)
        logits = self.pi_head.forward(z)
        logits[mask] -= pt.inf
        return v, logits
    
    def get_mask(self, s:pt.Tensor):
        occupied = s[:9, :9].any(dim=1)
        # print(occupied)
        mask = pt.zeros(36, dtype=bool)
        for a in range(36):
            i, j = ind2move(a)
            mask[a] = occupied[i] or occupied[j]
        # print(mask)
        # input()
        return mask

    def entropy(self, s:pt.Tensor):
        _, logits = self.forward(s)
        mask = self.get_mask(pt.tensor(s))
        logp = pt.log_softmax(logits[pt.logical_not(mask)], -1)
        return -pt.sum(logp.exp() * logp, -1)


def ind2move(n) -> tuple[int, int]:
    i = (17 - math.sqrt(17*17 - 8*n))/2
    i = int(i)
    j = (2 * n + 2 - 15 * i + i*i)//2
    return i, j