import torch as pt
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import math

class SkipBlock(nn.Module):
    def __init__(self, dim:int, h:int=512) -> None:
        super().__init__()
        self.f =  nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(),
            nn.Linear(h, dim),
            # nn.Tanh()
        )
    def forward(self, x):
        return self.f(x) + x

class Model(nn.Module):
    def __init__(self, n_vheads:int = 1, lr:float=1e-3) -> None:
        super().__init__()
        h = 512
        self.fc = nn.Sequential(
            nn.Linear(172, h),
            # nn.LayerNorm(h),
            # SkipBlock(h),
            # nn.ReLU(),
            # SkipBlock(h),
            # nn.ReLU(),
            SkipBlock(h),
            SkipBlock(h),
            SkipBlock(h),
            # nn.ReLU(),
            SkipBlock(h),
            # nn.ReLU(),
        )
        self.V_head = nn.Sequential(
            # nn.ReLU(),
            SkipBlock(h),
            # nn.ReLU(),
            # SkipBlock(h),
            nn.Linear(h, n_vheads),
        )
        self.pi_head = nn.Sequential(
            SkipBlock(h),
            # nn.ReLU(),
            # SkipBlock(h),
            # nn.ReLU(),
            nn.Linear(h, 36)
        )
        self.optim = pt.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)
        
    
    def forward(self, s:np.ndarray)->tuple[pt.Tensor, pt.Tensor]:
        if isinstance(s, np.ndarray):
            s = pt.tensor(s, dtype=pt.float32)
        if len(s.shape) == 1:
            v, logits = self.forward(s[None, ...])
            return v.squeeze(0), logits.squeeze(0)
        mask = self.get_mask(s)
        z = self.fc.forward(s)
        v = self.V_head.forward(z)
        logits = self.pi_head.forward(z)
        logits[mask] -= pt.inf
        return v, logits
    
    def get_mask(self, s:pt.Tensor):
        classic_state = s[..., :90].reshape(s.shape[:-1]+ (9, 10))
        occupied = classic_state[..., :-1].any(dim=-1)
        mask = pt.zeros(s.shape[:-1] + (36,), dtype=bool)
        for a in range(36):
            i, j = ind2move(a)
            mask[..., a] = pt.logical_or(occupied[..., i], occupied[..., j])
        return mask

    def entropy(self, s:pt.Tensor)->pt.Tensor:
        if isinstance(s, np.ndarray):
            s = pt.tensor(s)
        _, logits = self.forward(s)
        # mask = self.get_mask(s)
        # print(logits)
        # print(mask)
        p = pt.softmax(logits, -1)
        logp = pt.log(p + 1e-7)
        return -pt.sum(logp * p, -1)



class QNet(nn.Module):
    def __init__(self, n:int = 1, lr:float=1e-3) -> None:
        super().__init__()
        h = 512

        self.s_enc = nn.Sequential(
            nn.Linear(172, h),
            SkipBlock(h),
            SkipBlock(h),
            )
        self.a_enc = nn.Embedding(36, h)
        
        self.fc = nn.Sequential(
            # nn.Linear(h, h),
            # nn.ReLU(),
            SkipBlock(h),
            SkipBlock(h),
            SkipBlock(h),
            # nn.Linear(h, h),
            # nn.ReLU(),
            SkipBlock(h),
            nn.Linear(h, n),
            # nn.Tanh()
        )
        self.optim = pt.optim.Adam(self.parameters(), lr=lr, weight_decay=0., amsgrad=False)
        
    
    def forward(self, s:np.ndarray, a)->pt.Tensor:
        if isinstance(s, np.ndarray):
            s = pt.tensor(s, dtype=pt.float32)
        if isinstance(a, np.ndarray):
            a = pt.tensor(a, dtype=pt.long)
        if len(s.shape) == 1:
            q = self.forward(s[None, ...], a[None, ...])
            return q.squeeze(0)
        zs = self.s_enc.forward(s)
        # a = self.a_onehot[a]
        za = self.a_enc.forward(a)

        z = zs + za
        return self.fc.forward(z)

class VNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        h = 256

        self.fc = nn.Sequential(
            nn.Linear(171, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            # nn.Tanh()
        )
        self.optim = pt.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0, amsgrad=True)
        
    
    def forward(self, s:np.ndarray)->pt.Tensor:
        if isinstance(s, np.ndarray):
            s = pt.tensor(s, dtype=pt.float32)
        if len(s.shape) == 1:
            q = self.forward(s[None, ...])
            return q
        return self.fc.forward(s)

def ind2move(n) -> tuple[int, int]:
    i = (17 - math.sqrt(17*17 - 8*n))/2
    i = int(i)
    j = (2 * n + 2 - 15 * i + i*i)//2
    return i, j