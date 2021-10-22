from torch import nn
from torch.nn.modules.activation import Tanh
from . import MemoryUnit
from . import BaseModel
import torch


class MemAE(BaseModel):

    def __init__(self, D: int, mem_dim: int=50, rep_dim: int=1, device='cpu'):
        super(MemAE, self).__init__()
        self.D = D
        self.mem_dim = mem_dim
        self.device = device
        self.rep_dim = rep_dim
        self._build_net()
    
    def _build_net(self):
        D, L = self.D, self.rep_dim 
        self.enc = nn.Sequential(
            nn.Linear(D, D//2),
            nn.Tanh(),
            nn.Linear(D//2, D//4),
            nn.Tanh(),
            nn.Linear(D//4, L)
        ).to(self.device)
        self.dec = nn.Sequential(
            nn.Linear(L, D//4),
            nn.Tanh(),
            nn.Linear(D//4, D//2),
            nn.Tanh(),
            nn.Linear(D//2, D)
        ).to(self.device)
        self.mem_module = MemoryUnit(self.mem_dim, self.rep_dim)

    def forward(self, X: torch.Tensor):
        z = self.enc(X)
        z_hat, att = self.mem_module(z)
        x_hat = self.dec(z_hat)
        return x_hat, att

    def print_name(self) -> str:
        return "MemAE"
    
    def get_params(self) -> dict:
        return {
            'rep_dim': self.rep_dim,
            'D': self.D
        }