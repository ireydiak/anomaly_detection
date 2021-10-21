from _typeshed import _KT_contra
from torch import nn

from . import BaseModel
import torch


class MemAE(BaseModel):

    def __init__(self, D: int, **kwargs):
        super(MemAE, self).__init__()
        self.D = D
        self.device = kwargs.get('device', 'cpu')
        self.rep_dim = kwargs.get('rep_dim', 1)
        self._build_net()
    
    def _build_net(self):
        D, L = self.D, self.rep_dim 
        self.enc = nn.Sequential(nn.Linear(D, D//2), nn.Linear(D//2, D//4), nn.Linear(D//4, self.rep_dim))
        self.dec = nn.Sequential(nn.Linear(L, D//4), nn.Linear(D//4, D//2), nn.Linear(D//2, D))
        self.mem_module = None

    def forward(self, X: torch.Tensor):
        pass

    def print_name(self) -> str:
        return "MemAE"
    
    def get_params(self) -> dict:
        return {
            'rep_dim': self.rep_dim,
            'D': self.D
        }