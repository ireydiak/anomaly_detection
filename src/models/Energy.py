from torch.nn.modules.activation import Softplus
from . import BaseModel
import torch.nn as nn
from torch import Tensor


class DSEBM(BaseModel):
    def __init__(self, D: int, **kwargs):
        super(DSEBM, self).__init__()
        self.D = D
        self.device = kwargs.get('device', 'cuda')
        self.net = self._build_network()
        self.rep_dim = D // 4
    
    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.D, self.D // 2),
            nn.Softplus(),
            nn.Linear(self.D // 2, self.D // 4),
            nn.Softplus(),
            nn.Linear(self.D // 4, self.D // 2),
            nn.Softplus(),
            nn.Linear(self.D // 2, self.D),
            nn.Softplus(),
        ).to(self.device)
    
    def forward(self, X: Tensor):
        return self.net(X)