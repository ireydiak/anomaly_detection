import torch.nn as nn
from torch import Tensor
from src.tabular.models import BaseModel


class DeepSVDD(BaseModel):

    def __init__(self, in_features: int, **kwargs):
        super(DeepSVDD, self).__init__()
        self.in_features = in_features
        self.device = kwargs.get('device', 'cuda')
        self.net = self._build_network()
        self.rep_dim = in_features // 4

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.in_features, self.in_features // 2),
            nn.ReLU(),
            nn.Linear(self.in_features // 2, self.in_features // 4)
        ).to(self.device)

    def forward(self, X: Tensor):
        return self.net(X)

    def get_params(self) -> dict:
        return {'in_features': self.in_features, 'rep_dim': self.rep_dim}

    def print_name(self):
        return "DeepSVDD"
