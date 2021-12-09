from torch.nn import Parameter
from torch.nn.modules.activation import Softplus
from src.tabular.models import BaseModel
import torch.nn as nn
from torch import Tensor
import torch


class DSEBM(BaseModel):

    def __init__(self, D: int):
        super(DSEBM, self).__init__()
        self.D = D
        self.noise = None
        self._build_network()

    def _build_network(self):
        self.fc_1 = nn.Linear(self.D, 128)
        self.fc_2 = nn.Linear(128, 512)
        self.softp = nn.Softplus()

        self.bias_inv_1 = Parameter(Tensor(128))
        self.bias_inv_2 = Parameter(Tensor(self.D))

        torch.nn.init.xavier_normal_(self.fc_2.weight)
        torch.nn.init.xavier_normal_(self.fc_1.weight)
        self.fc_1.bias.data.zero_()
        self.fc_2.bias.data.zero_()
        self.bias_inv_1.data.zero_()
        self.bias_inv_2.data.zero_()

    def random_noise_like(self, X: torch.Tensor):
        return torch.normal(mean=0., std=1., size=X.shape).float()

    def forward(self, X: torch.Tensor):
        output = self.softp(self.fc_1(X))
        output = self.softp(self.fc_2(output))

        # inverse layer
        output = self.softp((output @ self.fc_2.weight) + self.bias_inv_1)
        output = self.softp((output @ self.fc_1.weight) + self.bias_inv_2)

        return output

    def get_params(self):
        return {
            'D': self.D
        }
