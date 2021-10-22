import math
import torch
from torch import nn
from torch.nn import functional as F

def hard_shrink_relu(input, lamb=0, epsilon=1e-12) -> float:
    return (F.relu(input - lamb) * input) / (torch.abs(input - lamb) + epsilon)

class MemoryUnit(nn.Module):
    
    def __init__(self, N: int, L: int, shrink_thresh: float=0.0025, device='cpu'):
        super(MemoryUnit, self).__init__()
        self.L = L
        self.N = N
        self.device = device
        self.shrink_thresh = shrink_thresh
        self.weight = nn.parameter.Parameter(torch.Tensor(N, L)).to(device)
        self.bias = None
        self.reset_params()

    def reset_params(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size()[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X: torch.Tensor):
        att_weight = F.linear(X, self.weight)
        att_weight = F.softmax(att_weight, dim=1)
        if self.shrink_thresh > 0:
            att_weight = hard_shrink_relu(att_weight, self.shrink_thresh)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        output = F.linear(att_weight, self.weight.T)
        return output, att_weight
    
    def get_params(self) -> dict:
        return {
            'mem_dim': self.N,
            'rep_dim': self.L
        }