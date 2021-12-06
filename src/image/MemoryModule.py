from torch import nn
from torch import Tensor
import math
from torch.nn import functional as F
import torch


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thresh=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thresh = shrink_thresh

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X):
        att_weight = F.linear(X, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lamb=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


class MemoryModule(nn.Module):
    def __init__(self, mem_dim: int, feature_dim: int, shrink_tresh=0.0025, device='cuda'):
        super(MemoryModule).__init__()
        self.mem_dim = mem_dim
        self.feature_dim = feature_dim
        self.shrink_thresh = shrink_tresh
        self.device = device
        self.memory = MemoryUnit(self.mem_dim, self.feature_dim, self.shrink_thresh)

    def forward(self, X: Tensor):
        s = X.data.shape
        l = len(s)

        if l == 3:
            x = X.permute(0, 2, 1)
        elif l == 4:
            x = X.permute(0, 2, 3, 1)
        elif l == 5:
            x = X.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att}


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(X, lamb=0, epsilon=1e-12):
    output = (F.relu(X - lamb) * X) / (torch.abs(X - lamb) + epsilon)
    return output
