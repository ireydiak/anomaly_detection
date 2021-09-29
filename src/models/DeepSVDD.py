import torch.nn as nn
import torch.utils.data
from torch import Tensor


class DeepSVDD(nn.Module):
    """
    Follows SKlearn's API
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM.decision_function)
    """

    def __init__(self, D: int, n_epoch: int, kwargs: dict):
        super(DeepSVDD, self).__init__()
        self.D = D
        self.n_epoch = n_epoch
        self.device = kwargs.get('device', 'cpu')

        self.c = Tensor(kwargs.get('c'), device=self.device)
        self.R = Tensor(kwargs.get('R'), device=self.device)
        self.lamb = kwargs.get('lambda', 1e-6)
        self._build_network()

    def _build_network(self):
        self.net = nn.Sequential(
            nn.Linear(self.D, self.D + (self.D // 4)),
            nn.ReLU(),
            nn.Linear(self.D, self.D + (self.D // 2)),
            nn.ReLU()
        ).to(self.device)

    def forward(self, X: Tensor):
        return self.net(X)

    def fit(self, X: torch.utils.data.DataLoader):
        for epoch in range(self.n_epoch):
            print(f"\nEpoch: {epoch + 1} of {self.n_epochs}")
            for _, X_i in enumerate(X):
                train_inputs = X_i[0].to(self.device).float()
                loss = self.fit_one(X_i)

    def fit_one(self, X_i: Tensor):
        return 0

    def score(self, X: Tensor):
        return (self.net(X) - self.c) ** 2

    def decision_function(self, X: Tensor):
        return (self.net(X) - self.c) ** 2
