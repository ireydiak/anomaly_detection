import time
from sklearn import metrics
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch
import numpy as np


class DeepSVDDTrainer:

    def __init__(self, model, R=None, c=None,
                 lr: float = 1e-4, n_epochs: int = 100, batch_size: int = 128, n_jobs_dataloader: int = 0,
                 device: str = 'cuda'):
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        self.n_epochs = n_epochs
        self.lr = lr
        self.c = c
        self.R = R

    def train(self, dataset: DataLoader):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print("Initializing center c...")
            self.c = self.init_center_c(dataset)
            print("Center c initialized.")

        print('Started training')
        epoch_loss = 0.0
        epoch_start_time = time.time()
        for epoch in range(self.n_epochs):
            for sample in dataset:
                X, _ = sample
                X = X.to(self.device).float()

                # Reset gradient
                optimizer.zero_grad()

                outputs = self.model(X)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                loss = torch.mean(dist)

                # Backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss

            epoch_train_time = time.time() - epoch_start_time
            print(
                "Epoch {}/{} \tTime: {:.3f}\t Loss: {:.8f}".format(
                    epoch + 1, self.n_epochs, epoch_train_time, epoch_loss
                )
            )
        print("Finished training")

    def test(self, dataset: DataLoader, pos_label: int = 1):
        self.model.eval()
        idx_label_score = []
        y_true, scores = [], []
        res = {"Accuracy": -1, "Precision": -1, "Recall": -1, "F1-Score": -1, "AUC": -1}
        with torch.no_grad():
            for row in dataset:
                X, y = row
                X = X.to(self.device).float()
                outputs = self.model(X)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                score = dist
                idx_label_score += list(zip(
                    y.cpu().data.numpy().tolist(),
                    score.cpu().data.numpy().tolist())
                )
                y_true.extend(y.cpu().tolist())
                scores.extend(score.cpu().tolist())

        thresh = np.percentile(scores, 80)
        y_pred = (scores >= thresh).astype(int)
        y_true = np.array(y_true)
        res["Precision"], res["Recall"], res["F1-Score"], _ = metrics.precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=pos_label
        )

        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        res["AUC"] = metrics.roc_auc_score(labels, scores)
        return res

    def init_center_c(self, train_loader: DataLoader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
           Code taken from https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py"""
        n_samples = 0
        c = torch.zeros(self.model.rep_dim, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for sample in train_loader:
                # get the inputs of the batch
                X, _ = sample
                X = X.to(self.device).float()
                outputs = self.model(X)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def get_params(self) -> dict:
        return {'c': self.c, 'R': self.R, **self.model.get_params()}


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)