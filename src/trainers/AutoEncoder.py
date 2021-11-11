from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
from . import BaseTrainer


class DAGMMTrainer(BaseTrainer):

    def __init__(self, lamb_1: float = 0.1, lamb_2: float = 0.005, **kwargs) -> None:
        super(DAGMMTrainer, self).__init__(**kwargs)
        self.lamb_1 = lamb_1
        self.lamb_2 = lamb_2
        self.phi = None
        self.mu = None
        self.cov_mat = None
        self.covs = None

    def train_iter(self, sample: torch.Tensor):
        z_c, x_prime, _, z_r, gamma_hat = self.model(sample)
        phi, mu, cov_mat = self.compute_params(z_r, gamma_hat)
        energy_result, pen_cov_mat = self.estimate_sample_energy(
            z_r, phi, mu, cov_mat
        )
        return self.loss(sample, x_prime, energy_result, pen_cov_mat)

    def loss(self, x, x_prime, energy, pen_cov_mat):
        rec_err = ((x - x_prime) ** 2).mean()
        return rec_err + self.lamb_1 * energy + self.lamb_2 * pen_cov_mat

    def test(self, dataset: DataLoader):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        N, gamma_sum, mu_sum, cov_mat_sum = 0, 0, 0, 0

        # Change the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for row in dataset:
                X, y = row
                X = X.to(self.device).float()

                # forward pass
                code, x_prime, cosim, z, gamma = self.model(X)
                phi, mu, cov_mat = self.compute_params(z, gamma)

                batch_gamma_sum = gamma.sum(axis=0)

                gamma_sum += batch_gamma_sum
                mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
                cov_mat_sum += cov_mat * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only
                N += X.shape[0]

            train_phi = gamma_sum / N
            train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            train_cov = cov_mat_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

            print("Train N:", N)
            print("\u03C6 :\n", train_phi.shape)
            print("\u03BC :\n", train_mu.shape)
            print("\u03A3 :\n", train_cov.shape)

            # Calculate energy using estimated parameters

            scores = []
            y_true = []

            for row in dataset:
                X, y = row
                X = X.to(self.device).float()

                # forward pass
                code, x_prime, cosim, z, gamma = self.model(X)
                sample_energy, pen_cov_mat = self.estimate_sample_energy(
                    z, train_phi, train_mu, train_cov, average_energy=False
                )

                scores.extend(sample_energy.cpu().numpy())
                y_true.extend(y.numpy())

            return np.array(y_true), np.array(scores)
            #
            # for data in test_loader:
            #     test_inputs, label_inputs = data[0].float().to(self.device), data[1]
            #
            #     # forward pass
            #     code, x_prime, cosim, z, gamma = self.model(test_inputs)
            #     sample_energy, pen_cov_mat = self.model.estimate_sample_energy(
            #         z, train_phi, train_mu, train_cov, average_energy=False, device=self.device
            #     )
            #     test_energy.append(sample_energy.cpu().numpy())
            #     test_z.append(z.cpu().numpy())
            #     test_labels.append(label_inputs.numpy())
            #
            # test_energy = np.concatenate(test_energy, axis=0)
            # test_z = np.concatenate(test_z, axis=0)
            # test_labels = np.concatenate(test_labels, axis=0)
            #
            # combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            #
            # thresh = np.percentile(combined_energy, energy_threshold)
            # print("Threshold :", thresh)
            #
            # # Prediction using the threshold value
            # y_pred = (test_energy > thresh).astype(int)
            # y_true = test_labels.astype(int)
            #
            # accuracy = metrics.accuracy_score(y_true, y_pred)
            # precision, recall, f_score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary',
            #                                                                         pos_label=pos_label)
            # res = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f_score}
            #
            # print(f"Accuracy:{accuracy}, "
            #       f"Precision:{precision}, "
            #       f"Recall:{recall}, "
            #       f"F-score:{f_score}, "
            #       f"\nconfusion-matrix: {confusion_matrix(y_true, y_pred)}")
            #
            # # switch back to train mode
            # self.model.train()

            # return res, test_z, test_labels, combined_energy

    def weighted_log_sum_exp(self, x, weights, dim):
        """
        Inspired by https://discuss.pytorch.org/t/moving-to-numerically-stable-log-sum-exp-leads-to-extremely-large-loss-values/61938

        Parameters
        ----------
        x
        weights
        dim

        Returns
        -------

        """
        m, idx = torch.max(x, dim=dim, keepdim=True)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m) * (weights.unsqueeze(2)), dim=dim))

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`

        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`

        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]
        K = gamma.shape[1]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        # Covariance (K x D x D)
        covs = []
        for i in range(0, K):
            xm = z - mu[i]
            cov = 1 / gamma_sum[i] * ((gamma[:, i].unsqueeze(-1) * xm).T @ xm)
            cov += 1e-12
            covs.append(cov)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        self.covs = covs
        # self.cov_mat = covs

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        eps = 1e-12
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def score(self, sample: torch.Tensor):
        pass
