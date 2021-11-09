import torch
import numpy as np


def multivariate_normal_pdf(X, phi, mu, Sigma, scaling=False, device='cuda'):
    d, det, scaler = len(X), 1, 1
    if X.ndim > 1:
        d = X.shape[1]
    if scaling:
        if Sigma.ndim > 1 and Sigma.shape[1] > 1:
            det = torch.sqrt(torch.linalg.det(Sigma))
        two_pi_power = np.power(2. * np.pi, d / 2.)
        scaler = (1. / (det * two_pi_power))
    Sigma_inv = (torch.eye(d) * 1e-6).to(device)
    exp_term = (torch.exp(-0.5 * (X - mu).T @ Sigma_inv @ (X - mu))).to(device)
    res = (phi * scaler * exp_term).to(device)
    return res.item()


def multivariate_normal_cholesky_pdf(X, phi, mu, Sigma):
    d, det = len(X), 1
    if X.ndim > 1:
        d = X.shape[1]
    L = torch.linalg.cholesky(Sigma)
    det = torch.prod(torch.diag(L))
    two_pi_pwr = torch.pow(2. * np.pi, d / 2.)
    term = (torch.linalg.inv(L) @ (X - mu).T) @ torch.linalg.inv(L) @ (X - mu)
    return phi * (1. / (det * two_pi_pwr) * torch.exp(-0.5 * term))


def estimate_GMM_params(X, gamma, device='cuda') -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Estimates GMM parameters :math:`\phi`, :math:`\mu` and :math:`\Sigma`
    Parameters
    ----------
    X: Samples
    gamma: Output of a softmax

    Returns
    -------
    :math:`\phi`: The mixture component
    :math:`\mu`: The mean vector
    :math:`\Sigma`: The covariance matrix
    """
    # K: number of mixtures
    # D: dimensionality
    # N: number of inputs
    K, D, N = gamma.shape[1], X.shape[1], X.shape[0]
    phi = torch.mean(gamma, dim=0).to(device)
    mu = torch.zeros([K, D]).to(device)
    Sigma = torch.zeros([K, D, D]).to(device)
    # TODO: replace loops by matrix operations
    for k in range(0, K):
        mu_tmp = torch.zeros([D]).to(device)
        sig_tmp = torch.zeros([D, D]).to(device)
        for i in range(0, N):
            mu_tmp = mu_tmp + gamma[i, k] * X[i, :]
        mu[k, :] = (mu_tmp / N).to(device)
        for i in range(0, N):
            sig_tmp = sig_tmp + gamma[i, k] * ((X[i, :] - mu[k, :]).T @ X[i, :] - mu[k, :])
        Sigma[k, :, :] = (sig_tmp / N).to(device)

    return phi, mu, Sigma
