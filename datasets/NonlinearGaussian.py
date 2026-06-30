import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal, norm, t
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class NonlinearGaussian(Dataset):
    """Nonlinearly transformed Gaussian dataset with known MI."""

    def __init__(self, n_samples=100000, n_dims=80, rho=0.80, mu=0, case=0):
        self.case = case
        self.n_dims = n_dims
        self.mu = np.zeros(self.n_dims) + mu
        self.rho = rho
        self.cov_matrix = block_diag(*[[[1, self.rho], [self.rho, 1]] for _ in range(n_dims // 2)])
        self.data = self._sample_gaussian(n_samples, self.cov_matrix).astype(dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _sample_gaussian(self, n_samples, cov_matrix):
        mvn = multivariate_normal(mean=self.mu, cov=cov_matrix)
        return mvn.rvs(n_samples)

    @staticmethod
    def _get_rho_from_mi(mi, n_dims):
        """Analytically calculate correlation coefficient from MI value."""
        return (1 - np.exp(-4 * mi / n_dims)) ** 0.5

    @staticmethod
    def _get_mi_from_rho(rho, n_dims):
        """Analytically calculate mutual information from correlation value."""
        return -n_dims / 4.0 * np.log(1 - rho**2)

    @staticmethod
    def u2xy(u):
        return u[:, ::2], u[:, 1::2]

    @staticmethod
    def xy2u(X, Y):
        n, d = X.shape
        samples = np.zeros((n, d * 2))
        samples[:, ::2], samples[:, 1::2] = X, Y
        return samples

    def _numerator_log_prob(self, u):
        mvn = multivariate_normal(mean=self.mu, cov=self.cov_matrix)
        return mvn.logpdf(u)

    def _denominator_log_prob(self, u):
        mvn = multivariate_normal(mean=self.mu, cov=np.eye(self.n_dims))
        return mvn.logpdf(u)

    def sample_data(self, n_samples, mode='joint'):
        cov = self.cov_matrix if mode == 'joint' else np.eye(self.n_dims)
        data = self._sample_gaussian(n_samples, cov)
        X = torch.Tensor(data[:, ::2]).clone()
        Y = torch.Tensor(data[:, 1::2]).clone()
        return X, Y

    def log_ratio(self, X, Y):
        """Return log p(x, y) / p(x)p(y)."""
        samples = self.xy2u(X, Y)
        return self._numerator_log_prob(samples) - self._denominator_log_prob(samples)

    def true_mutual_info(self):
        return self._get_mi_from_rho(self.rho, self.n_dims)

    def empirical_mutual_info(self):
        samples = self._sample_gaussian(100000, self.cov_matrix)
        return np.mean(self._numerator_log_prob(samples) - self._denominator_log_prob(samples))

    def transformation(self, x, y):
        """Apply nonlinear transformation to make MI estimation harder."""
        case = self.case
        if case == '0':                                             # identity
            pass
        elif case == '1a':                                          # tanh(x), exp(y)
            x, y = torch.tanh(x), torch.exp(y)
        elif case == '1b':                                          # x^3, y^3
            x, y = x**3, y**3
        elif case == '1c':                                          # sign(x)x^2, sign(y)y^2
            x, y = torch.sign(x)*x**2, torch.sign(y)*y**2
        elif case == '1d':                                          # 3x, y/2
            x, y = 3*x, y/2
        elif case == '2':                                           # Ax, By (linear mixing)
            d = x.size(1)
            A = torch.ones(d, d).tril()
            B = torch.ones(d, d).tril()
            A, B = A/A.sum(dim=1, keepdim=True), B/B.sum(dim=1, keepdim=True)
            self.A, self.B = A, B
            x, y = x @ A, y @ B
        elif case == '3a':                                          # A @ tanh(x), B @ exp(y)
            d = x.size(1)
            x, y = torch.tanh(x), torch.exp(y)
            A = torch.ones(d, d).tril()
            B = torch.ones(d, d).tril()
            A, B = A/A.sum(dim=1, keepdim=True), B/B.sum(dim=1, keepdim=True)
            self.A, self.B = A, B
            x, y = x @ A, y @ B
        elif case == '3b':                                          # A @ x^3, B @ exp(y)
            d = x.size(1)
            x, y = x**3, torch.exp(y)
            A = torch.ones(d, d).tril()
            B = torch.ones(d, d).tril()
            self.A, self.B = A, B
            x, y = x @ A, y @ B
        elif case == '3c':                                          # A @ sign(x)x^2, B @ sign(y)y^2
            d = x.size(1)
            x, y = torch.sign(x)*x**2, torch.sign(y)*y**2
            A = torch.ones(d, d).tril()
            B = torch.ones(d, d).tril()
            self.A, self.B = A, B
            x, y = x @ A, y @ B
        elif case == '3d':                                          # student-t + linear mixing
            d = x.size(1)
            v = 3
            gaussian_cdf_x = norm.cdf(x.cpu().numpy())
            gaussian_cdf_y = norm.cdf(y.cpu().numpy())
            x = torch.Tensor(t.ppf(gaussian_cdf_x, df=v)).float()
            y = torch.Tensor(t.ppf(gaussian_cdf_y, df=v)).float()
            A = torch.ones(d, d).tril()
            B = torch.ones(d, d).tril()
            self.A, self.B = A, B
            x, y = x @ A, y @ B
        self.mu_x, self.mu_y = x.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)
        self.std_x, self.std_y = x.std(dim=0, keepdim=True), y.std(dim=0, keepdim=True)
        return x, y

    def log_prob(self, X, Y):
        case = self.case
        if case == '0':
            eps_x, eps_y = X, Y
            log_dxde = (0*X).sum(dim=1)
            log_dyde = (0*Y).sum(dim=1)
        elif case == '1a':
            eps_x, eps_y = torch.atanh(X), torch.log(Y)
            log_dxde = torch.log(1-torch.tanh(eps_x)**2).sum(dim=1)
            log_dyde = eps_y.sum(dim=1)
        elif case == '1b':
            eps_x = torch.sign(X)*X.abs().pow(1/3)
            eps_y = torch.sign(Y)*Y.abs().pow(1/3)
            log_dxde = torch.log(3*eps_x.abs().pow(2)).sum(dim=1)
            log_dyde = torch.log(3*eps_y.abs().pow(2)).sum(dim=1)
        elif case == '1c':
            eps_x = torch.sign(X)*X.abs().sqrt()
            eps_y = torch.sign(Y)*Y.abs().sqrt()
            log_dxde = torch.log(2*eps_x.abs()).sum(dim=1)
            log_dyde = torch.log(2*eps_y.abs()).sum(dim=1)
        elif case == '1d':
            eps_x, eps_y = X/3, Y*2
            log_dxde = (0*X + np.log(3)).sum(dim=1)
            log_dyde = (0*Y + np.log(0.5)).sum(dim=1)
        elif case == '2':
            eps_x = X.cpu() @ self.A.inverse()
            eps_y = Y.cpu() @ self.B.inverse()
            log_dxde = self.A.logdet().repeat(len(X))
            log_dyde = self.B.logdet().repeat(len(X))
        elif case == '3a':
            log_dxde1 = self.A.logdet().repeat(len(X))
            log_dyde1 = self.B.logdet().repeat(len(X))
            x = X.cpu() @ self.A.inverse()
            y = Y.cpu() @ self.B.inverse()
            eps_x, eps_y = torch.atanh(x), torch.log(y)
            log_dxde = log_dxde1 + torch.log(1-torch.tanh(eps_x)**2).sum(dim=1)
            log_dyde = log_dyde1 + eps_y.sum(dim=1)
        elif case == '3b':
            x = X.cpu() @ self.A.inverse()
            y = Y.cpu() @ self.B.inverse()
            eps_x = torch.sign(x)*x.abs().pow(1/3)
            eps_y = torch.log(y + 1e-30)
            log_dxde = torch.log(3*eps_x.abs().pow(2)).sum(dim=1)
            log_dyde = eps_y.sum(dim=1)
        elif case == '3c':
            x = X.cpu() @ self.A.inverse()
            y = Y.cpu() @ self.B.inverse()
            eps_x = torch.sign(x)*x.abs().sqrt()
            eps_y = torch.sign(y)*y.abs().sqrt()
            log_dxde = torch.log(2*eps_x.abs()).sum(dim=1)
            log_dyde = torch.log(2*eps_y.abs()).sum(dim=1)
        elif case == '3d':
            v = 3
            x = X.cpu() @ self.A.inverse()
            y = Y.cpu() @ self.B.inverse()
            # Invert student-t: t -> uniform -> Gaussian
            eps_x = torch.Tensor(norm.ppf(t.cdf(x.numpy(), df=v))).float()
            eps_y = torch.Tensor(norm.ppf(t.cdf(y.numpy(), df=v))).float()
            # log |dx/deps| = log |det(A)| + log(t_pdf / norm_pdf) per dim
            log_dxde_linear = self.A.logdet().repeat(len(X))
            log_dyde_linear = self.B.logdet().repeat(len(X))
            from scipy.stats import t as t_dist
            log_t_pdf_x = torch.Tensor(t_dist.logpdf(x.numpy(), df=v)).sum(dim=1)
            log_t_pdf_y = torch.Tensor(t_dist.logpdf(y.numpy(), df=v)).sum(dim=1)
            log_norm_pdf_x = torch.Tensor(norm.logpdf(eps_x.numpy())).sum(dim=1)
            log_norm_pdf_y = torch.Tensor(norm.logpdf(eps_y.numpy())).sum(dim=1)
            log_dxde = log_dxde_linear + log_t_pdf_x - log_norm_pdf_x
            log_dyde = log_dyde_linear + log_t_pdf_y - log_norm_pdf_y
        else:
            raise ValueError(f"Unknown case: {case}")
        u = self.xy2u(eps_x.cpu().numpy(), eps_y.cpu().numpy())
        log_dxy_du = log_dxde + log_dyde
        log_pu = self._numerator_log_prob(u)
        return log_pu - log_dxy_du.cpu().numpy()

    def entropy(self, XY):
        n, d = XY.size()
        X, Y = XY[:, 0:d//2], XY[:, d//2:]
        return -self.log_prob(X, Y).mean()

    def plot_samples(self, X, i, j, title="Samples in 2D"):
        plt.figure(figsize=(5, 5))
        plt.scatter(X[:, i], X[:, j], alpha=0.5, s=10)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.title(title)
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
