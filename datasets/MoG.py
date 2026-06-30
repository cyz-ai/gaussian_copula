import torch
import torch.distributions as distribution
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MoG(Dataset):
    """Mixture of Gaussians dataset."""

    def __init__(self, n_samples=100000, n_dims=80, coeff_array=None, mu_array=None, cov_array=None):
        self.dim = n_dims
        self.K = len(mu_array)
        self.coeff_array = torch.Tensor(coeff_array)
        self.mu_array = mu_array
        self.cov_array = cov_array
        self.data = self.sample_data(n_samples=n_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def entropy(self):
        """Estimate entropy via Monte Carlo."""
        data = self.sample_data(n_samples=50000)
        return -self.log_probs(data).mean().item()

    def sample_data(self, n_samples=1):
        coeff, mu_array, C_array = self.coeff_array, self.mu_array, self.cov_array
        # Sample component indices all at once
        categorical = distribution.Categorical(coeff)
        indices = categorical.sample((n_samples,))
        # Sample from each component
        samples = []
        for k in range(self.K):
            mask = (indices == k)
            count = mask.sum().item()
            if count > 0:
                normal = distribution.MultivariateNormal(mu_array[k], C_array[k])
                samples.append(normal.sample((count,)))
        data = torch.cat(samples, dim=0)
        # Shuffle to mix components
        perm = torch.randperm(data.size(0))
        return data[perm]

    def log_probs(self, inputs):
        """Log probability under the MoG: log sum_k coeff[k] * N(x; mu[k], cov[k])."""
        n = inputs.size(0)
        coeff, mu_array, C_array = self.coeff_array, self.mu_array, self.cov_array
        log_probs = []
        for k in range(self.K):
            mu = mu_array[k].view(-1)
            V = C_array[k].view(self.dim, self.dim)
            normal = distribution.MultivariateNormal(mu, V)
            log_prob = normal.log_prob(inputs) + coeff[k].log()
            log_probs.append(log_prob.view(n, 1))
        return torch.cat(log_probs, dim=1).logsumexp(dim=1).view(-1)

    def log_probs_marginal(self, inputs, marginals):
        """Log probability of a subset of dimensions (marginals)."""
        n = inputs.size(0)
        coeff, mu_array, C_array = self.coeff_array, self.mu_array, self.cov_array
        log_probs = []
        for k in range(self.K):
            mu = mu_array[k].view(-1)[marginals]
            V = C_array[k].view(self.dim, self.dim)[marginals, :][:, marginals]
            normal = distribution.MultivariateNormal(mu, V)
            log_prob = normal.log_prob(inputs[:, marginals]) + coeff[k].log()
            log_probs.append(log_prob.view(n, 1))
        return torch.cat(log_probs, dim=1).logsumexp(dim=1).view(-1)

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
