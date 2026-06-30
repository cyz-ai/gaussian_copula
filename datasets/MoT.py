import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def _triangle_pdf(x, center, half_width):
    """Symmetric triangular PDF centered at `center` with half-width `h`.

    PDF = (1/h) * max(0, 1 - |x - center| / h)
    Support: [center - h, center + h]. Peak: 1/h. Integrates to 1.

    Implements the triangle kernel from Pichler et al., "A Differential Entropy
    Estimator for Training Neural Networks", ICML 2022, Appendix A.2.
    """
    return (1.0 / half_width) * np.maximum(0, 1.0 - np.abs(x - center) / half_width)


def _triangle_log_pdf(x, center, half_width, eps=1e-40):
    """Log of the triangle PDF, with eps floor to avoid log(0)."""
    return np.log(_triangle_pdf(x, center, half_width) + eps)


def _triangle_sample(center, half_width, n):
    """Sample from a symmetric triangular distribution via inverse CDF.

    Triangular(a, c, b) with a = center - h, c = center, b = center + h.
    """
    a = center - half_width
    b = center + half_width
    u = np.random.uniform(0, 1, n)
    return np.where(
        u < 0.5,
        a + half_width * np.sqrt(2 * u),
        b - half_width * np.sqrt(2 * (1 - u)),
    )


def _triangle_entropy(half_width):
    """Differential entropy of a symmetric triangular distribution.

    For Triangular(a, c, b) with b - a = 2h: H = 1/2 + ln(h).
    """
    return 0.5 + np.log(half_width)


class MoT(Dataset):
    """Mixture of Triangles dataset.

    Each 1D component is a symmetric triangular distribution with random
    center, scale, and mixture weight. For d > 1, the multivariate distribution
    is formed from d i.i.d. copies of the 1D mixture, giving c^d modes in
    d dimensions.

    Reference: Pichler et al., "A Differential Entropy Estimator for Training
    Neural Networks", ICML 2022, Section 3.1.2 and Appendix A.2.
    """

    def __init__(self, n_samples=10000, n_dims=1, n_components=2,
                 weights=None, centers=None, half_widths=None, seed=None):
        """
        Args:
            n_samples: number of samples to generate
            n_dims: dimensionality (d i.i.d. copies of the 1D mixture)
            n_components: number of mixture components (c)
            weights: mixture weights, shape (c,). Default: random from Dirichlet(1,...,1)
            centers: component centers, shape (c,). Default: random in [-5, 5]
            half_widths: triangle half-widths, shape (c,). Default: random in [0.5, 5.0]
            seed: random seed for reproducibility of parameters and data
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_dims = n_dims
        self.n_components = n_components

        # Random parameters as in the paper (Pichler et al., ICML 2022)
        if weights is None:
            weights = np.random.dirichlet(np.ones(n_components))
        if centers is None:
            centers = np.random.uniform(-5, 5, n_components)
        if half_widths is None:
            half_widths = np.random.uniform(0.5, 5.0, n_components)

        self.weights = np.asarray(weights, dtype=np.float64)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.half_widths = np.asarray(half_widths, dtype=np.float64)

        self.data = self.sample_data(n_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample_data(self, n_samples):
        """Sample from the d-dimensional mixture of triangles (d i.i.d. copies)."""
        samples = np.zeros((n_samples, self.n_dims), dtype=np.float32)
        for dim in range(self.n_dims):
            # Pick component for each sample
            component_idx = np.random.choice(self.n_components, size=n_samples, p=self.weights)
            for k in range(self.n_components):
                mask = component_idx == k
                count = mask.sum()
                if count > 0:
                    samples[mask, dim] = _triangle_sample(
                        self.centers[k], self.half_widths[k], count
                    )
        return torch.from_numpy(samples)

    def log_probs_1d(self, x):
        """Log probability of 1D data under the mixture.

        Args:
            x: numpy array, shape (n,)
        Returns:
            log p(x), shape (n,)
        """
        # log p(x) = log sum_k w_k * tri_k(x)
        log_components = np.zeros((len(x), self.n_components))
        for k in range(self.n_components):
            log_components[:, k] = (np.log(self.weights[k])
                                    + _triangle_log_pdf(x, self.centers[k], self.half_widths[k]))
        # logsumexp across components
        max_log = log_components.max(axis=1, keepdims=True)
        log_probs = max_log.squeeze() + np.log(np.exp(log_components - max_log).sum(axis=1))
        return log_probs

    def log_probs(self, inputs):
        """Log probability of d-dimensional data (product of d i.i.d. 1D mixtures).

        Args:
            inputs: torch Tensor, shape (n, d)
        Returns:
            log p(x), torch Tensor, shape (n,)
        """
        x = inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else np.asarray(inputs)
        total_log_prob = np.zeros(len(x))
        for dim in range(self.n_dims):
            total_log_prob += self.log_probs_1d(x[:, dim])
        return torch.from_numpy(total_log_prob).float()

    def entropy(self, n_mc=100000):
        """Estimate entropy via Monte Carlo: H = -E[log p(x)].

        Args:
            n_mc: number of MC samples
        Returns:
            entropy estimate (float)
        """
        samples = self.sample_data(n_mc)
        return -self.log_probs(samples).mean().item()

    def plot_samples(self, X, i=0, j=1, title="MoT Samples in 2D"):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.scatter(X[:, i], X[:, j], alpha=0.5, s=10)
        plt.xlabel(f"dim {i}")
        plt.ylabel(f"dim {j}")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_pdf_1d(self, ax=None, n_grid=1000):
        """Plot the 1D mixture PDF and per-component PDFs."""
        if self.n_dims != 1:
            print("plot_pdf_1d only works for 1D mixtures.")
            return
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        # Determine plot range from component supports
        x_min = (self.centers - self.half_widths).min() - 0.5
        x_max = (self.centers + self.half_widths).max() + 0.5
        x_grid = np.linspace(x_min, x_max, n_grid)
        # Total PDF
        total_pdf = np.zeros(n_grid)
        for k in range(self.n_components):
            comp_pdf = self.weights[k] * _triangle_pdf(x_grid, self.centers[k], self.half_widths[k])
            ax.plot(x_grid, comp_pdf, '--', alpha=0.5, label=f'Component {k+1}')
            total_pdf += comp_pdf
        ax.plot(x_grid, total_pdf, 'k-', linewidth=2, label='Mixture')
        ax.set_xlabel('x')
        ax.set_ylabel('p(x)')
        ax.legend()
        ax.set_title('Mixture of Triangles PDF')
        plt.tight_layout()
