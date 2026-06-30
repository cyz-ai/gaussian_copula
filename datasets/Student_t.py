import numpy as np
from scipy.special import digamma, gamma, gammaln, psi
from scipy.stats import multivariate_normal
from numpy.linalg import det


def _differential_entropy(k, dof):
    """Differential entropy of a Student-t(0, I_k, dof).

    See Eq. (7) of Arellano-Valle, Contreras-Reyes, Genton (2013),
    "Shannon Entropy and Mutual Information for Multivariate
    Skew-Elliptical Distributions", Scand. J. Stat., vol. 40, pp. 46-47.
    """
    half_sum = 0.5 * (dof + k)
    digamma_term = half_sum * (digamma(half_sum) - digamma(0.5 * dof))
    log_term = -np.log(gamma(half_sum)) + np.log(gamma(0.5 * dof)) + 0.5 * k * np.log(dof * np.pi)
    return log_term + digamma_term


def multivariate_t_sample(mu, sigma, dof, n):
    """Sample n points from a multivariate t-distribution.

    Args:
        mu: mean vector, shape (d,)
        sigma: scale matrix, shape (d, d)
        dof: degrees of freedom
        n: number of samples

    Returns:
        samples, shape (n, d)
    """
    d = len(sigma)
    g = np.tile(np.random.gamma(dof / 2, 2 / dof, n), (d, 1)).T
    z = np.random.multivariate_normal(np.zeros(d), sigma, n)
    return mu + z / np.sqrt(g)


class _Multinormal:
    """Auxiliary multivariate normal for entropy/MI calculations."""

    def __init__(self, mean, covariance):
        self._mean = np.asarray(mean)
        self._covariance = np.asarray(covariance)
        self._det_covariance = np.linalg.det(self._covariance)
        self._dim = self._mean.shape[0]
        if self._covariance.shape != (self._dim, self._dim):
            raise ValueError(
                f"Covariance shape {self._covariance.shape}, expected ({self._dim}, {self._dim})."
            )

    def sample(self, n_samples):
        return multivariate_normal.rvs(mean=self._mean, cov=self._covariance, size=n_samples)

    def entropy(self):
        """Entropy in nats."""
        return 0.5 * (np.log(self._det_covariance) + self._dim * (1 + np.log(2 * np.pi)))


class _SplitMultinormal:
    """Joint multivariate normal split into X and Y for MI calculation."""

    def __init__(self, dim_x, dim_y, covariance, mean=None):
        self.dim_total = dim_x + dim_y
        if mean is None:
            mean = np.zeros(self.dim_total)
        self._mean = np.asarray(mean)
        self._covariance = np.asarray(covariance)
        if self._mean.shape != (self.dim_total,):
            raise ValueError(f"Mean shape {self._mean.shape}, expected ({self.dim_total},).")
        if self._covariance.shape != (self.dim_total, self.dim_total):
            raise ValueError(f"Covariance shape {self._covariance.shape}, expected ({self.dim_total}, {self.dim_total}).")
        self._joint = _Multinormal(mean=self._mean, covariance=self._covariance)
        self._x = _Multinormal(mean=self._mean[:dim_x], covariance=self._covariance[:dim_x, :dim_x])
        self._y = _Multinormal(mean=self._mean[dim_x:], covariance=self._covariance[dim_x:, dim_x:])

    def mutual_information(self):
        """Exact MI via I(X;Y) = H(X) + H(Y) - H(X,Y)."""
        return max(0.0, self._x.entropy() + self._y.entropy() - self._joint.entropy())


class MultivariateStudentT:
    """Multivariate Student-t distribution with known MI.

    MI formula from Arellano-Valle, Contreras-Reyes, Genton (2013).
    MI = MI_normal(dispersion) + correction(dims, df).
    """

    def __init__(self, *, dim_x, dim_y, df, dispersion, mean=None):
        """
        Args:
            dim_x: dimension of X
            dim_y: dimension of Y
            df: degrees of freedom (positive; use np.inf for Gaussian)
            dispersion: scale matrix, shape (dim_x+dim_y, dim_x+dim_y).
                Note: dispersion != covariance. Covariance = df/(df-2) * dispersion for df > 2.
            mean: mean vector, shape (dim_x+dim_y,). Default: zeros.
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_total = dim_x + dim_y

        if mean is None:
            mean = np.zeros(self.dim_total)
        self._mean = np.asarray(mean)

        if df <= 0:
            raise ValueError("Degrees of freedom must be positive.")
        self._df = df

        self._dispersion = np.asarray(dispersion)

        # Used for the Gaussian part of the MI formula
        self._multinormal = _SplitMultinormal(
            dim_x=dim_x, dim_y=dim_y, mean=mean, covariance=dispersion
        )

    def sample(self, n_points):
        """Sample (X, Y) from the distribution, standardized to zero mean and unit variance."""
        xy = multivariate_t_sample(
            mu=self._mean, sigma=self._dispersion, dof=self._df, n=n_points
        )
        x, y = xy[:, :self.dim_x], xy[:, self.dim_x:]
        x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
        y = (y - y.mean(axis=0, keepdims=True)) / y.std(axis=0, keepdims=True)
        return x, y

    @property
    def df(self):
        return self._df

    def covariance(self):
        """Covariance matrix (only defined for df > 2)."""
        if self._df <= 2:
            raise ValueError(f"Covariance undefined for df={self._df} <= 2.")
        return self._df * self._dispersion / (self._df - 2.0)

    def mutual_information(self):
        """MI = MI_normal(dispersion) + correction(dims, df)."""
        return self._mi_normal() + self._mi_correction()

    def entropy(self):
        """Differential entropy of the multivariate Student-t."""
        V = self._dispersion
        nu = self._df
        d = V.shape[0]
        return (gammaln((nu + d) / 2) - gammaln(nu / 2)
                + 0.5 * np.log(det(V)) + (d / 2) * np.log(nu * np.pi)
                + ((nu + d) / 2) * (psi((nu + d) / 2) - psi(nu / 2)))

    def _mi_normal(self):
        """Gaussian part of the MI (using dispersion as if it were covariance)."""
        return self._multinormal.mutual_information()

    def _mi_correction(self):
        """Correction term depending only on dims and df."""
        h_x = _differential_entropy(k=self.dim_x, dof=self._df)
        h_y = _differential_entropy(k=self.dim_y, dof=self._df)
        h_xy = _differential_entropy(k=self.dim_total, dof=self._df)
        return h_x + h_y - h_xy
