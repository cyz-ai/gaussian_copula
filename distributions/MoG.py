import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from torch.utils.data import Dataset
import matplotlib.pyplot as plt



class MoG(Dataset):
    """"""


    def __init__(self, n_samples=100000, n_dims=80, coeff_array=None, mu_array=None, cov_array=None):
        """ """
        self.dim = n_dims
        self.K = len(mu_array)
        # self.x_dims_indices = x_dims_indices if x_dims_indices is not None else [i for i in range(n_dims//2)]
        # self.y_dims_indices = y_dims_indices if y_dims_indices is not None else [i+n_dims//2 for i in range(n_dims//2)]  
        self.coeff_array = torch.Tensor(coeff_array)
        self.mu_array = mu_array
        self.cov_array = cov_array
        
        self.data = self.sample_data(n_samples=n_samples)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    # def empirical_mutual_info(self):
    #     data = self.data
    #     X, Y = data[:, self.x_dims_indices], data[:, self.y_dims_indices]
    #     log_px, log_py = self.log_probs_marginal(data, self.x_dims_indices), self.log_probs_marginal(data, self.y_dims_indices)
    #     log_pxy = self.log_probs(data)
    #     mi = log_pxy.mean() - log_px.mean() - log_py.mean()
    #     return mi.item()



    '''
        compute the entropy of MoG model
    '''
    def entropy(self):
        data = self.sample_data(n_samples=50000)
        H = -self.log_probs(data).mean()
        return H.item()
    
    '''
        n_samples: number of data 
        return_in_paris: whether to break the data into X, Y pair, or treat them as a whole
    '''
    def sample_data(self, n_samples=1, return_in_pairs=False):
        coeff, mu_array, C_array = self.coeff_array, self.mu_array, self.cov_array
        categorical = distribution.Categorical(coeff)
        samples = []
        for i in range(n_samples):
            k = categorical.sample()    # pick a component
            mu, V = mu_array[k], C_array[k]
            normal = distribution.MultivariateNormal(mu, V)
            x = normal.sample().view(1, -1)
            samples.append(x)
        data = torch.cat(samples, dim=0)
        if return_in_pairs:
            return data[:, self.x_dims_indices], data[:, self.y_dims_indices]
        else:
            return data
            

    '''
        inputs: n*d data
    '''
    def log_probs(self, inputs):  
        n, d = inputs.size()
        # pdf = \sum coeff[k] * N(x; mu[k], cov[k])
        coeff, mu_array, C_array = self.coeff_array, self.mu_array, self.cov_array
        prob = torch.zeros(len(inputs)).to(inputs.device)
        normal = distribution.Normal(torch.tensor([0.0]).to(inputs.device), torch.tensor([1.0]).to(inputs.device))
        log_probs = []
        for k in range(self.K):   # <- pdf for each Gaussian component
            mu, C = mu_array[k], C_array[k]
            mu, C = mu.view(-1), C.view(self.dim, self.dim)
            V = C
            normal = distribution.MultivariateNormal(mu, V)
            log_prob = normal.log_prob(inputs)
            log_prob_with_weight = log_prob + coeff[k].log()
            log_probs.append(log_prob_with_weight.view(n, 1))
        log_probs = torch.cat(log_probs, dim=1)
        return log_probs.logsumexp(dim=1).view(-1)

    '''
        inputs: n*d data
        marginals: list, indices of marginals of interests (e.g. [0, 1, 2, 3, 5])
    '''
    def log_probs_marginal(self, inputs, marginals):  
        n, d = inputs.size()
        # pdf = \sum coeff[k] * N(x; mu[k], cov[k])
        coeff, mu_array, C_array = self.coeff_array, self.mu_array, self.cov_array
        prob = torch.zeros(len(inputs)).to(inputs.device)
        normal = distribution.Normal(torch.tensor([0.0]).to(inputs.device), torch.tensor([1.0]).to(inputs.device))
        log_probs = []
        for k in range(self.K):   # <- pdf for each Gaussian component
            mu, C = mu_array[k], C_array[k]
            mu, C = mu.view(-1), C.view(self.dim, self.dim)
            V = C
            mu2 = mu[marginals]
            V2 = V[marginals, :][:, marginals]
            inputs2 = inputs[:, marginals]
            normal = distribution.MultivariateNormal(mu2, V2)
            log_prob = normal.log_prob(inputs2)
            log_prob_with_weight = log_prob + coeff[k].log()
            log_probs.append(log_prob_with_weight.view(n, 1))
        log_probs = torch.cat(log_probs, dim=1)
        return log_probs.logsumexp(dim=1).view(-1)
    

    def plot_samples(self, X, i, j, title="Samples in 2D"):
        """
        u: np.ndarray of shape (n_samples, 2)
        """
        plt.figure(figsize=(5, 5))
        plt.scatter(X[:, i], X[:, j], alpha=0.5, s=10)
        plt.xlabel("u₁")
        plt.ylabel("u₂")
        plt.title(title)
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()