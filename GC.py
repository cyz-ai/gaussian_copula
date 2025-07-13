import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distribution
import math
import numpy as np
import time
from scipy import stats
import scipy.special as special
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import gaussian_kde



        
            

class GC(nn.Module):
    """ 
        Gaussian copula (non-parametric)
    """
    def __init__(self):
        super().__init__()

    '''
        learn the gaussian copula from data
    '''
    def learn(self, x):
        n, d = x.size()
        # calculate the latent Z
        z = self.forward(x)
        V = torch.matmul(z.t(), z)/(len(z)+1)
        A = torch.cholesky(V, upper=False)
        A_t_inv = torch.inverse(A.t())
        # construct latent correlation matrix
        self.V = V
        self.V2 = torch.eye(d).to(x.device)
        self.normal = distribution.multivariate_normal.MultivariateNormal(torch.zeros(d).to(x.device), self.V)
        self.normal2 = distribution.multivariate_normal.MultivariateNormal(torch.zeros(d).to(x.device), self.V2)
        # sort data (which will be reused during sampling)
        self.sorted_data, idx = torch.sort(x, dim=0)
        # learn marginals via KDE
        self.marginals = []
        x = x.cpu().numpy()
        for j in range(d):
            kde = gaussian_kde(x[:, j])
            self.marginals.append(kde)
        return 
    
    '''
        x -> z
    '''
    def forward(self, x):
        # calculate empirical CDF
        sorted_data, idx = torch.sort(x, dim=0)
        _, idx2 = torch.sort(idx, dim=0)
        u = (idx2.float()+1)/(len(x)+1)    
        zeros, ones = torch.zeros(x.size()).to(x.device), torch.ones(x.size()).to(x.device)
        normal = distribution.Normal(zeros, ones)
        # calculate the latent Z
        z = normal.icdf(u)
        n, d = z.size()
        return z

    '''
        sample x ~ GC
    '''        
    def sample(self, n=10000):
        # some preparation
        sorted_xy = self.sorted_data
        N, D = sorted_xy.size()
        # sample z ~ N(0, V)
        z = self.normal.rsample([n])   
        # convert z to u
        normal = distribution.Normal(torch.zeros(N, D).to(sorted_xy.device), torch.ones(N, D).to(sorted_xy.device))
        u = normal.cdf(z).clamp(0.00001, 0.99999)
        # convert u to idx
        idx = (N*u).long()
        # idx to x
        x = torch.zeros(N, D).to(sorted_xy.device)
        for d in range(D):
            idx_d = idx[:, d]
            sorted_x_d = sorted_xy[:, d]
            x_d = sorted_x_d[idx_d]
            x[:, d] = x_d
        rand_idx = torch.randperm(len(x))
        return x[rand_idx][0:n]
    
    '''
        calculate log pdf
    '''
    def log_probs(self, x):
        log_pdf_marginals = self.log_marginal_density(x)
        z = self.forward(x)
        log_copula_density = self.log_copula_density(z)
        return log_pdf_marginals + log_copula_density
    
    '''
        calculate copula density
    '''
    def log_copula_density(self, z):
        log_z_joint = self.normal.log_prob(z)
        log_z_marginal = self.normal2.log_prob(z)
        return log_z_joint - log_z_marginal
    
    '''
        calculate marginal density
    '''
    def log_marginal_density(self, x, eps=1e-40):
        n, d = x.size()
        device = x.device
        x = x.cpu().numpy()
        densities = np.zeros((n, d))
        for j in range(d):
            kde = self.marginals[j]
            densities[:, j] = kde.evaluate(x[:, j]) + eps
        return torch.Tensor(densities).log().sum(dim=1).to(device)
