import numpy as np
import torch
    
def get_batch_mvnormal(means, covs, cov_type='diag',device="cpu"):
    """return torch multivariate normal for sampling/prob evaluation"""
    if cov_type=='diag':
        cov_mat = torch.diag_embed(torch.exp(covs))
        batch_mvnormal = torch.distributions.MultivariateNormal(means,cov_mat)
    elif cov_type=='dense':
        batch_size, ns = covs.shape[0], means.shape[1]
        cov_mat = torch.zeros(batch_size,ns,ns)
        cov_mat[:,torch.tril(torch.ones(ns,ns))==1] = covs
        batch_mvnormal = torch.distributions.MultivariateNormal(means,scale_tril=cov_mat)
    elif cov_type=='scalar':
        batch_size, ns = covs.shape[0], means.shape[1]
        cov_mat = torch.exp(covs.to(device)).reshape(batch_size,1,1)*(torch.eye(ns).to(device))
        print(cov_mat.shape)
        batch_mvnormal = torch.distributions.MultivariateNormal(means.to(device),cov_mat)
    elif cov_type=='fixed':
        cov_mat = covs
        batch_mvnormal = torch.distributions.MultivariateNormal(means,cov_mat)
    return batch_mvnormal

def get_cov_mat(covs, ns, cov_type='diag',device="cpu"):
    if cov_type=='diag':
        cov_mat = torch.diag_embed(torch.exp(covs))
    elif cov_type=='dense':
        batch_size = covs.shape[0]
        cov_mat = torch.zeros(batch_size,ns,ns)
        cov_mat[:,torch.tril(torch.ones(ns,ns))==1] = covs
    elif cov_type=='scalar':
        batch_size = covs.shape[0]
        cov_mat = (torch.exp(covs)).reshape(batch_size,1,1)*(torch.eye(ns).to(device))
    elif cov_type=='fixed':
        cov_size = 1e-2
        cov_mat = cov_size*torch.eye(ns)
    return cov_mat
    
def log_transition_probs(means, covs, ops, cov_type='diag',device="cpu"):
    """given network outputs, calculates log probability for next observations"""
    batch_mvnormal = get_batch_mvnormal(means,covs,cov_type,device)    
    return batch_mvnormal.log_prob(ops.to(device))

def log_rew_probs(means, covs, rews):
    dists = torch.distributions.Normal(means,torch.exp(covs))
    return dists.log_prob(rews)

def product_of_gaussians(means,prec_mats):
    """return (normalized) Gaussian PDFs of the products of K Multivariate Gaussians of dim N
       inputs: means (of shape [,K,N]), and prec_mats (of shape [,K,N,N])"""
    prec_mat = torch.sum(prec_mats,1)
    mean = torch.matmul(torch.inverse(prec_mat),torch.sum(torch.matmul(prec_mats,means),1))
    return mean, prec_mat
    
def gaussian_product_posterior(mean_prior,prec_prior,new_mean,new_prec):
    """adds one more Gaussian to product of Gaussians. used for computing belief posterior online."""
    prec_posterior = prec_prior + new_prec
    mean_posterior = torch.matmul(torch.inverse(prec_posterior),torch.matmul(prec_prior,mean_prior) + torch.matmul(new_prec,new_mean))
    return mean_posterior, prec_posterior
    
    