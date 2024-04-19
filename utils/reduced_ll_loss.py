import torch
from typing import Callable

def sample_prior(timesteps,Ks,samples,batch_size):
    '''
    returns prior_samples - shape[batch_size,samples,timesteps,latent_dims]:
                                samples from the prior
    '''
    prior_samples = torch.distributions.MultivariateNormal(torch.zeros(timesteps),Ks.permute(2,0,1)).rsample((batch_size,samples))
    return prior_samples.permute(0,1,3,2)

def approx_data_log_likelihood(timesteps:torch.Tensor,batch_X:torch.Tensor,decoding_function:Callable,
                     R:torch.Tensor,Ks:torch.tensor,samples:int=10,device="cpu"):
    '''
    returns approx_individual_log_likelihood - shape[batch_size]
    '''
    batch_size = batch_X.shape[0]
    prior_dist_samples = sample_prior(timesteps,Ks,samples,batch_size)
    decoding_manifold_means = decoding_function(prior_dist_samples)
    other_loss = ll_loss(batch_X,decoding_manifold_means,R)
    log_likelihood_sample_losses = other_loss
    log_inv_samples = torch.log(torch.tensor(1/samples,requires_grad=False,device=device))
    return torch.logsumexp(log_likelihood_sample_losses,dim=1) + log_inv_samples

def approx_batch_log_likelihood_loss(timesteps:torch.Tensor,batch_X:torch.Tensor,decoding_function:Callable,
                     R:torch.Tensor,Ks:torch.tensor,samples:int=10,device="cpu"):
    return torch.sum(approx_data_log_likelihood(timesteps,batch_X,decoding_function,R,Ks,samples,device=device))/batch_X.shape[0]

def ll_loss(batch_X,decoding_manifold_means,R):
    data_dist = torch.distributions.MultivariateNormal(decoding_manifold_means.permute(1,0,2,3),R)
    #testing
    data_prob = data_dist.log_prob(batch_X)
    summed_over_time = torch.sum(data_prob,dim=2)
    return summed_over_time.permute(1,0)