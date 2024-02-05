import torch
from typing import Callable
from utils.elbo_loss import log_likelihood_loss
from torch import nn
from torch.nn.modules import Module

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
    #shape [batch_size,samples,timesteps,latent_dims]
    prior_dist_samples = sample_prior(timesteps,Ks,samples,batch_size)
    decoding_manifold_means = decoding_function(prior_dist_samples)
    log_likelihood_sample_losses = ll_loss(batch_X,decoding_manifold_means,R)
    #print(other_loss - log_likelihood_sample_losses)
    #print(f"ll {log_likelihood_sample_losses.shape}")
    log_inv_samples = torch.log(torch.tensor(1/samples,requires_grad=False,device=device))
    return torch.logsumexp(log_likelihood_sample_losses,dim=1) + log_inv_samples
    #have to convert to likelihood before sum
    #approx_individual_log_likelihood = torch.mean(log_likelihood_sample_losses,dim=1)
    #return approx_individual_log_likelihood

def approx_batch_log_likelihood_loss(timesteps:torch.Tensor,batch_X:torch.Tensor,decoding_function:Callable,
                     R:torch.Tensor,Ks:torch.tensor,samples:int=10,device="cpu"):
    return torch.sum(approx_data_log_likelihood(timesteps,batch_X,decoding_function,R,Ks,samples,device=device))/batch_X.shape[0]

def ll_loss(batch_X,decoding_manifold_means,R):
    
    #decoding_manifold_means shape [batchsize, samples, timesteps, observed_dims]
    #print(decoding_manifold_means[0,:,:,:].mean(dim=0))
    data_dist = torch.distributions.MultivariateNormal(decoding_manifold_means.permute(1,0,2,3),R)
    #testing
    data_prob = data_dist.log_prob(batch_X)
    summed_over_time = torch.sum(data_prob,dim=2)
    return summed_over_time.permute(1,0)

class NNBBOptimizer(nn.Module):
    def __init__(self,decoder_nn, observed_dims,timesteps, taus, R_diag,layers_list,device="cpu"):
        super().__init__()
        nn_parameter_inputs = torch.cat([torch.flatten(p) for p in decoder_nn.parameters()],)
        self.total_parameter_inputs = torch.cat((nn_parameter_inputs,taus,R_diag))
        self.total_parameter_size = self.total_parameter_inputs.shape[0]
        self.data_size = observed_dims*timesteps
        total_input_size = self.data_size + self.total_parameter_size

        self.layers = nn.ModuleList()
        self.input_dim = total_input_size
        input_size = total_input_size
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.layers.append(nn.Linear(input_size,1))
        self.to(device)
    
    def forward(self,batch_X,decoder_nn_params,taus,R_diag):
        nn_parameter_inputs = torch.cat([torch.flatten(p) for p in decoder_nn_params])

        total_parameter_inputs = torch.cat((nn_parameter_inputs,taus,R_diag))
        batch_size = batch_X.shape[0]
        batched_parameter_inputs = torch.outer(torch.ones(batch_size),total_parameter_inputs)
        input_data = torch.concat((torch.flatten(batch_X,start_dim=1),batched_parameter_inputs),dim=1)
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    
def deactivate_model(model):
    for param in model.parameters():
        param.requires_grad = False
def activate_model(model):
    for param in model.parameters():
        param.requires_grad = True
