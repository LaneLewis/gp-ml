import torch
import copy
from torch import nn , optim
from torch.nn.modules import Module

class FeedforwardNNDecoder(nn.Module):
    #NOTE this is completely untested!
    def __init__(self,layers_list,latent_dims,observed_dims,device="cpu"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = latent_dims
        input_size = latent_dims
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.layers.append(nn.Linear(input_size,observed_dims))
        self.to(device)

    def forward(self,input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    
class FeedforwardNNDecoderTau(nn.Module):
    #NOTE this is completely untested!
    def __init__(self,layers_list,latent_dims,observed_dims,device="cpu"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = 2*latent_dims
        input_size = 2*latent_dims
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.layers.append(nn.Linear(input_size,observed_dims))
        self.to(device)

    def forward(self,latent_data,taus):
        batch_size = latent_data.shape[0]
        samples = latent_data.shape[1]
        timesteps = latent_data.shape[2]
        latent_dims = latent_data.shape[3]
        tau_batch_sample_append = torch.ones((batch_size,samples,timesteps,latent_dims))*taus
        #taus_arr = torch.stack(batch_size*[torch.outer(torch.ones(samples,timesteps),taus)])
        latent_data_plus_tau = torch.concat([latent_data,tau_batch_sample_append],dim=3)
        input_data = latent_data_plus_tau
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    
class ParabolaDecoder():
    def __init__(self,alpha,device="cpu"):
        self.alpha = torch.tensor(alpha,requires_grad=True,device=device)
    def forward(self,Z):
        '''takes in a two dimensional latent variable and embeds it into a 3d parabola
        Z - tensor of shape [batch_size,samples, timesteps, dims_in]
        returns X a tensor of shape [batch_size, samples, timesteps, dims_in+1]
        '''
        extra_dim = torch.linalg.norm(Z,dim=-1)**2
        return torch.cat((Z,self.alpha*extra_dim.unsqueeze(-1)),dim=-1)
        #return torch.cat((self.alpha*Z+10.0,self.alpha*Z+10.0),dim=-1)
    def parameters(self):
        return [self.alpha]