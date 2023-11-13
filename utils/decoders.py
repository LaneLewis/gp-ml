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

class ParabolaDecoder():
    def __init__(self,dims_in,alpha):
        self.dims_in = dims_in
        self.alpha = torch.tensor(alpha)
    def forward_single(self,z):
        '''takes in a two dimensional latent variable and embeds it into a 3d parabola
        z - tensor of shape [samples, timesteps, dims_in]
        returns X a tensor of shape [samples, timesteps, dims_in+1]
        '''
        extra_dim = self.alpha*torch.linalg.norm(z,dim=2)**2
        return torch.cat((z,extra_dim.unsqueeze(2)),dim=2)
    def forward(self,Z):
        '''takes in a two dimensional latent variable and embeds it into a 3d parabola
        Z - tensor of shape [batch_size,samples, timesteps, dims_in]
        returns X a tensor of shape [batch_size, samples, timesteps, dims_in+1]
        '''
        return torch.vmap(self.forward_single)(Z)
    
    def parameters(self):
        return [self.alpha]