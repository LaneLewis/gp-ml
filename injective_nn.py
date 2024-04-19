import torch
import torch.nn as nn
import math
from torch.func import jacrev
class SquareLeakyReLU(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha = alpha
        self.relu = nn.LeakyReLU(alpha)
    def forward(self,x):
        return torch.square(self.relu(x))
    
class PositiveLinear(nn.Module):
    #from ghub
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())
    
class ConvexLayer(nn.Module):
    def __init__(self,input_size,output_size,activation,first_layer=False,layer_minus_1_size=None):
        #wasnt sure the proper way to init bias here
        if not first_layer:
            assert not (layer_minus_1_size is None)
        super().__init__()
        self.first_layer = first_layer
        self.input_u_w = PositiveLinear(input_size,output_size)
        if not first_layer:
            self.input_z_w = PositiveLinear(layer_minus_1_size,output_size)
        self.activation = activation
        self.bias = nn.Parameter(torch.Tensor(output_size))
        std = 1/math.sqrt(output_size)
        self.bias.data.uniform_(-std,std)
    def forward(self,initial_input, layer_minus_1=None):
        if not self.first_layer:
            linear_sum = self.input_u_w(initial_input) + self.input_z_w(layer_minus_1) + self.bias
        else:
            linear_sum = self.input_u_w(initial_input) + self.bias
        return self.activation(linear_sum)

class BrenierMapNN(nn.Module):
    def __init__(self,hidden_layer_sizes,input_size,alpha=0.2):
        super().__init__()
        self.layers = []
        for index,layer_size in enumerate(hidden_layer_sizes):
            if index == 0:
                l = ConvexLayer(input_size,layer_size,activation=SquareLeakyReLU(alpha),first_layer=True)
            else:
                l = ConvexLayer(input_size,layer_size,activation=nn.LeakyReLU(alpha),first_layer=False,layer_minus_1_size=hidden_layer_sizes[index-1])
            self.layers.append(l)
        final_layer = ConvexLayer(input_size,1,nn.LeakyReLU(alpha),first_layer=False,layer_minus_1_size=layer_size)
        self.layers.append(final_layer)
        self.layers = nn.ModuleList(self.layers)
    def forward_convex(self,u):
        z = u
        for index, l in enumerate(self.layers):
            if index == 0:
                z = l(u)
            else:
                z = l(u, z)
        return z
    def forward(self, u):
        return batched_jacobian(self.forward_convex,u)

def batched_jacobian(func,inputs):
    return torch.squeeze(torch.vmap(jacrev(func))(inputs),dim=1)

class ConvexFeedforward(nn.Module):
    def __init__(self, first_map_layer_sizes, second_map_layer_sizes,latent_dims,observed_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.first_map_layer_sizes = first_map_layer_sizes
        self.second_map_layer_sizes = second_map_layer_sizes
        self.first_map = BrenierMapNN(first_map_layer_sizes,latent_dims)
        self.second_map = BrenierMapNN(second_map_layer_sizes,observed_dims)        
        self.betas = torch.concat([torch.eye(latent_dims,requires_grad=False),torch.zeros(observed_dims - latent_dims,latent_dims,requires_grad=False)])

    def forward(self,latent_input):
        collapsed = math.prod(latent_input.shape[0:-1])
        batch_collapsed_input = latent_input.reshape((collapsed,latent_input.shape[-1]))
        out_shape = list(latent_input.shape)
        out_shape[-1] = self.observed_dims
        v = self.first_map.forward(batch_collapsed_input)
        v = torch.matmul(self.betas,v.T).T
        v = self.second_map.forward(v)
        out_v = v.reshape(out_shape)
        return out_v

#latent_dims = 2
#batch_size = 6
#observed_dims = 3

#tester = torch.rand((batch_size,100,latent_dims))
#p = ConvexFeedforward([10,10],[15,15],latent_dims,observed_dims)
#p.forward(tester)




    
