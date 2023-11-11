import torch
class ParabolaDecoder():
    def __init__(self,dims_in,alpha):
        self.dims_in = dims_in
        self.alpha = torch.tensor(alpha)
    def forward_single(self,z):
        '''takes in a two dimensional latent variable and embeds it into a 3d parabola
        z - tensor of shape [samples, timesteps, dims_in]
        returns X a tensor of shape [samples, timesteps, dims_in+1]
        '''
        extra_dim = self.alpha*torch.linalg.norm(z)**2
        return torch.cat((z,extra_dim.unsqueeze(2)),dim=2)
    def forward(self,Z):
        '''takes in a two dimensional latent variable and embeds it into a 3d parabola
        Z - tensor of shape [batch_size,samples, timesteps, dims_in]
        returns X a tensor of shape [batch_size, samples, timesteps, dims_in+1]
        '''
        torch.vmap(self.forward_single)(Z)
    def parameters(self):
        return [self.alpha]