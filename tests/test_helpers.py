import torch

    
class LinearEncoder():
    def __init__(self,dim_observed,dim_latents,weights_means,weights_cov):
        assert weights_means.shape[1] == dim_observed == weights_cov.shape[1]
        assert weights_cov.shape[0] == dim_latents == weights_cov.shape[0]
        self.dim_observed = dim_observed
        self.dim_latents = dim_latents
        self.weights_means = weights_means
        self.weights_cov = weights_cov

    def forward(self,X):
        '''X has shape [batch_size, timesteps, observed_dims]'''
        permuted_X = X.permute([2,0,1])
        means = torch.tensordot(self.weights_means,permuted_X,dims=1).permute([1,2,0])
        covs = torch.tensordot(self.weights_cov,permuted_X,dims=1)
        batch_create_diagonal = torch.diag_embed(covs)
        correct_diagonal = batch_create_diagonal.permute([1,2,3,0])
        return means,correct_diagonal
    
    def parameters(self):
        return torch.concat([self.weights_means.flatten(),self.weights_cov.flatten()])