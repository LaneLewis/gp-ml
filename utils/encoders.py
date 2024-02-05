import torch
from torch import nn
from torch.nn.modules import Module

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

class LSTM_Encoder(nn.Module):

    def __init__(self,device,latent_dims,observed_dims,
                 data_to_init_hidden_state_size=10,hidden_state_to_posterior_size=10,slack_term=0.00001):
        super().__init__()
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.data_to_hidden_layer = nn.LSTM(observed_dims,data_to_init_hidden_state_size,batch_first=True)
        self.hidden_layer_to_initial = nn.Linear(data_to_init_hidden_state_size,hidden_state_to_posterior_size)
        self.initial_to_posterior_states = nn.LSTM(1,hidden_state_to_posterior_size,batch_first=True)
        self.to_mean_sd = nn.Linear(hidden_state_to_posterior_size,2*self.latent_dims)
        self.slack = slack_term
        self.device = device

    def forward(self,X,taus):
        #X has shape [batch_size,timesteps,observed_dims]
        batch_size = X.shape[0]
        timesteps = X.shape[1]
        observed_dims = X.shape[2]
        X = X.float()
        #should have shape [hidden_dim_1]
        hidden_states,(_,_) = self.data_to_hidden_layer(X)
        last_hidden_states = hidden_states[:,-1,:]
        dynamics_initial_cond = self.hidden_layer_to_initial(last_hidden_states)
        dummy_inputs = torch.zeros((batch_size,timesteps,1)).float()
        dummy_initial_cxs =  torch.zeros(dynamics_initial_cond.shape).float()
        dynamics_hidden_states,_ = self.initial_to_posterior_states(dummy_inputs,(dynamics_initial_cond.reshape(1,batch_size,dynamics_initial_cond.shape[1]),dummy_initial_cxs.reshape(1,batch_size,dummy_initial_cxs.shape[1])))
        mean_sds = self.to_mean_sd(dynamics_hidden_states)
        means,sds = mean_sds[:,:,:self.latent_dims],mean_sds[:,:,self.latent_dims:]
        sds_tensor = torch.stack([torch.diag_embed(sds[batch_i,:,:].T**2 + self.slack) for batch_i in range(sds.shape[0])])
        corr_sds = sds_tensor.permute(0,2,3,1)
        return means,corr_sds
    
class LSTM_Encoder_tau(nn.Module):

    def __init__(self,device,latent_dims,observed_dims,
                 data_to_init_hidden_state_size=10,hidden_state_to_posterior_size=10,slack_term=0.00001):
        super().__init__()
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.data_to_hidden_layer = nn.LSTM(observed_dims+latent_dims,data_to_init_hidden_state_size,batch_first=True)
        self.hidden_layer_to_initial = nn.Linear(data_to_init_hidden_state_size,hidden_state_to_posterior_size)
        self.initial_to_posterior_states = nn.LSTM(1,hidden_state_to_posterior_size,batch_first=True)
        self.to_mean_sd = nn.Linear(hidden_state_to_posterior_size,2*self.latent_dims)
        self.slack = slack_term
        self.device = device
        
    def forward(self,X,taus):
        #X has shape [batch_size,timesteps,observed_dims]
        batch_size = X.shape[0]
        timesteps = X.shape[1]
        observed_dims = X.shape[2]
        X = X.float()
        taus_arr = torch.stack(batch_size*[torch.outer(torch.ones(timesteps),taus)])
        #R_diag_arr = torch.stack(batch_size*[torch.outer(torch.ones(timesteps),R_diag)])
        combined_arr = torch.concat([X,taus_arr],dim=2)
        #should have shape [hidden_dim_1]
        hidden_states,(_,_) = self.data_to_hidden_layer(combined_arr)
        last_hidden_states = hidden_states[:,-1,:]
        dynamics_initial_cond = self.hidden_layer_to_initial(last_hidden_states)
        dummy_inputs = torch.zeros((batch_size,timesteps,1)).float()
        dummy_initial_cxs =  torch.zeros(dynamics_initial_cond.shape).float()
        dynamics_hidden_states,_ = self.initial_to_posterior_states(dummy_inputs,(dynamics_initial_cond.reshape(1,batch_size,dynamics_initial_cond.shape[1]),dummy_initial_cxs.reshape(1,batch_size,dummy_initial_cxs.shape[1])))
        mean_sds = self.to_mean_sd(dynamics_hidden_states)
        means,sds = mean_sds[:,:,:self.latent_dims],mean_sds[:,:,self.latent_dims:]
        sds_tensor = torch.stack([torch.diag_embed(sds[batch_i,:,:].T**2 + self.slack) for batch_i in range(sds.shape[0])])
        corr_sds = sds_tensor.permute(0,2,3,1)
        return means,corr_sds


class FeedforwardVAEEncoder(nn.Module):
    #NOTE this is completely untested!
    def __init__(self,layers_list,latent_dims,observed_dims,timesteps,device="cpu"):
        super().__init__()
        self.slack = 0.0001
        self.timesteps = timesteps
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.flattened_input_dims = observed_dims*timesteps
        self.flattened_output_dims = latent_dims*timesteps*2
        self.layers = nn.ModuleList()
        self.norm_layer = nn.BatchNorm1d(observed_dims)
        input_size = self.flattened_input_dims
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.layers.append(nn.Linear(input_size,self.flattened_output_dims))
        self.to(device)

    def forward(self,X):
        batch_X = X.permute(0,2,1)
        normed_batch = self.norm_layer(batch_X)
        X = normed_batch.permute(0,2,1)
        batch_size = X.shape[0]
        flattened_X = torch.flatten(X,1)
        input_data = flattened_X
        for layer in self.layers:
            input_data = layer(input_data)
        output_matrix = input_data.reshape(2,batch_size,self.timesteps,self.latent_dims)
        means = output_matrix[0,:,:,:]
        sds = output_matrix[1,:,:,:]
        sds_tensor = torch.stack([torch.diag_embed(sds[batch_i,:,:].T**2 + self.slack) for batch_i in range(sds.shape[0])])
        corr_sds = sds_tensor.permute(0,2,3,1)
        return means,corr_sds