import torch
import math
import torch.nn as nn
class GPMLDataStruct():
    def __init__(self,latent_dims,observation_dims,taus=None,R_diag=None,
                 signal_sds=None,noise_sds=None,forward_model=None,device="cpu",hetero_r=True):
        self.latent_dims = latent_dims
        self.observation_dims = observation_dims
        if taus == None:
            self.log_taus = torch.rand((latent_dims),requires_grad=False,device=device)
        else:
            assert len(taus) == latent_dims
            log_taus = [math.log(tau) for tau in taus]
            self.log_taus = torch.tensor(log_taus,requires_grad=False,device=device)
        #initializes R
        if R_diag == None:
            if hetero_r:
                self.log_R_diag = torch.rand((self.observation_dims),requires_grad=False,device=device)
            else:
                self.log_R_diag = torch.rand((1),requires_grad=False,device=device)
        else:
            if hetero_r:
                assert len(R_diag) == observation_dims
                log_R_diag = [math.log(r) for r in R_diag]
                self.log_R_diag = torch.tensor(log_R_diag,requires_grad=False,device=device)
            else:
                self.log_R_diag = torch.tensor((math.log(R_diag)),requires_grad=False,device=device)
        #initializes signal sds
        if signal_sds == None:
            self.signal_sds = (1.0 - 0.01)*torch.ones((self.latent_dims),requires_grad=False,device=device)
        else:
            assert len(signal_sds) == latent_dims
            self.signal_sds = torch.tensor(signal_sds,requires_grad=False,device=device)
        #initializes noise_sds
        if noise_sds == None:
            self.noise_sds = 0.01*torch.ones((self.latent_dims),requires_grad=False,device=device)
        else:
            assert len(noise_sds) == latent_dims
            self.noise_sds = torch.tensor(noise_sds,requires_grad=False,device=device)
        if forward_model == None:
            self.forward_model = nn.Linear(latent_dims,observation_dims)
        else:
            self.forward_model = forward_model
        self.device=device
        self.hetero_r = hetero_r
        assert self.log_taus.shape == self.signal_sds.shape == self.noise_sds.shape

    def sde_kernel_matrices(self,times:torch.Tensor,no_tau=False)->torch.Tensor:
        '''implements simultaneous construction of the sde kernal
            for multiple taus, signal_sds, noise_sds

            timesteps - tensor of floats with shape (timesteps)
            taus - tensor of shape (latent_dims): time consts of each GP
            signal_sds - tensor of shape (latent_dims): signal_sds of the GP
            noise_sds - tensor of shape (latent_dims): noise_sds of the GP

            returns kernels - tensor of shape [timesteps,timesteps,latent_dims]:
                                which gives the kernal matrices for each latent_dim
        '''
        if no_tau:
            timesteps = len(times)
            batched_identity = torch.eye(timesteps).reshape(1,timesteps,timesteps).repeat(self.latent_dims,1,1)
            return batched_identity.permute(1,2,0)
        #NOTE: this function passes its test cases
        taus = torch.exp(self.log_taus)
        timesteps = times.shape[0]
        data_times_square = torch.outer(torch.square(times),torch.ones(timesteps,device=self.device,requires_grad=False))
        exp_shared_term = -1*(data_times_square + data_times_square.T - 2 * torch.outer(times,times))
        exp_taus_term = 2*torch.square(taus)
        #has shape [timesteps,timesteps,latent_dims]
        signal_term = (self.signal_sds**2)*torch.exp(exp_shared_term.unsqueeze(2)/exp_taus_term)
        noise_term = (self.noise_sds**2)*(torch.eye(timesteps).unsqueeze(2))
        output_matrix =  signal_term + noise_term
        return output_matrix
    
    def taus(self):
        return torch.exp(self.log_taus)
    
    def R_diag(self):
        if self.hetero_r:
            return torch.exp(self.log_R_diag)
        else:
            return torch.ones(self.observation_dims)*torch.exp(self.log_R_diag)
    
    def sample_model(self,samples,times):
        kernel_matrices = self.sde_kernel_matrices(times)
        R = torch.diag(self.R_diag())
        time_steps = torch.zeros(kernel_matrices.shape[0],device=self.device)
        prior_samples = torch.distributions.MultivariateNormal(time_steps,kernel_matrices.permute(2,0,1)).sample((samples,)).permute(0,2,1)
        decoding_manifold_means = self.forward_model.forward(prior_samples)
        data_samples = torch.distributions.MultivariateNormal(decoding_manifold_means,R).sample((1,)).squeeze()
        return prior_samples, data_samples
