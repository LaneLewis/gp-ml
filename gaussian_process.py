import torch
def sde_kernel_matrices(timesteps:int,taus:torch.Tensor,
                        signal_sds:torch.Tensor,noise_sds:torch.Tensor)->torch.Tensor:
    '''implements simultaneous construction of the sde kernal
       for multiple taus, signal_sds, noise_sds

       timesteps - int: number of timesteps of the data (X)
       taus - tensor of shape (latent_dims): time consts of each GP
       signal_sds - tensor of shape (latent_dims): signal_sds of the GP
       noise_sds - tensor of shape (latent_dims): noise_sds of the GP

       returns kernels - tensor of shape [latent_dims,timesteps,timesteps]:
                         which gives the kernal matrices for each latent_dim
    '''
    assert taus.shape == signal_sds.shape == noise_sds.shape
    raise NotImplementedError
    return 