import torch
def sde_kernel_matrices(times:torch.Tensor,taus:torch.Tensor,
                        signal_sds:torch.Tensor,noise_sds:torch.Tensor)->torch.Tensor:
   '''implements simultaneous construction of the sde kernal
      for multiple taus, signal_sds, noise_sds

      timesteps - tensor of floats with shape (timesteps)
      taus - tensor of shape (latent_dims): time consts of each GP
      signal_sds - tensor of shape (latent_dims): signal_sds of the GP
      noise_sds - tensor of shape (latent_dims): noise_sds of the GP

      returns kernels - tensor of shape [latent_dims,timesteps,timesteps]:
                        which gives the kernal matrices for each latent_dim
      
   '''
   #NOTE: this function passes its test cases
   assert taus.shape == signal_sds.shape == noise_sds.shape
   timesteps = times.shape[0]
   data_times_square = torch.outer(torch.square(times),torch.ones(timesteps))
   exp_shared_term = -1*(data_times_square + data_times_square.T - 2 * torch.outer(times,times))
   exp_taus_term = 2*torch.square(taus)
   #has shape [timesteps,timesteps,latent_dims]
   signal_term = (signal_sds**2)*torch.exp(exp_shared_term.unsqueeze(2)/exp_taus_term)
   noise_term = (noise_sds**2)*(torch.eye(timesteps).unsqueeze(2))
   output_matrix =  signal_term + noise_term
   correct_shape_output = output_matrix.movedim(2,0)
   return correct_shape_output
