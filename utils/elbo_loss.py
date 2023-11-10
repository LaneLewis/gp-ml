import torch

def approx_elbo_loss(batch_X:torch.Tensor,encoding_function:function,decoding_function:function,samples:int=100)->torch.Tensor:
    '''
    computes the elbo loss for a batched dataset
    batch_X - tensor of shape [batch_size, timesteps, observation_dims] : The batched dataset to compute the loss over
    encoder_function - function:(X) -> mu, Sigma. 
                    This function takes data and returns the mean and covariance of a mv normal distribution.
                    X has shape [batch_size,timesteps,observation_dims]
                    mu has shape [batch_size,timesteps,latent_dims]
                    Sigma has shape [batch_size,timesteps,latent_dims,latent_dims]
    decoder_function - function:(Z) -> manifold_mean
                    This function takes in samples from the latent space and returns the mean
                    of the random manifold that the data is generated from.
                    Z has shape [batch_size,samples,timesteps,observation_dims]
    samples - int: number of samples to sample from the encoder before passing
    '''