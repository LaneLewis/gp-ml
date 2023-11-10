import torch

def approx_elbo_loss(batch_X:torch.Tensor,encoding_function:function,decoding_function:function,R:torch.Tensor,samples:int=100)->torch.Tensor:
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

    returns individual_elbos - shape [batch_size]: loss for each sample of X in the batch
    '''
    encoding_mus,encoding_Sigmas = encoding_function(batch_X)
    encoding_dist_samples = sample_encoding_dist(encoding_mus,encoding_Sigmas,samples)
    decoding_manifold_means = decoding_function(encoding_dist_samples)

    raise NotImplementedError

    return 

def sample_encoding_dist(encoding_mus:torch.Tensor,encoding_Sigmas:torch.Tensor,samples:int)->torch.Tensor:
    '''
    samples from the encoding distribution a number of times equal to samples

    encoding_mus - tensor of shape [batch_size,timesteps,latent_dims]
    encoding_Sigmas - tensor of shape [batch_size,time_steps,timesteps,latent_dims]

    returns encoding_samples - shape[batch_size,samples,timesteps,latent_dims]:
                                samples from the encoding distribution
    '''
    raise NotImplementedError

    return 

def log_likelihood_loss(batch_X,decoding_manifold_means,R)->torch.tensor:
    '''
    batch_X - tensor of shape [batch_size, timesteps, observation_dims]: the batched data
    decoding_manifold_means - tensor of shape [batch_size, samples timesteps, observation_dims]: 
                              manifold mean embedding by passing the random latent samples through the decoder
    R - tensor of shape 
    returns log_likelihood_losses - tensor of shape [batch_size, samples]
    '''