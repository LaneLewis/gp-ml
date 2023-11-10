import torch

def approx_elbo_loss(batch_X:torch.Tensor,encoding_function:function,decoding_function:function,
                     R:torch.Tensor,K:torch.tensor,samples:int=100)->torch.Tensor:
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
    #computes the ll expectation term
    encoding_dist_samples = sample_encoding_dist(encoding_mus,encoding_Sigmas,samples)
    decoding_manifold_means = decoding_function(encoding_dist_samples)
    log_likelihood_sample_losses = log_likelihood_loss(batch_X,decoding_manifold_means,R)
    approx_individual_log_likelihood= torch.mean(log_likelihood_sample_losses,dim=1)
    #computes the kl divergence term
    kl_divergence_term = kl_divergence(encoding_mus,encoding_Sigmas,K)
    #combines them together to get the approximate elbo per item in the batch
    individual_elbos = -1*approx_individual_log_likelihood + kl_divergence_term
    return individual_elbos

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
    R - tensor of shape [observation_dims,observation_dims]: the 
    returns log_likelihood_losses - tensor of shape [batch_size, samples]
    '''
    raise NotImplementedError

    return 

def kl_divergence(encoding_mus,encoding_Sigmas,K)->torch.Tensor:
    '''
    computes the kl divergence between the encoding distribution and the prior distribution across batches
    encoding_mus - tensor of shape [batch_size, timesteps, observation_dims]
    encoding_Sigmas - tensor of shape [batch_size, timesteps, timesteps, observation_dims]
    K - tensor of shape [timesteps, timesteps]

    returns kl_divergences - tensor of shape [batch_size]
    '''
    raise NotImplementedError
    return 