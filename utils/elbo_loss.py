import torch
from typing import Callable

def approx_elbo_loss(taus:torch.Tensor,batch_X:torch.Tensor,encoding_function:Callable,decoding_function:Callable,
                     R:torch.Tensor,Ks:torch.tensor,samples:int=10,loss_hyper=10.0,taus_to_encoder=False,taus_to_decoder=False)->torch.Tensor:
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
    
    if taus_to_encoder:
        encoding_mus,encoding_Sigmas = encoding_function(batch_X,taus)
    else:
        encoding_mus,encoding_Sigmas = encoding_function(batch_X)
    #computes the ll expectation term
    encoding_dist_samples = sample_encoding_dist(encoding_mus,encoding_Sigmas,samples)

    if taus_to_decoder:
        decoding_manifold_means = decoding_function(encoding_dist_samples,taus)
    else:
        decoding_manifold_means = decoding_function(encoding_dist_samples)

    log_likelihood_sample_losses = ll_loss(batch_X,decoding_manifold_means,R)
    approx_individual_log_likelihood= torch.mean(log_likelihood_sample_losses,dim=1)
    #computes the kl divergence term
    kl_divergence_term = kl_divergence(encoding_mus,encoding_Sigmas,Ks)
    #combines them together to get the approximate elbo per item in the batch
    #print(torch.sum(kl_divergence_term))
    individual_elbos = -1*approx_individual_log_likelihood + loss_hyper*kl_divergence_term
    return individual_elbos, kl_divergence_term, -1*approx_individual_log_likelihood 

def sample_encoding_dist(encoding_mus:torch.Tensor,encoding_Sigmas:torch.Tensor,samples:int)->torch.Tensor:
    '''
    samples from the encoding distribution a number of times equal to samples

    encoding_mus - tensor of shape [batch_size,timesteps,latent_dims]
    encoding_Sigmas - tensor of shape [batch_size,time_steps,timesteps,latent_dims]

    returns encoding_samples - shape[batch_size,samples,timesteps,latent_dims]:
                                samples from the encoding distribution
    '''
    #consistent batch size
    assert encoding_mus.shape[0] == encoding_Sigmas.shape[0] == encoding_mus.shape[0]
    #consistent timestep size
    assert encoding_mus.shape[1] == encoding_Sigmas.shape[1] == encoding_Sigmas.shape[2]
    #consistent latent_dims
    assert encoding_mus.shape[2] == encoding_Sigmas.shape[3]

    batch_size = encoding_mus.shape[0]
    timesteps = encoding_mus.shape[1]
    latent_dims = encoding_mus.shape[2]
    #permutes to allow broadcasting
    batch_flat_mus = torch.permute(encoding_mus,(0,2,1)).reshape(batch_size,timesteps*latent_dims)
    batch_flat_covs = torch.vmap(convert_to_flat_cov)(encoding_Sigmas)
    batch_dist = torch.distributions.MultivariateNormal(batch_flat_mus,batch_flat_covs)
    flat_samples = batch_dist.rsample((samples,))
    #samples are currently flat, they need to be unflattened
    reshaped_samples = flat_samples.reshape(samples,batch_size,latent_dims,timesteps).permute((1,0,3,2))
    return reshaped_samples

def log_likelihood_loss(batch_X,decoding_manifold_means,R)->torch.tensor:
    '''
    batch_X - tensor of shape [batch_size, timesteps, observation_dims]: the batched data
    decoding_manifold_means - tensor of shape [batch_size, samples, timesteps, observation_dims]: 
                              manifold mean embedding by passing the random latent samples through the decoder
    R - tensor of shape [observation_dims,observation_dims]: the covariance matrix of the observations
    returns log_likelihood_losses - tensor of shape [batch_size, samples]
    '''
    #makes sure requirements are satisfied
    assert batch_X.shape[0] == decoding_manifold_means.shape[0]
    assert batch_X.shape[1] == decoding_manifold_means.shape[2]
    assert R.shape[0] == R.shape[1] == batch_X.shape[2] == decoding_manifold_means.shape[3]
    timesteps = batch_X.shape[1]

    time_flat_X = torch.flatten(batch_X,start_dim=1)
    time_flat_manifold_means = torch.flatten(decoding_manifold_means,start_dim=2)
    time_flat_block_R = torch.block_diag(*timesteps*[R])
    #permutes means to allow for broadcasting
    permuted_time_flat_manifold_means = time_flat_manifold_means.permute((1,0,2))
    #broacasting covariance across samples and batch size
    time_flat_batch_normal = torch.distributions.MultivariateNormal(permuted_time_flat_manifold_means,time_flat_block_R)
    broadcast_log_prob = time_flat_batch_normal.log_prob(time_flat_X)
    #undo the permute carried out for broadcasting
    return broadcast_log_prob.T

def kl_divergence(encoding_mus:torch.Tensor,encoding_Sigmas:torch.Tensor,K:torch.Tensor)->torch.Tensor:
    '''
    computes the kl divergence between the encoding distribution and the prior distribution across batches
    encoding_mus - tensor of shape [batch_size, timesteps, latent_dims]
    encoding_Sigmas - tensor of shape [batch_size, timesteps, timesteps, latent_dims]
    K - tensor of shape [timesteps, timesteps, latent_dims]

    returns kl_divergences - tensor of shape [batch_size]
    '''
    #NOTE passes its unit test, should have more since there is some method overlap between test and this
    assert encoding_mus.shape[0] == encoding_Sigmas.shape[0]
    assert encoding_mus.shape[1] == encoding_Sigmas.shape[1] == K.shape[0] == K.shape[1]
    assert encoding_mus.shape[2] == encoding_Sigmas.shape[3] == K.shape[2]
    batch_size = encoding_mus.shape[0]
    timesteps = encoding_mus.shape[1]
    latent_dims = encoding_mus.shape[2]
    block_diag_K = torch.block_diag(*[K[:,:,i] for i in range(K.shape[2])])

    #permutes in order to reshape correctly into flattened
    batch_flat_mus = torch.permute(encoding_mus,(0,2,1)).reshape(batch_size,timesteps*latent_dims)
    batch_flat_covs = torch.vmap(convert_to_flat_cov)(encoding_Sigmas)
    batched_normal_dist = torch.distributions.MultivariateNormal(batch_flat_mus,batch_flat_covs)
    target_dist = torch.distributions.MultivariateNormal(torch.zeros(timesteps*latent_dims,requires_grad=False),block_diag_K)
    return torch.distributions.kl.kl_divergence(batched_normal_dist,target_dist)

def convert_to_flat_cov(single_encoding_Sigmas):
    '''
    single_encoding_Sigmas - shape [timesteps,timesteps,latent_dims] : covariance matrices across latents
    return flat_block_cov - shape [timesteps*latent_dims,timesteps*latent_dims] : 
                            block diagonal cov matrix for covariance of flattened distribution
    '''
    return torch.block_diag(*map(torch.squeeze, single_encoding_Sigmas.split(1,dim=2)))

def ll_loss(batch_X,decoding_manifold_means,R):
    
    #decoding_manifold_means shape [batchsize, samples, timesteps, observed_dims]
    data_dist = torch.distributions.MultivariateNormal(decoding_manifold_means.permute(1,0,2,3),R)
    #testing
    data_prob = data_dist.log_prob(batch_X)
    summed_over_time = torch.sum(data_prob,dim=2)
    return summed_over_time.permute(1,0)