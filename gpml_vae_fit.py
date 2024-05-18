import torch
from tqdm import tqdm
from utils.gaussian_process import sde_kernel_matrices
from utils.dummy_optimizer import DummyOptimizer
from utils.elbo_loss import approx_elbo_loss
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import torch.nn as nn
from gpml_data_struct import GPMLDataStruct
def approx_elbo_loss(taus:torch.Tensor,batch_X:torch.Tensor,encoding_function:object,decoding_function:object,
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

class GPML_VAE(nn.Module,GPMLDataStruct):
    def __init__(self,device:str,latent_dims:int,observation_dims:int,times:torch.Tensor,
                 encoder_model:object,decoder_model:object,
                 signal_sds:list[float]=None, noise_sds:list[float]=None,
                 initial_taus:list[float]=None,initial_R_diag:list[float]=None,
                 train_taus = True, train_R_diag = True, train_decoder=True,train_encoder=True,
                 hetero_r = True)->object:
        '''
        Implements the GPML framework.
        device - torch device to use
        latent_dims - int specifying the number of latent dims (Z)
        observation_dims - int specifying the number of observation dims (X)
        times - tensor of times specifying the time of each observation
        signal_sds - list of positive floats (size latent_dims) specifying the signal sd of the GP covariance
                    None initializes all parameters to 1.0 - 0.001
        noise_sds - list of positive floats (size latent_dims) specifying the signal sd of the GP covariance
                    None initializes all parameters to 0.001
        initial_taus - list of positive floats (size latent_dims) specifying the time constant of the GP covariance.
                       None initializes the parameters randomly
        initial_R_diag - list of positive floats (size latent_dims) specifying the diagonal noise of the observations (X)
        encoder_model - object with methods: .parameters() which returns the an iterator on the model parameters
                                             .forward(X) -> mu, Sigma. 
                                                This function takes data and returns the mean and covariance of a mv normal distribution.
                                                X has shape [batch_size,timesteps,observation_dims]
                                                mu has shape [batch_size,timesteps,latent_dims]
                                                Sigma has shape [batch_size,timesteps,latent_dims,latent_dims]
        decoder_model - .parameters() which returns the an iterator on the model parameters
                        .forward(Z) -> manifold_mean
                            This function takes in samples from the latent space and returns the mean
                            of the random manifold that the data is generated from.
                            Z has shape [batch_size,samples,timesteps,observation_dims]
        '''
        self.latent_dims = latent_dims
        self.observation_dims = observation_dims
        #initializes tau
        #initialize base classes
        nn.Module.__init__(self)
        GPMLDataStruct.__init__(self,latent_dims,observation_dims,initial_taus,
                                initial_R_diag,signal_sds,noise_sds,decoder_model,
                                device)
        #builds all the passed internal variables
        self.encoder_model = encoder_model
        self.timesteps = times.shape[0]
        self.times = times
        #adds to the parameter set
        self.log_R_diag = nn.Parameter(self.log_R_diag)

        self.log_taus = self.log_taus
        if train_taus:
            self.log_taus.requires_grad = True
        else:
            self.log_taus.requires_grad = False

        if train_R_diag:
            self.log_R_diag.requires_grad = True
        else:
            self.log_R_diag.requires_grad = False

        if train_decoder:
            for p in decoder_model.parameters():
                p.requires_grad = True
        else:
            for p in decoder_model.parameters():
                p.requires_grad = False
        if train_encoder:
            for p in encoder_model.parameters():
                p.requires_grad = True
        else:
            for p in encoder_model.parameters():
                p.requires_grad = False
    
    def forward(self,X,approx_elbo_samples=1,loss_hyper=1.0,no_tau=False):
        Ks = self.sde_kernel_matrices(self.times,no_tau = no_tau)
        encoding_mus,encoding_Sigmas = self.encoder_model.forward(X)
        #computes the ll expectation term
        encoding_dist_samples = sample_encoding_dist(encoding_mus,encoding_Sigmas,approx_elbo_samples)
        decoding_manifold_means = self.forward_model.forward(encoding_dist_samples)

        log_likelihood_sample_losses = ll_loss(X,decoding_manifold_means,torch.diag(self.R_diag()))
        approx_individual_log_likelihood = torch.mean(log_likelihood_sample_losses,dim=1)
        #computes the kl divergence term
        kl_divergence_term = kl_divergence(encoding_mus,encoding_Sigmas,Ks)
        #combines them together to get the approximate elbo per item in the batch
        #print(torch.sum(kl_divergence_term))
        individual_elbos = -1*approx_individual_log_likelihood + loss_hyper*kl_divergence_term
        return individual_elbos, kl_divergence_term, -1*approx_individual_log_likelihood 
    
    def fit(self,X_train:torch.Tensor,X_validation:torch.Tensor,lr=0.001,epochs=100,batch_size=1,approx_elbo_loss_samples=1,
            log_save_name="latest",hyper_scheduler=None,save_cycle=100,grad_clip=0.01,no_tau_epochs=0,mm=0.99,
            tau_lr=0.001,plotting_true_taus=None,plotting_true_R_diag=None):
        '''
        X_train- tensor of shape (iid_samples,time_steps,observation_dims): Gives the training data consisting of 
                     data assumed to be generated by independent gaussian process latents with an sde kernel and then 
                     tranformed onto a nonlinear statistical manifold and then sampled from with a normal distribution.
        encoder_optimizer - optimizer object that implements the method .step() which is called after every batch: 
                            optimizer for the encoder model parameters.
        decoder_optimizer - optimizer object that implements the method .step() which is called after every batch:
                            optimizer for the encoder model parameters.
        optimize_taus - bool: indicates whether or not to run the optimizer on taus
        optimize_R - bool: indicates whether or not to run the optimizer on R
        batch_size - int: indicates the batch size to split the data into for each gradient step
        '''
        #constructs an optimizer for the parameters tau and R, makes a dummy optimizer if none is passed
        self.taus_trajectory = []
        self.R_diag_trajectory = []
        self.validation_loss = []
        self.training_loss = []
        self.neg_ll_train_loss = []
        self.kl_loss = []
        if tau_lr is None:
            optimizer = torch.optim.SGD([{"params":self.log_taus},{"params":self.parameters()}],lr=lr,momentum=mm)
        else:
            optimizer = torch.optim.SGD([{"params":self.log_taus,"lr":tau_lr},{"params":self.parameters()}],lr=lr,momentum=mm)
        #optimizer = torch.optim.SGD(self.parameters(),lr=lr,momentum=mm)
        #constructs an optimizer for the parameters tau and R, makes a dummy optimizer if none is passed
        #begins the main training loop
        batched_X_train = DataLoader(TensorDataset(X_train),batch_size=batch_size,shuffle=True)
        for epoch in tqdm(range(epochs),desc=f"VAE Fit",colour="cyan"):
            if not hyper_scheduler is None:
                hyper_param = hyper_scheduler(epoch)
            else:
                hyper_param = 1.0
            if epoch < no_tau_epochs:
                no_tau = True
            else:
                no_tau = False
            epoch_total_losses = []
            epoch_kl_losses = []
            epoch_neg_ll_losses = []
            for (batch_X,) in batched_X_train:
                optimizer.zero_grad()
                batch_neg_elbos,batch_kls,batch_neg_lls = self.forward(batch_X,approx_elbo_loss_samples,loss_hyper=hyper_param,no_tau=no_tau)    
                loss = torch.mean(batch_neg_elbos) #+ r_hyper*torch.linalg.norm(self.R_diag())**2
                loss.backward()
                #clips the grad
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                #steps the grad
                optimizer.step()
                #training loss
                epoch_total_losses.append(loss.clone().detach().numpy())
                epoch_neg_ll_losses.append(torch.mean(batch_neg_lls).clone().detach().numpy())
                epoch_kl_losses.append(torch.mean(batch_kls).clone().detach().numpy())
            self.training_loss.append(np.mean(np.array(epoch_total_losses)))
            self.neg_ll_train_loss.append(np.mean(np.array(epoch_neg_ll_losses)))
            self.kl_loss.append(np.mean(np.array(epoch_kl_losses)))
            self.taus_trajectory.append(self.taus().clone().detach().numpy())
            self.R_diag_trajectory.append(self.R_diag().clone().detach().numpy())
            if not X_validation is None:
                with torch.no_grad():
                    validation_neg_lls = self.forward(X_validation,approx_elbo_loss_samples)[0]
                    self.validation_loss.append(torch.mean(validation_neg_lls).clone().detach().numpy())
            if epoch%save_cycle == 0:
                #print([x for x in self.encoder_model.parameters()][1])
                plot_loss(self.training_loss,self.neg_ll_train_loss,self.kl_loss,self.validation_loss,log_save_name)
                plot_taus(self.taus_trajectory,log_save_name,plotting_true_taus)
                plot_R_diag(self.R_diag_trajectory,log_save_name,plotting_true_R_diag)

def save_log(data,filename):
    with open(f"./logs/{filename}.pkl","wb") as f:
        pkl.dump(data,f)

def plot_loss(epochs_total_loss,epochs_ll_loss,epochs_kl_loss,epochs_validation_loss,filename):
    epochs = range(len(epochs_total_loss))
    plt.title(f"GPML VAE Estimator: {filename} Loss Over Epochs")
    last_total_loss = '%.4f'%(epochs_total_loss[-1])
    last_kl_loss = '%.4f'%(epochs_ll_loss[-1])
    last_ll_loss =  '%.4f'%(epochs_kl_loss[-1])
    plt.plot(epochs,epochs_total_loss,color="black",label=f"Total Loss: {last_total_loss}")
    plt.plot(epochs,epochs_ll_loss,color = "red",label=f"LL Loss: {last_kl_loss}")
    plt.plot(epochs,epochs_kl_loss,color="blue",label=f" KL Loss: {last_ll_loss}")
    if len(epochs_validation_loss) > 0:
        plt.plot(epochs,epochs_validation_loss,label="Test Loss",color="purple")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.yscale("symlog")
    plt.legend()
    plt.savefig(f"./logs/{filename}_Loss.png")
    plt.cla()

def plot_taus(epochs_taus,filename,true_taus=None):
    if not true_taus is None:
      for tau in true_taus:
          plt.axhline(tau,color="red",linestyle="--")
    epochs = range(len(epochs_taus))
    plt.title(f"GPML VAE Estimator: {filename} Taus Over Epochs")
    epochs_taus = np.array(epochs_taus)
    for tau_i in range(epochs_taus.shape[1]):
      decimals = '%.4f'%(epochs_taus[-1,tau_i])
      plt.plot(epochs,epochs_taus[:,tau_i],label=f"{decimals}")
    plt.ylabel("Tau")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f"./logs/{filename}_Taus.png")
    plt.cla()

def plot_R_diag(epochs_R_diag,filename,true_R_diag=None):
    if not true_R_diag is None:
      for r in true_R_diag:
          plt.axhline(r,color="red", linestyle="--")
    epochs = range(len(epochs_R_diag))
    plt.title(f"GPML VAE Estimator: {filename} R Over Epochs")
    epochs_R_diag = np.array(epochs_R_diag)
    for r_i in range(epochs_R_diag.shape[1]):
      decimals = '%.4f'%(epochs_R_diag[-1,r_i])
      plt.plot(epochs,epochs_R_diag[:,r_i],label=f"{decimals}")
    plt.ylabel("R")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f"./logs/{filename}_R_diag.png")
    plt.cla()