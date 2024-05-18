import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset,DataLoader
import gc
import matplotlib.pyplot as plt
from torch import nn
from gpml_data_struct import GPMLDataStruct
import numpy as np

def sample_prior(timesteps,Ks,samples,batch_size):
    '''
    returns prior_samples - shape[batch_size,samples,timesteps,latent_dims]:
                                samples from the prior
    '''
    prior_samples = torch.distributions.MultivariateNormal(torch.zeros(timesteps),Ks.permute(2,0,1)).rsample((batch_size,samples))
    return prior_samples.permute(0,1,3,2)

def ll_loss(batch_X,decoding_manifold_means,R):
    #decoding_manifold_means shape [batchsize, samples, timesteps, observed_dims]
    #print(decoding_manifold_means[0,:,:,:].mean(dim=0))
    data_dist = torch.distributions.MultivariateNormal(decoding_manifold_means.permute(1,0,2,3),R)
    #testing
    data_prob = data_dist.log_prob(batch_X)
    summed_over_time = torch.sum(data_prob,dim=2)
    return summed_over_time.permute(1,0)

class GPML_LLF(nn.Module,GPMLDataStruct):
    def __init__(self,device:str,latent_dims:int,observation_dims:int,times:torch.Tensor,decoder_model:object,
                 signal_sds:list[float]=None, noise_sds:list[float]=None,
                 initial_taus:list[float]=None,initial_R_diag:list[float]=None,
                 train_taus = True, train_R_diag = True, train_decoder=True)->object:
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
                                             .forward(X) -> mu
                                                This function takes data and returns the mean of the posterior latents.
                                                X has shape [batch_size,timesteps,observation_dims]
                                                mu has shape [batch_size,timesteps,latent_dims]
        decoder_model - .parameters() which returns the an iterator on the model parameters
                        .forward(Z) -> manifold_mean
                            This function takes in samples from the latent space and returns the mean
                            of the random manifold that the data is generated from.
                            Z has shape [batch_size,samples,timesteps,observation_dims]
        '''
        #initialize base classes
        nn.Module.__init__(self)
        GPMLDataStruct.__init__(self,latent_dims,observation_dims,initial_taus,
                                initial_R_diag,signal_sds,noise_sds,decoder_model,device)
        #builds all the passed internal variables
        self.log_R_diag = nn.Parameter(self.log_R_diag)
        self.log_taus = nn.Parameter(self.log_taus)
        #registers them to the nn
        self.decoder_model = decoder_model
        self.times = times
        self.timesteps = times.shape[0]

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


    def forward(self,X,samples):
        batch_size = X.shape[0]
        R = torch.diag(self.R_diag())
        Ks = self.sde_kernel_matrices(self.times)
        prior_samples = sample_prior(self.timesteps,Ks,samples,batch_size)
        decoding_manifold_means = self.decoder_model(prior_samples)
        log_likelihood_sample_losses = ll_loss(X,decoding_manifold_means,R)
        log_inv_samples = torch.log(torch.tensor(1/samples,requires_grad=False,device=self.device))
        approx_neg_log_likelihood = -1*(torch.logsumexp(log_likelihood_sample_losses,dim=1) + log_inv_samples)
        return approx_neg_log_likelihood

    def fit(self,X_train:torch.Tensor,X_validation=None,epochs=100,batch_size=1,lr=0.001,
                             approx_log_likelihood_loss_samples=100,save_cycle=100,log_save_name="MC_Latest"):
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
        self.taus_trajectory = []
        self.R_diag_trajectory = []
        self.validation_loss = []
        self.training_loss = []
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        #constructs an optimizer for the parameters tau and R, makes a dummy optimizer if none is passed
        #begins the main training loop
        batched_X_train = DataLoader(TensorDataset(X_train),batch_size=batch_size)
        for epoch in tqdm(range(epochs),desc=f"MC Fit",colour="cyan"):
            batch_total_losses = []
            for (batch_X,) in batched_X_train:
                optimizer.zero_grad()
                batch_neg_lls = self.forward(batch_X,approx_log_likelihood_loss_samples)
                loss = torch.mean(batch_neg_lls)
                loss.backward()
                #steps the grad
                optimizer.step()
                #training loss
                batch_total_losses.append(loss.clone().detach().numpy())
            self.training_loss.append(np.mean(np.array(batch_total_losses)))
            self.taus_trajectory.append(self.taus().clone().detach().numpy())
            self.R_diag_trajectory.append(self.R_diag().clone().detach().numpy())
            gc.collect()
            if epoch%save_cycle == 0 and epoch > 0:
                plot_taus(self.taus_trajectory,log_save_name)
            if not X_validation is None:
                with torch.no_grad():
                    validation_neg_lls = self.forward(X_validation,approx_log_likelihood_loss_samples)
                    self.validation_loss.append(torch.mean(validation_neg_lls).clone().detach().numpy())

def plot_taus(epochs_taus,filename,average=99):
    epochs = range(len(epochs_taus))
    plt.title("GPML MC: Loss Over Epochs")
    plt.plot(epochs,epochs_taus,label=f"{epochs_taus[-1]}")
    if len(epochs_taus) > average:
        avgs = np.mean(np.array(epochs_taus)[-average:,:],axis=0)
    else:
        avgs = np.mean(np.array(epochs_taus),axis=0)
    for v in avgs:
        plt.axhline(v,label=f"mean taus: {v}",color="red")
    plt.ylabel("Tau")
    plt.legend()
    plt.savefig(f"./logs/{filename}_Taus.png")
    plt.cla()