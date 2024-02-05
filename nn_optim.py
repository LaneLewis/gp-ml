import torch
from tqdm import tqdm
from utils.gaussian_process import sde_kernel_matrices
from utils.dummy_optimizer import DummyOptimizer
from utils.data_ll_loss import approx_batch_log_likelihood_loss
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
from utils.latent_prediction import LSTM_Encoder,FeedforwardPrediction
from utils.nn_optimzation import approx_data_log_likelihood,NNBBOptimizer,activate_model,deactivate_model
import gc
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import math

class GPML_NN_LLF():
    def __init__(self,device:str,latent_dims:int,observation_dims:int,times:torch.Tensor,decoder_model:object,
                 signal_sds:list[float]=None, noise_sds:list[float]=None,
                 initial_taus:list[float]=None,initial_R_diag:list[float]=None,NN_layers=None,nn_adam_lr=0.005)->object:
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
        self.latent_dims = latent_dims
        self.observation_dims = observation_dims
        self.timesteps = times.shape[0]
        self.device = device
        #initializes tau
        if initial_taus == None:
            self.log_taus = torch.rand((latent_dims),requires_grad=False,device=device)
        else:
            assert len(initial_taus) == latent_dims
            initial_log_taus = [math.log(tau) for tau in initial_taus]
            self.log_taus = torch.tensor(initial_log_taus,requires_grad=False,device=device)
        #initializes R
        if initial_R_diag== None:
            self.log_R_diag = torch.rand((self.observation_dims),requires_grad=False,device=device)
        else:
            assert len(initial_R_diag) == observation_dims
            initial_log_R_diag = [math.log(R_d) for R_d in initial_R_diag]
            self.log_R_diag = torch.tensor(initial_log_R_diag,requires_grad=False,device=device)
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
        #builds all the passed internal variables
        self.decoder_model = decoder_model
        self.times = times
        self.timesteps = times.shape[0]
        #constructs the kernel matricies
        self.training_loss = []
        #constructs initial variables
        self.taus = torch.exp(self.log_taus)
        self.R_diag = torch.exp(self.log_R_diag)
        self.R = torch.diag(self.R_diag)

        self.kernel_matrices = sde_kernel_matrices(self.times,self.taus,self.signal_sds,self.noise_sds)

        self.nn_optimizer = NNBBOptimizer(decoder_model,self.observation_dims,self.timesteps,self.taus,self.R_diag,NN_layers)
        self.nn_adam = torch.optim.Adam(self.nn_optimizer.parameters(),lr=nn_adam_lr)

    def fit_generative_model(self,X_train:torch.Tensor,decoding_optimizer, epochs=100,
            tau_lr=0.01,R_diag_lr =0.001, optimize_taus=True,optimize_R=True,batch_size=1,approx_log_likelihood_loss_samples=100,print_batch_values=True):
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
        #constructs an optimizer for the parameters tau and R, makes a dummy optimizer if none is passed
        if optimize_taus:
            self.log_taus.requires_grad = True
            tau_optimizer = optim.Adam([self.log_taus],lr=tau_lr)
        else:
            tau_optimizer = DummyOptimizer()

        if optimize_R:
            self.log_R_diag.requires_grad = True
            R_optimizer = optim.Adam([self.log_R_diag],lr=R_diag_lr)
        else:
            R_optimizer = DummyOptimizer()
        #constructs an optimizer for the decoder parameters
        #begins the main training loop
        batched_X_train = DataLoader(TensorDataset(X_train),batch_size=batch_size)
        for epoch in range(epochs):
            batch_size = 0
            batch_total_losses = []
            for (batch_X,) in tqdm(batched_X_train,desc=f"epoch: {epoch}",colour="cyan"):
                tau_optimizer.zero_grad()
                R_optimizer.zero_grad()
                self.nn_adam.zero_grad()

                self.R_diag = torch.exp(self.log_R_diag)
                self.R = torch.diag(self.R_diag)
                self.taus = torch.exp(self.log_taus)

                decoding_optimizer.zero_grad()
                #computes the loss
                with torch.no_grad():
                    self.kernel_matrices = sde_kernel_matrices(self.times,self.taus,self.signal_sds,self.noise_sds)
                    batch_approx_samples = approx_data_log_likelihood(self.timesteps,batch_X,self.decoder_model.forward,self.R,self.kernel_matrices,approx_log_likelihood_loss_samples)
                #should freeze decoder_model parameters
                deactivate_model(self.decoder_model)
                self.log_taus.requires_grad = False
                self.log_R_diag.requires_grad = False

                nn_ll_predictions = self.nn_optimizer.forward(batch_X,self.decoder_model.parameters(),self.taus,self.R_diag)
                prediction_loss = torch.sum(torch.square(batch_approx_samples - nn_ll_predictions))
                prediction_loss.backward()
                activate_model(self.decoder_model)
                self.log_taus.requires_grad = True
                self.log_R_diag.requires_grad = True
                #reset all the R_diags
                self.R_diag = torch.exp(self.log_R_diag)
                self.R = torch.diag(self.R_diag)
                self.taus = torch.exp(self.log_taus)
                #should freeze nn_optimizer
                deactivate_model(self.nn_optimizer)
                nn_ll = sum(-1*self.nn_optimizer.forward(batch_X,self.decoder_model.parameters(),self.taus,self.R_diag))
                nn_ll.backward()
                activate_model(self.nn_optimizer)

                #steps the grad
                decoding_optimizer.step()
                tau_optimizer.step()
                R_optimizer.step()
                self.nn_adam.step()
                #training loss
                batch_total_losses.append(prediction_loss.detach().numpy())
                batch_size +=1
            print(self.decoder_model.parameters())
            self.training_loss.append(prediction_loss.detach().numpy())
            self.taus_trajectory.append(torch.exp(self.log_taus).clone().detach().numpy())
            self.R_diag_trajectory.append(torch.exp(self.log_R_diag).clone().detach().numpy())
            if print_batch_values:
                print(f"R_diag: {torch.exp(self.log_R_diag).detach().numpy()}")
                print(f"Tau: {torch.exp(self.log_taus).detach().numpy()}")
                print(f"total loss:{sum(batch_total_losses)/batch_size}")
            gc.collect()
        #saves the final kernel and R matrices for use later
        self.kernel_matrices = sde_kernel_matrices(self.times,self.taus,self.signal_sds,self.noise_sds)
        self.R = torch.diag(self.R_diag)

    def sample_model(self,samples):
        with torch.no_grad():
            kernel_matrices = self.kernel_matrices
            time_steps = torch.zeros(kernel_matrices.shape[0],device=self.device)
            prior_samples = torch.distributions.MultivariateNormal(time_steps,kernel_matrices.permute(2,0,1)).sample((samples,)).permute(0,2,1)
            decoding_manifold_means = self.decoder_model.forward(prior_samples)
            data_samples = torch.distributions.MultivariateNormal(decoding_manifold_means,self.R).sample((1,)).squeeze()
            return prior_samples, data_samples
    
    def fit_latent_posterior_mean(self,nn_model,optimizer,batch_size=10,iterations=5000,p=0.1,epochs=10,show_validations=3,verbose=False,model_desc="Fitting Latents"):
        '''
        data_to_latents_model - neural network that takes in shape (batch_size, neurons, timesteps) -> (batch_size, latents, timesteps)
        '''
        #LSTM_Encoder("cpu",self.latent_dims,self.observation_dims,data_to_init_hidden_state_size=50,hidden_state_to_posterior_size=50)
        validation_losses = []

        train_samples_Z,train_samples_X = self.sample_model(iterations)
        validation_samples_Z,validation_samples_X = self.sample_model(5000)
        train_dataset = DataLoader(TensorDataset(train_samples_X,train_samples_Z),batch_size=batch_size)

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2, T_mult=2, eta_min=0, last_epoch=-1,verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=3)

        def loss_f(predicted_Zs,true_Zs,use_p=True):
            loss = torch.sum(torch.square(predicted_Zs - true_Zs))/(self.latent_dims*self.timesteps*predicted_Zs.shape[0])
            var_term = p*torch.square(torch.norm(predicted_Zs)-torch.norm(true_Zs))/(self.latent_dims*self.timesteps*predicted_Zs.shape[0])
            if use_p:
                loss += var_term
            return loss

        for epoch in tqdm(range(epochs),desc=model_desc,colour="green"):
            epoch_loss = np.zeros(1)
            for (i,(batch_Xs,batch_Zs)) in enumerate(train_dataset):
                optimizer.zero_grad()
                predicted_Zs = nn_model.forward(batch_Xs)
                loss = loss_f(predicted_Zs,batch_Zs,use_p=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.clone().detach().numpy()
            #scheduler.step(epoch)
            if verbose:
                print(f"epoch train loss:{epoch_loss[0]/epochs}")

            with torch.no_grad():

                for l in range(self.latent_dims):
                    training_pred_Z = predicted_Zs[0,:,l]
                    training_true_Z = batch_Zs[0,:,l]
                    plt.plot(training_pred_Z,color="red")
                    plt.plot(training_true_Z,color="black")
                    plt.savefig(f"./test_results/posterior_train_latent_{l}.png")
                    plt.cla()

                predicted_Zs = nn_model.forward(validation_samples_X)
                validation_loss = loss_f(predicted_Zs,validation_samples_Z,use_p=False)

                if verbose:
                    print(f"epoch validation loss: {validation_loss}")
                validation_losses.append(validation_loss.clone().detach().numpy())

                for i in range(show_validations):
                    for l in range(self.latent_dims):
                        plt.plot(predicted_Zs[i,:,l],color="red",label="post-train")
                        plt.plot(validation_samples_Z[i,:,l],color="black",label="correct latent")
                        plt.legend()
                        plt.savefig(f"./test_results/posterior_pred_{i}_latent_{l}.png")
                        plt.cla()
            scheduler.step(validation_loss)

        return validation_losses

        #prior_samples = torch.distributions.MultivariateNormal(torch.zeros(timesteps),Ks.permute(2,0,1)).rsample((batch_size,samples))