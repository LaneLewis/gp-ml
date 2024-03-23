import unittest
import torch
from utils.decoders import FeedforwardNNDecoder,ParabolaDecoder
from torch import nn,optim
from utils.datasets import parabola_imbedded_dataset1
from gpml_ll_fit import GPML_LLF
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from utils.latent_prediction import FeedforwardPrediction
from nn_optim import GPML_NN_LLF
from torch import nn

class Test_LL_Fit(unittest.TestCase):

    def test_LL_fit_with_nn_decoder(self):
        latent_dims = 1
        observed_dims = 2
        tau_test = 1.5
        timesteps = 80
        Z,X,times,parameters = parabola_imbedded_dataset1(samples=5000,times=torch.linspace(0.0,10.0,timesteps),tau=tau_test)
        taus = [10.0]#list(parameters["taus"].detach().numpy())
        signal_sds = list(parameters["signal_sds"].clone().detach().numpy())
        noise_sds = list(parameters["noise_sds"].clone().detach().numpy())
        R_diag = [5.0,5.0]#list(parameters["R_diag"])
        train_epochs = 1
        true_taus = parameters["taus"].clone().detach().numpy()
        true_R_diag = parameters["R_diag"].clone().detach().numpy()
        decoding_model = FeedforwardNNDecoder([(5,nn.Sigmoid())],latent_dims,observed_dims)
        decoding_optimizer = optim.Adam(decoding_model.parameters(),lr=0.005)
        gpml = GPML_LLF("cpu",latent_dims,observed_dims,times,decoding_model,initial_taus=taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=R_diag)
        gpml.fit_generative_model(X,decoding_optimizer,optimize_taus=True,optimize_R=True,batch_size=20,epochs=train_epochs,approx_log_likelihood_loss_samples=25,tau_lr=0.1,R_diag_lr=0.05,print_batch_values=True)
        total_models = 3
        for i in range(total_models):
            prediction_model = FeedforwardPrediction([(100,nn.LeakyReLU()),(50,nn.LeakyReLU()),(50,nn.LeakyReLU())],latent_dims,observed_dims,timesteps)
            prediction_optimizer = optim.Adam(prediction_model.parameters(),lr = 0.0005)
            validation_loss = gpml.fit_latent_posterior_mean(prediction_model,prediction_optimizer,10,10000,p=0.0,epochs=10,model_desc=f"model: {i} / {total_models} ")
            print(f" model {i} {validation_loss[-1]}")
            if i == 0:
                best_validation_loss = validation_loss[-1]
                best_prediction_model = prediction_model
            else:
                if validation_loss[-1] < best_validation_loss:
                    best_prediction_model = prediction_model
                    best_validation_loss = validation_loss[-1]
        taus_train_matrix = np.stack(gpml.taus_trajectory)
        R_diag_train_matrix = np.stack(gpml.R_diag_trajectory)
        plt.title("Tau Over Epochs")
        plt.plot(range(0,train_epochs),taus_train_matrix,marker="o",color="red",label="Tau Hat")
        plt.axhline(y=true_taus[0],color="black",label="True Tau")
        plt.xlabel("Epochs")
        plt.ylabel("Tau")
        plt.savefig("./test_results/taus_trajectory.png")
        plt.cla()

    def test_LL_fit_nn_dataset(self):
        device= "cpu"
        #true parameters
        latent_dims = 2
        observed_dims = 3
        times = torch.linspace(0.0,10.0,10,device=device)
        true_R_diag = [5.0]*observed_dims
        true_taus = [5.0,9.0]
        true_signal_sds = latent_dims*[0.98]
        true_noise_sds = latent_dims*[0.02]
        alpha = 10.0
        true_decoder_model = ParabolaDecoder(alpha,device=device)#FeedforwardNNDecoder([(3,nn.Sigmoid())],latent_dims,observed_dims)
        training_samples = 1000
        true_model = GPML_LLF(device,latent_dims,observed_dims,times,true_decoder_model,true_signal_sds,true_noise_sds,initial_taus=true_taus,initial_R_diag=true_R_diag)
        #Z,X = sample_assumed_distribution(true_decoder_model.forward,times,true_R_diag,true_taus,training_samples)
        Z,X = true_model.sample_model(training_samples)
        #initial parameters
        model_latent_dims = 2
        model_taus = [5.0,9.0]
        signal_sds = model_latent_dims*[0.98]
        noise_sds = model_latent_dims*[0.02]
        model_R_diag = [5.0,5.0,5.0]
        #Fitting parameters
        generative_fit_epochs = 1000
        decoding_model = FeedforwardNNDecoder([(20,nn.LeakyReLU())],model_latent_dims,observed_dims,device=device)
        decoding_optimizer = optim.Adam(decoding_model.parameters(),lr=0.0001)

        gpml = GPML_LLF("cpu",model_latent_dims,observed_dims,times,decoding_model,initial_taus=model_taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=model_R_diag)
        gpml.fit_generative_model(X,decoding_optimizer,optimize_taus=False,optimize_R=False,batch_size=20,epochs=generative_fit_epochs,approx_log_likelihood_loss_samples=1000,
                                  tau_lr=0.005,R_diag_lr=0.05,print_batch_values=True)
    def test_NN_optim(self):
        device= "cpu"
        #true parameters
        latent_dims = 2
        observed_dims = 3
        times = torch.linspace(0.0,10.0,10,device=device,requires_grad=False)
        true_R_diag = [5.0]*observed_dims
        true_taus = [5.0,9.0]
        true_signal_sds = latent_dims*[0.99]
        true_noise_sds = latent_dims*[0.01]
        alpha = 10.0
        true_decoder_model = ParabolaDecoder(alpha,device=device)#FeedforwardNNDecoder([(3,nn.Sigmoid())],latent_dims,observed_dims)
        training_samples = 1000
        true_model = GPML_LLF(device,latent_dims,observed_dims,times,true_decoder_model,true_signal_sds,true_noise_sds,initial_taus=true_taus,initial_R_diag=true_R_diag)
        #Z,X = sample_assumed_distribution(true_decoder_model.forward,times,true_R_diag,true_taus,training_samples)
        Z,X = true_model.sample_model(training_samples)
        #initial parameters
        model_latent_dims = 2
        model_taus = [10.0,10.0]
        signal_sds = model_latent_dims*[0.99]
        noise_sds = model_latent_dims*[0.01]
        model_R_diag = [15.0,15.0,15.0]
        #Fitting parameters
        generative_fit_epochs = 1000
        decoding_model = ParabolaDecoder(8.0,device=device)#FeedforwardNNDecoder([(20,nn.LeakyReLU()),(20,nn.LeakyReLU())],model_latent_dims,observed_dims,device=device)
        nn_optimizer_layers = [(50,nn.LeakyReLU()),(50,nn.LeakyReLU())]
        decoding_optimizer = optim.Adam(decoding_model.parameters(),lr=0.0001)
        #exp model params
        gpml = GPML_NN_LLF("cpu",model_latent_dims,observed_dims,times,decoding_model,initial_taus=model_taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=model_R_diag,NN_layers=nn_optimizer_layers)
        gpml.fit_generative_model(X,decoding_optimizer,optimize_taus=True,optimize_R=True,batch_size=20,epochs=generative_fit_epochs,approx_log_likelihood_loss_samples=10000,
                                  tau_lr=0.001,R_diag_lr=0.001,print_batch_values=True)

if __name__ == "__main__":
    unittest.main()