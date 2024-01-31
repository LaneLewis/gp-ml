import unittest
import torch
from utils.gaussian_process import sde_kernel_matrices
from utils.elbo_loss import kl_divergence,sample_encoding_dist,log_likelihood_loss,approx_elbo_loss
from utils.decoders import FeedforwardNNDecoder,FeedforwardNNDecoderTau,ParabolaDecoder
from torch import nn,optim
from utils.datasets import sample_assumed_distribution,parabola_imbedded_dataset1,parabola_imbedded_dataset
from utils.encoders import LSTM_Encoder,LSTM_Encoder_tau,FeedforwardVAEEncoder
from gpml_vae import GPML_VAE
from gpml_vae_alt_train import GPML_VAE_Alt
import pickle as pkl
class TestGPMLTotal(unittest.TestCase):

    def test_adding_tau(self):
        latent_dims = 1
        observed_dims = 2
        Z,X,times,parameters = parabola_imbedded_dataset1(samples=200,times=torch.linspace(0,1,20),tau=2.0,R_diag=[0.1,0.1])

        taus = list(parameters["taus"].detach().numpy())
        signal_sds = list(parameters["signal_sds"].detach().numpy())
        noise_sds = list(parameters["noise_sds"].detach().numpy())
        R_diag = list(parameters["R_diag"])

        encoding_model = LSTM_Encoder_tau("cpu",latent_dims,observed_dims,data_to_init_hidden_state_size=20,hidden_state_to_posterior_size=20)
        encoding_optimizer = optim.Adam(encoding_model.parameters(),lr=0.005)
        decoder_tau = False
        decoding_model = FeedforwardNNDecoder([(10,nn.ReLU())],latent_dims,observed_dims)
        decoding_optimizer = optim.Adam(decoding_model.parameters(),lr=0.005)

        initial_taus = [10.0]
        initial_R_diag = [1.0,1.0]
        gpml = GPML_VAE_Alt("cpu",latent_dims,observed_dims,times,encoding_model,decoding_model,
                            initial_taus=initial_taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=initial_R_diag)
        pre_train_loss = torch.linalg.norm(gpml.predict_Z(X,encoder_pass_taus=True) - Z)
        pre_train_reconstruction = gpml.reconstruction_loss(X,encoder_pass_taus=True,decoder_pass_taus=decoder_tau)
        #kernel_fit_pre = torch.linalg.norm(gpml.predict_Z_std(X) - gpml.kernel_matrices)
        #mean_fit_pre = torch.linalg.norm(gpml.predict_Z(X))
        gpml.fit(Z,X,encoding_optimizer,decoding_optimizer,optimize_taus=True,optimize_R=True,batch_size=10,epochs=2000,
                 approx_elbo_loss_samples=100,tau_lr=0.01,R_diag_lr=0.05,loss_hyperparameter=10.0,encoder_pass_taus=True,decoder_pass_taus=decoder_tau)
        #kernel_fit_post = torch.linalg.norm(gpml.predict_Z_std(X) - gpml.kernel_matrices)
        #mean_fit_post = torch.linalg.norm(gpml.predict_Z(X))

        post_train_loss = torch.linalg.norm(gpml.predict_Z(X,encoder_pass_taus=True) - Z)
        post_train_reconstruction = gpml.reconstruction_loss(X,encoder_pass_taus=True,decoder_pass_taus=decoder_tau)
        print(f"correct R:{R_diag}")
        print(f"before training R_diag: {initial_R_diag}")
        print(f"after training R_diag {gpml.R_diag}")
        print(f"correct tau:{taus}")
        print(f"before training tau: {initial_taus}")
        print(f"after training tau {gpml.taus}")
        print(f"before training Z: {pre_train_loss}")
        print(f"after training reconstruction Z: {post_train_loss}")
        print(f"before training X: {pre_train_reconstruction}")
        print(f"after training X: {post_train_reconstruction}")

    def test_vae_with_nn_decoder(self):
        latent_dims = 1
        observed_dims = 2
        timesteps = 50
        Z,X,times,parameters = parabola_imbedded_dataset1(samples=5000,times=torch.linspace(0,10,timesteps),tau=5.0)

        taus = [30.0]
        signal_sds = list(parameters["signal_sds"].detach().numpy())
        noise_sds = list(parameters["noise_sds"].detach().numpy())
        R_diag = list(parameters["R_diag"])

        encoding_model = FeedforwardVAEEncoder([(100,nn.LeakyReLU()),(50,nn.LeakyReLU()),(50,nn.LeakyReLU())],latent_dims,observed_dims,timesteps)
        encoding_optimizer = optim.Adam(encoding_model.parameters(),lr=0.005)

        decoding_model = FeedforwardNNDecoder([(10,nn.Sigmoid())],latent_dims,observed_dims)
        decoding_optimizer = optim.Adam(decoding_model.parameters(),lr=0.005)

        gpml = GPML_VAE("cpu",latent_dims,observed_dims,times,encoding_model,decoding_model,initial_taus=taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=R_diag)
        pre_train_loss = torch.linalg.norm(gpml.predict_Z(X) - Z)
        pre_train_reconstruction = gpml.reconstruction_loss(X)
        #kernel_fit_pre = torch.linalg.norm(gpml.predict_Z_std(X) - gpml.kernel_matrices)
        #mean_fit_pre = torch.linalg.norm(gpml.predict_Z(X))
        gpml.fit(Z,X,encoding_optimizer,decoding_optimizer,optimize_taus=True,optimize_R=True,batch_size=10,epochs=1500,approx_elbo_loss_samples=20)
        #kernel_fit_post = torch.linalg.norm(gpml.predict_Z_std(X) - gpml.kernel_matrices)
        #mean_fit_post = torch.linalg.norm(gpml.predict_Z(X))
        post_train_loss = torch.linalg.norm(gpml.predict_Z(X) - Z)
        post_train_reconstruction = gpml.reconstruction_loss(X)

        print(f"before training Z: {pre_train_loss}")
        print(f"after training reconstruction Z: {post_train_loss}")
        print(f"before training X: {pre_train_reconstruction}")
        print(f"after training X: {post_train_reconstruction}")


    def test_vae_on_parabola(self):
        latent_dims = 2
        observed_dims = 3
        timesteps = 50
        Z,X,times,parameters = parabola_imbedded_dataset(samples=200,times=torch.linspace(0,10,timesteps))
        parabola_alpha = parameters["alpha"]

        taus = list(parameters["taus"].detach().numpy())
        signal_sds = list(parameters["signal_sds"].detach().numpy())
        noise_sds = list(parameters["noise_sds"].detach().numpy())
        R_diag = list(parameters["R_diag"])

        decoding_model = ParabolaDecoder(latent_dims,parameters["alpha"])
        encoding_model = FeedforwardVAEEncoder([(100,nn.LeakyReLU()),(50,nn.LeakyReLU()),(50,nn.LeakyReLU())],latent_dims,observed_dims,timesteps)

        encoding_optimizer = optim.Adam(encoding_model.parameters(),lr=0.001)
        gpml = GPML_VAE("cpu",2,3,times,encoding_model,decoding_model,initial_taus=taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=R_diag)
        gpml.fit(Z,X,encoding_optimizer,optimize_taus=False,optimize_R=False,batch_size=10,epochs=350)

        #saves model for future tests
        with open("./tests/posterior_test_model.pkl","wb") as f:
            pkl.dump(gpml,f)
        #print(z_norm_1)
        #print(z_norm_2)

if __name__ == "__main__":
    unittest.main()