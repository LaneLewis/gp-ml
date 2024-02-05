import torch 
from utils.decoders import ParabolaDecoder
from utils.gaussian_process import sde_kernel_matrices
def parabola_imbedded_dataset(samples=100,tau1=0.5,tau2=0.1,signal_sds1=0.99,signal_sds2=0.99,
                              noise_sds1=0.01,noise_sds2=0.01,parabola_alpha=1.0,
                              times=torch.linspace(0.0,1.0,10),R_diag=[0.1,0.1,0.1]):
    with torch.no_grad():
        taus = torch.tensor([tau1,tau2])
        signal_sds = torch.tensor([signal_sds1,signal_sds2])
        noise_sds = torch.tensor([noise_sds1,noise_sds2])
        Ks = sde_kernel_matrices(times,taus,signal_sds,noise_sds)
        decoder = ParabolaDecoder(parabola_alpha)
        R = torch.diag(torch.tensor(R_diag))
        Z,X = sample_assumed_distribution(decoder.forward,times,R,Ks,samples)
        parameters = {"taus":taus,"signal_sds":signal_sds,"noise_sds":noise_sds,"alpha":parabola_alpha,"R_diag":R_diag}
        return Z,X,times,parameters
    
def parabola_imbedded_dataset1(samples=100,tau=5.0,signal_sds=0.99,noise_sds=0.01,parabola_alpha=1.0,times=torch.linspace(0.0,100.0,500),R_diag=[0.1,0.1]):
    with torch.no_grad():
        taus = torch.tensor([tau])
        signal_sds = torch.tensor([signal_sds])
        noise_sds = torch.tensor([noise_sds])
        Ks = sde_kernel_matrices(times,taus,signal_sds,noise_sds)
        decoder = ParabolaDecoder(2,parabola_alpha)
        R = torch.diag(torch.tensor(R_diag))
        Z,X = sample_assumed_distribution(decoder.forward,times,R,Ks,samples)
        parameters = {"taus":taus,"signal_sds":signal_sds,"noise_sds":noise_sds,"alpha":parabola_alpha,"R_diag":R_diag}
        return Z,X,times,parameters
    
@torch.no_grad
def sample_assumed_distribution(decoder_func,times,R_diag,taus,samples,signal_sd = 0.99,noise_sd = 0.01):
    #defines the variables
    latent_dims = len(taus)
    observed_dims = len(R_diag)
    taus = torch.tensor(taus)
    R_diag = torch.tensor(R_diag)
    R = torch.diag(R_diag)
    signal_sds = torch.tensor(latent_dims*[signal_sd])
    noise_sds = torch.tensor(latent_dims*[noise_sd])

    Ks = sde_kernel_matrices(times,taus,signal_sds,noise_sds)
    timesteps = times.shape[0]
   # Ks.permute(2,0,1)
    prior_samples_broadcast = torch.distributions.MultivariateNormal(torch.zeros(timesteps),Ks.permute(2,0,1)).sample((samples,))
    prior_samples = torch.permute(prior_samples_broadcast, [0,2,1])
    #adds batch to samples to standardize it
    added_dim = prior_samples.expand(1,*prior_samples.shape)
    out_means = decoder_func(added_dim).squeeze(0)
    out_samples = torch.distributions.MultivariateNormal(out_means,R).sample((1,))
    return prior_samples,out_samples.squeeze(0)