import torch.nn as nn
import torch
from gpml_data_struct import GPMLDataStruct
from utils.encoders import FeedforwardVAEEncoder
from gpml_vae_fit import GPML_VAE
from injective_nn import ConvexFeedforward
import matplotlib.pyplot as plt
import torch.nn.modules as Module
import math

class FeedforwardNNDecoder(nn.Module):
    #NOTE this is completely untested!
    def __init__(self,layers_list,latent_dims,observed_dims,device="cpu"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = latent_dims
        input_size = latent_dims
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                #assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.layers.append(nn.Linear(input_size,observed_dims))
        self.to(device)

    def forward(self,input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

class HighDParabolaDecoder(nn.Module):
    def __init__(self,latent_dims,observation_dims,device="cpu",alpha=1.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha,requires_grad=True,device=device))
        self.parabola_to_embedding_layer = nn.Linear(latent_dims+1,observation_dims)
        self.parabola_mixing = nn.Linear(latent_dims,latent_dims)

    def forward(self,Z):
        '''takes in a k dimensional latent variable and embeds it into an n d parabola
        Z - tensor of shape [batch_size,samples, timesteps, dims_in]
        returns X a tensor of shape [batch_size, samples, timesteps, dims_in+1]
        '''
        #mixing_Z = self.parabola_mixing(Z)
        extra_dim = torch.linalg.norm(Z,dim=-1)**2
        parabola = torch.cat((Z,self.alpha*extra_dim.unsqueeze(-1)),dim=-1)
        return self.parabola_to_embedding_layer(parabola)

class LinearDecoder(nn.Module):
    def __init__(self,latent_dim,observed_dim,init_vals = None,device="cpu"):
        super().__init__()
        self.weights = nn.Linear(latent_dim,observed_dim,device=device)
        if not init_vals is None:
            with torch.no_grad():
                self.weights.weight.copy_(init_vals)
    def forward(self,z):
        return self.weights(z)

def dataset_constructor(forward_model,latent_dims,observed_dims,time_start=0.0,time_end=10.0,time_divisions=10,
            R_diag = [0.1,0.1,0.1],taus=[2.0,5.0],signal_sds = 0.99,
            noise_sds = 0.01,samples = 1000,device="cpu"):
    times = torch.linspace(time_start,time_end,time_divisions,device=device,requires_grad=False)
    true_model = GPMLDataStruct(latent_dims,observed_dims,taus,R_diag,
                                [signal_sds]*latent_dims,[noise_sds]*latent_dims,
                                forward_model)
    Z,X = true_model.sample_model(samples,times)
    variable_dict = {"times":times,"X":X,"taus":torch.tensor(taus,device=device,requires_grad=False),
                     "R_diag":torch.tensor(R_diag,device=device,requires_grad=False),"Z":Z,"model":true_model}
    return variable_dict

class LinearEncoder(nn.Module):
    def __init__(self,latent_dim,observed_dim,timesteps,device="cpu"):
        super().__init__()
        self.weights = nn.Linear(observed_dim*timesteps,*latent_dim*timesteps,device=device)
        self.latent_dims = latent_dim
        self.observed_dims = observed_dim
    def forward(self,x):
        #x has shape batch, observed_dims, timesteps
        batch,timesteps,observed_dims = x.shape
        x_in = x.reshape((batch, observed_dims*timesteps))
        x_out = self.weights(x_in)
        out = x_out.reshape((batch,timesteps,2*self.latent_dims))
        means = out[:,:,self.latent_dims:]
        sds = out[:,:,:self.latent_dims]
        sds_tensor = torch.stack([torch.diag_embed(sds[batch_i,:,:].T**2 + 0.0001) for batch_i in range(sds.shape[0])])
        corr_sds = sds_tensor.permute(0,2,3,1)
        return means,corr_sds

class FeedforwardEncoder(nn.Module):
    def __init__(self,layers_list,latent_dims,observed_dims,
                 timesteps,device="cpu",rank=0):
        super().__init__()
        self.input_dims = observed_dims*timesteps
        self.output_dims = latent_dims*timesteps*(2+rank)
        self.layers = nn.ModuleList()
        self.latent_dims = latent_dims
        self.rank = rank
        input_size = self.input_dims
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                assert isinstance(activation, nn.Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.layers.append(nn.Linear(input_size,self.output_dims))
        self.to(device)

    def forward(self,x):
        batch,timesteps,observed_dims = x.shape
        u = x.reshape((batch, observed_dims*timesteps))
        for layer in self.layers:
            u = layer(u)
        out = u.reshape((batch,timesteps,(2+self.rank)*self.latent_dims))
        means = out[:,:,:self.latent_dims]
        sds = out[:,:,self.latent_dims:2*self.latent_dims]
        sds_tensor = torch.stack([torch.diag_embed(sds[batch_i,:,:].T**2 + 0.0001) for batch_i in range(sds.shape[0])])
        corr_sds = sds_tensor.permute(0,2,3,1)
        for r in range(self.rank):
            rank_vecs = out[:,:,(r+2)*self.latent_dims:(r+3)*self.latent_dims]
            corr_sds += torch.einsum("bil,bjl -> bijl",rank_vecs,rank_vecs)
        return means,corr_sds

class TestEncoder(nn.Module):
    def __init__(self,latent_dims,times,model:GPMLDataStruct,device="cpu"):
        super().__init__()
        self.model = model
        self.kernels = model.sde_kernel_matrices(times)
        self.means = torch.zeros((len(times),latent_dims,))
        self.times = times
        self.latent_dims = latent_dims

    def forward(self,x):
        batch_size = x.shape[0]
        self.means = torch.zeros((batch_size,len(self.times),self.latent_dims))
        out_kernels = torch.stack(batch_size*[self.kernels])
        return self.means, out_kernels    


def steady_ramp_scheduler(zero_epoch,full_value_epoch):
    def steady_ramp(epoch):
        if epoch < zero_epoch:
            return 0
        else:
            return min((epoch-zero_epoch/full_value_epoch,1))
    return steady_ramp

def scaled_inv_ramp_scheduler(zero_epoch,full_value_epoch,scaling):
    def steady_ramp(epoch):
        if epoch < zero_epoch:
            return scaling
        else:
            out =  1 - min((epoch - zero_epoch)/full_value_epoch,1)
            return out*scaling
    return steady_ramp

def cycle_scheduler(epochs,cycle_length,stop_cycle=0.9,ratio=0.5):
    full_value_epochs = math.floor(epochs*stop_cycle)
    num_cycles = math.floor(full_value_epochs/cycle_length)
    cycles = 0
    out = []
    for epoch in range(epochs):
        if cycles == num_cycles:
            out.append(1)
        else:
            if epoch%cycle_length == 0:
                cycles += 1
            cycle_dist = (epoch - math.floor(epoch/cycle_length)*cycle_length)/(ratio*cycle_length)
            cycle_dist = min(cycle_dist,1)
            out.append(cycle_dist)
    return out

def cycler(epochs, cycle_length,stop_cycle=0.9,ratio=0.5):
    out = cycle_scheduler(epochs,cycle_length,stop_cycle,ratio=0.5)
    def n_one(epoch):
        return out[epoch]
    return n_one

def const_scheduler(c):
  def out(epoch):
    return c
  return out

def get_mean_var(data):
    #data = [batch_size, timesteps, latents]
    batch_size,timesteps,latents = data.shape
    batch_collapsed = data.reshape(batch_size*timesteps,latents)
    variances = torch.var(batch_collapsed,dim=0)
    means = torch.mean(batch_collapsed,dim=0)
    return means, variances
def plot_dataset(dataset, index):

    plt.title("Example Latent Sample")
    plt.xlabel("Time")
    plt.ylabel("Z")
    latent = dataset["Z"][index]
    for i in range(latent.shape[1]):
        plt.plot(dataset["times"],latent[:,i],label=f"Tau: {dataset['taus'][i]}")
    plt.legend()
    plt.savefig(f"./logs/Z_sample{index}.png")
    plt.cla()

    plt.title("Example X Sample")
    plt.xlabel("Time")
    plt.ylabel("X")
    X = dataset["X"][index]
    for i in range(X.shape[1]):
        plt.plot(dataset["times"],X[:,i],label=f"neuron {i}")
    plt.legend()
    plt.savefig(f"./logs/X_sample{index}.png")
    plt.cla()

def working_1d(true_tau = 1.0):
    device = "cpu"
    latent_dims = 1
    observation_dims = 2
    true_taus = [true_tau]
    true_R_diag = [0.001]*observation_dims
    with torch.no_grad():
        decoder = LinearDecoder(latent_dims,observation_dims)
        linear_dataset = dataset_constructor(decoder,latent_dims,observation_dims,
                                             taus = true_taus,R_diag=true_R_diag,samples=500,time_divisions=50,time_end=20.0)
    #linear encoder as is works
    #LinearEncoder(1,2,len(linear_dataset["times"]))
    encoder = LinearEncoder(latent_dims,observation_dims,len(linear_dataset["times"]))#FeedforwardEncoder([(200,nn.LeakyReLU()),(200,nn.LeakyReLU())],latent_dims,observation_dims,len(linear_dataset["times"]))#FeedforwardVAEEncoder([],latent_dims,observation_dims,len(linear_dataset["times"]),device=device)
    fit_decoder =  LinearDecoder(latent_dims,observation_dims)
    vae = GPML_VAE(device,latent_dims,observation_dims,linear_dataset["times"],encoder,decoder,train_decoder=False,
                initial_R_diag=true_R_diag,train_R_diag=False,train_taus=True,train_encoder=True)

    vae.fit(linear_dataset["X"],None,lr=0.001,epochs=100000,batch_size=20,approx_elbo_loss_samples=1,hyper_scheduler=None,grad_clip=5.0)


def tester():
    device = "cpu"
    latent_dims = 2
    observation_dims = 5
    true_taus = [2.0,5.0]
    true_R_diag = [0.001]*observation_dims
    signal_sds = 0.99
    noise_sds = 1 - signal_sds

    with torch.no_grad():
        forward_model = LinearDecoder(latent_dims,observation_dims)
        linear_dataset = dataset_constructor(forward_model,latent_dims, observed_dims=observation_dims,
                                             taus = true_taus,R_diag=true_R_diag,samples=500,time_divisions=40,time_end=20.0,
                                             signal_sds=signal_sds,noise_sds=noise_sds)
    linear_decoder = forward_model
    plot_dataset(linear_dataset,0)
    #test_encoder = TestEncoder(latent_dims,linear_dataset["times"],linear_dataset["model"])
    #linear encoder as is works
    #LinearEncoder(1,2,len(linear_dataset["times"]))
    encoder = FeedforwardEncoder([(200,nn.LeakyReLU()),(200,nn.LeakyReLU())],latent_dims,observation_dims,len(linear_dataset["times"]))#LinearEncoder(latent_dims,observation_dims,len(linear_dataset["times"]))#FeedforwardEncoder([(200,nn.LeakyReLU()),(200,nn.LeakyReLU())],latent_dims,observation_dims,len(linear_dataset["times"]))
    fit_decoder =  LinearDecoder(latent_dims,observation_dims)
    vae = GPML_VAE(device,latent_dims,observation_dims,linear_dataset["times"],encoder,forward_model,
                   train_decoder=False,initial_R_diag=true_R_diag,train_R_diag=False,train_taus=True,
                   train_encoder=True,signal_sds=[signal_sds]*latent_dims,noise_sds=[noise_sds]*latent_dims)
    vae.fit(linear_dataset["X"],None,lr=0.001,epochs=100000,batch_size=20,approx_elbo_loss_samples=1,
            grad_clip=20.0,hyper_scheduler=steady_ramp_scheduler(0,500))

def tester_parabola():
    device = "cpu"
    latent_dims = 1
    observation_dims = 5
    true_taus = [4.0]
    true_R_diag = 0.0001
    hetero_r = True
    if hetero_r:
        true_R_diag = [true_R_diag]*observation_dims
    signal_sds = 0.99
    noise_sds = 1 - signal_sds

    with torch.no_grad():
        forward_model = HighDParabolaDecoder(latent_dims,observation_dims,device=device,alpha=1.0)
        parabola_dataset = dataset_constructor(forward_model,latent_dims, observation_dims,taus = true_taus,R_diag=true_R_diag,
                                               samples=1000,time_divisions=30,time_end=20.0,signal_sds=signal_sds,
                                               noise_sds=noise_sds)
    #plots latent
    plot_dataset(parabola_dataset,0)
    #test_encoder = TestEncoder(latent_dims,parabola_dataset["times"],parabola_dataset["model"])
    #linear encoder as is works
    #LinearEncoder(1,2,len(linear_dataset["times"]))
    mean, var = get_mean_var(parabola_dataset["X"])
    #encoder = FeedforwardEncoder([(2000,nn.LeakyReLU()),(2000,nn.LeakyReLU())],
    #                             latent_dims,observation_dims,len(parabola_dataset["times"]))#
    encoder = FeedforwardEncoder([(3000,nn.LeakyReLU()),(3000,nn.LeakyReLU())],
                                 latent_dims,observation_dims,len(parabola_dataset["times"]),rank=1)
    fit_decoder = FeedforwardNNDecoder([(30, nn.LeakyReLU())],latent_dims=latent_dims,observed_dims=observation_dims)
    #FeedforwardNNDecoder([(30, nn.LeakyReLU()),(30, nn.LeakyReLU())],latent_dims=latent_dims,observed_dims=observation_dims)
    #FeedforwardNNDecoder([(80, nn.LeakyReLU())],latent_dims=latent_dims,observed_dims=observation_dims)
    #FeedforwardNNDecoder([(30, nn.LeakyReLU()),(30, nn.LeakyReLU())],latent_dims=latent_dims,observed_dims=observation_dims)# HighDParabolaDecoder(latent_dims,observation_dims,device=device)
    #FeedforwardNNDecoder([(100, nn.LeakyReLU()),(100, nn.LeakyReLU())],latent_dims=latent_dims,observed_dims=observation_dims)
    #FeedforwardNNDecoder([(30, nn.LeakyReLU()),(30, nn.LeakyReLU())],latent_dims=latent_dims,observed_dims=observation_dims)
    #ConvexFeedforward([30],[30],latent_dims,observation_dims)
    #if didnt work, try with correct decoder
    #try fixing R and decoder, see if taus go correct
    #try off diagonal on encoder
    vae = GPML_VAE(device,latent_dims,observation_dims,parabola_dataset["times"],encoder,fit_decoder,
                   train_decoder=True,train_R_diag=True,train_taus=True,initial_R_diag=[0.01]*observation_dims,
                   train_encoder=True,signal_sds=[signal_sds]*latent_dims,noise_sds=[noise_sds]*latent_dims,initial_taus=[2.0]*latent_dims)
    vae.compile()
    epochs = 100000
    #if this thing fits, could try dropping the low rank and running for a bunch of epochs, behavior sort of looks the same
    vae.fit(parabola_dataset["X"],None,lr=1e-7,epochs=epochs,batch_size=100,approx_elbo_loss_samples=1,grad_clip=1000.0,no_tau_epochs=0,
            hyper_scheduler=None,mm=0.7,tau_lr=None,plotting_true_taus=true_taus,plotting_true_R_diag=true_R_diag,log_save_name="Adam")
    #correct fit
    #e-7, 3000,3000,mm=0.6,parabola,fixed r,rank 1
    #correct fit with fixed decoder
    #vae.fit(parabola_dataset["X"],None,lr=1e-7,epochs=100000,batch_size=20,approx_elbo_loss_samples=1,grad_clip=100000.0,no_tau_epochs=0,
     #       hyper_scheduler=None,mm=0.6,tau_lr=None)
torch.manual_seed(0)
tester_parabola()
#use more momentum to get over hump
#works for lr = e-5
#rs go correct to ll of -500
#maybe plot p(Z|X)
#tonight try:[(3000,nn.LeakyReLU()),(2000,nn.LeakyReLU()),(2000,nn.LeakyReLU())], lr=0.00001
#make sure dims on encoder match up properlys
#maybe parabola is behaving poorly, ll is much higher
#make sure 
#tester()
#tester_parabola()
#maybe grad clip is messing with?
#52 converged to 2 with 5k
#maybe look at K, see if it lines up
#tester_parabola()
#too high of learning rate caused tau to collapse
#doesnt converge ofr high d parabola, seems to be encoder/tau related
#maybe clamp r to a tenth of the data variance
# decoder doesnt fit when R is unclamped
#tomorrow, unclamp r diag, try in higher d
#ll leveling out early usually means the encoder needs to be bigger
#seemed to work for observed = 8
#too big of alpha can mess it up!
#working_1d(3.0)
#enforce small Rs?
#maybe add more observation nodes, might constrain a little better
#are the noise terms equal?
#r diag strongly influences
#so, there was mismatch in the signal and noise sds between train and test
#plot the first predicted latent vs true latent
#pass correct prior dist into the output of the decoder, this should give a 0 kl divergence