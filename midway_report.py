import torch
from gpml_data_struct import GPMLDataStruct
from utils.decoders import FeedforwardNNDecoder,ParabolaDecoder,HighDParabolaDecoder
from utils.encoders import FeedforwardVAEEncoder
from gpml_vae_fit import GPML_VAE
from torch import nn,optim
import matplotlib.pyplot as plt
#from gpml_ll_fit import GPML_LLF
from mc_ll_fit import GPML_LLF
import pickle as pkl
import numpy as np
from utils.latent_prediction import FeedforwardPrediction
from torch import nn
import math
from injective_nn import ConvexFeedforward

def two_latent_three_pop_dataset(time_start=0.0,time_end=10.0,time_divisions=10,
            R_diag = [5.0,5.0,5.0],taus=[2.0,5.0],signal_sds = [0.98,0.98],
            noise_sds = [0.02,0.02],alpha=10.0,samples = 1000,device="cpu"):
    latent_dims = 2
    observed_dims = 3
    true_decoder_model = ParabolaDecoder(alpha,device=device)
    times = torch.linspace(time_start,time_end,time_divisions,device=device,requires_grad=False)
    true_model = GPML_LLF(device,latent_dims,observed_dims,times,true_decoder_model,
                            signal_sds,noise_sds,taus,R_diag)
    Z,X = true_model.sample_model(samples,times)
    variable_dict = {"times":times,"X":X,"taus":torch.tensor(taus,device=device,requires_grad=False),"R_diag":torch.tensor(R_diag,device=device,requires_grad=False),"Z":Z}
    return variable_dict

def high_d_parabola_dataset(time_start=0.0,time_end=10.0,time_divisions=10,
            R_diag_val = 1.0 ,taus=[2.0,5.0,3.0,7.0,8.0],signal_sds_val = 0.98,
            noise_sds_val = 0.02,alpha=10.0,samples = 1000,device="cpu",observation_dims=10):
    latent_dims = len(taus)
    true_decoder_model = HighDParabolaDecoder(latent_dims,observation_dims,device,alpha)
    times = torch.linspace(time_start,time_end,time_divisions,device=device,requires_grad=False)
    R_diag = [R_diag_val]*observation_dims
    signal_sds = [signal_sds_val]*observation_dims
    noise_sds = [noise_sds_val]*observation_dims
    true_model = GPMLDataStruct(latent_dims,observation_dims,taus,R_diag,signal_sds,noise_sds,true_decoder_model)
    Z,X = true_model.sample_model(samples,times)
    variable_dict = {"times":times,"X":X,"taus":torch.tensor(taus,device=device,requires_grad=False),"R_diag":torch.tensor(R_diag,device=device,requires_grad=False),"Z":Z}
    return variable_dict

def two_latent_three_pop_MC(train_dataset,test_dataset,layers=[(20,nn.GELU())],epochs=100,batch_size=10,initial_taus=[5.0,9.0],
                            initial_R_diag=[5.0,5.0,5.0],signal_sds=[0.98,0.98],noise_sds=[0.02,0.02],lr=0.001,
                            samples=1000,device="cpu",run_save_name="latest"):
    latent_dims = 2
    observed_dims = 3
    decoding_model = FeedforwardNNDecoder(layers,latent_dims,observed_dims,device=device)
    #decoding_model = ParabolaDecoder(10.0)
    gpml = GPML_LLF(device,latent_dims,observed_dims,train_dataset["times"],decoding_model,
                    initial_taus=initial_taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=initial_R_diag,train_decoder=True)
    c_gpml = torch.compile(gpml)
    #add validation set
    c_gpml.fit(train_dataset["X"],test_dataset["X"],batch_size=batch_size,epochs=epochs,approx_log_likelihood_loss_samples=samples)
    with open(f"./midway_models/{run_save_name}_MC_model.pkl","wb") as f:
        pkl.dump(gpml,f)

def steady_ramp_scheduler(full_value_epoch):
    def steady_ramp(epoch):
        return min((epoch/full_value_epoch,1))
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


def two_latent_three_pop_VAE(train_dataset,test_dataset,encoding_model_layers=[(50,nn.GELU()),(50,nn.GELU())],decoding_model_layers=[(20,nn.GELU())],epochs=100,batch_size=10,initial_taus=[5.0,9.0],
                            initial_R_diag=[5.0,5.0,5.0],signal_sds=[0.98,0.98],noise_sds=[0.02,0.02],lr=0.01,
                            samples=10,device="cpu",run_save_name="latest_vae",cycle_length=1000):    
    times = train_dataset["times"]
    latent_dims = 2
    observed_dims = 3
    lr = lr
    encoding_model = FeedforwardVAEEncoder(encoding_model_layers,latent_dims,observed_dims,len(times))
    decoding_model =  ParabolaDecoder(alpha=10.0,device=device)#ConvexFeedforward([10],[10],latent_dims,observed_dims)#FeedforwardNNDecoder(decoding_model_layers,latent_dims,observed_dims)

    #add validation set 
    gpml = GPML_VAE(device,latent_dims,observed_dims,times,encoding_model,decoding_model,initial_taus=initial_taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=initial_R_diag,train_decoder=False)
    c_gpml = torch.compile(gpml)
    c_gpml.fit(train_dataset["X"],None,lr,epochs,batch_size,samples,grad_clip=0.001)
    with open(f"./midway_models/{run_save_name}_VAE_model.pkl","wb") as f:
        pkl.dump(gpml,f)

def const_hyper_scheduler(alpha=0.1):
    def schedule(epoch):
        return alpha
    return schedule

def two_latent_three_pop_Convex(train_dataset,test_dataset,encoding_model_layers=[(50,nn.GELU()),(50,nn.GELU())],decoding_model_layers=[(20,nn.GELU())],epochs=100,batch_size=10,initial_taus=[5.0,9.0],
                            initial_R_diag=[5.0,5.0,5.0],signal_sds=[0.98,0.98],noise_sds=[0.02,0.02],lr=0.001,
                            samples=10,device="cpu",run_save_name="latest_vae",cycle_length=1000):    
    times = train_dataset["times"]
    latent_dims = 2
    observed_dims = 3
    encoding_model = FeedforwardVAEEncoder(encoding_model_layers,latent_dims,observed_dims,len(times))
    encoding_optimizer = optim.Adam(encoding_model.parameters(),lr=0.001,amsgrad=True)

    decoding_model = ConvexFeedforward([10],[10],latent_dims,observed_dims)
    decoding_optimizer = optim.Adam(decoding_model.parameters(),lr=0.0001,amsgrad=True)
    #add validation set
    gpml = GPML_VAE(device,latent_dims,observed_dims,times,encoding_model,decoding_model,initial_taus=initial_taus,signal_sds=signal_sds,noise_sds=noise_sds,initial_R_diag=initial_R_diag)
    gpml.fit(train_dataset["X"],test_dataset["X"],encoding_optimizer,decoding_optimizer,optimize_taus=True,optimize_R=True,batch_size=batch_size,epochs=epochs,approx_elbo_loss_samples=samples,hyper_scheduler=const_hyper_scheduler(1.0))
    with open(f"./midway_models/{run_save_name}_VAE_model.pkl","wb") as f:
        pkl.dump(gpml,f)

def graph_MC(run_save_name):

    with open(f"./midway_models/train_dataset.pkl","rb") as f:
        dataset = pkl.load(f)
    with open(f"./midway_models/{run_save_name}_MC_model.pkl","rb") as f:
        gpml = pkl.load(f)

    training_loss = np.array(gpml.training_loss)
    taus_trajectory = np.array(gpml.taus_trajectory)
    R_diag_trajectory = np.array(gpml.R_diag_trajectory)
    testing_loss = np.array(gpml.validation_loss)

    plt.title("MC Estimator: Train/Test Loss Over Epochs")
    plt.plot(training_loss,label="Train Loss",color="black")
    plt.plot(testing_loss,label="Test Loss",color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig(f"./midway_report_figs/{run_save_name}_train_loss.png")
    plt.cla()

    true_taus = dataset["taus"].clone().detach().numpy()
    plt.title("MC Estimator: Taus Over Epochs")
    plt.plot(taus_trajectory[:,0],color="red",label="tau 1")
    plt.axhline(true_taus[0],color="red",linestyle="--",label="True tau 1")
    plt.plot(taus_trajectory[:,1],color="blue", label="tau 2")
    plt.axhline(true_taus[1],color="blue",linestyle="--",label="True tau 2")
    plt.xlabel("Epochs")
    plt.ylabel("Tau")
    plt.legend()
    plt.savefig(f"./midway_report_figs/{run_save_name}_taus.png")
    plt.cla()

    true_R_diag = dataset["R_diag"].clone().detach().numpy()
    plt.title("MC Estimator: R Over Epochs")
    plt.plot(R_diag_trajectory[:,0],color="red")
    plt.axhline(true_R_diag[0],color="red",linestyle="--")
    plt.plot(R_diag_trajectory[:,1],color="blue")
    plt.axhline(true_R_diag[1],color="blue",linestyle="--")
    plt.plot(R_diag_trajectory[:,2],color="green")
    plt.axhline(true_R_diag[2],color="green",linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Standard Deviation")
    plt.savefig(f"./midway_report_figs/{run_save_name}_R_diag.png")
    plt.cla()

def plot_dataset(dataset,first_n,save_name):
    time = dataset["times"]
    Z = dataset["Z"]
    for l in range(Z.shape[2]):
        for i in range(first_n):
            plt.plot(time, Z[i,:,l])
        plt.title(f"Trajectories From Latent {l}")
        plt.savefig(f'./datasets/{save_name}_{l}.png')
        plt.cla()
    

def graph_VAE(run_save_name):

    with open(f"./midway_models/dataset.pkl","rb") as f:
        dataset = pkl.load(f)
    with open(f"./midway_models/{run_save_name}_VAE_model.pkl","rb") as f:
        gpml = pkl.load(f)

    training_loss = np.array(gpml.epochs_total_loss)
    training_ll_loss = np.array(gpml.epochs_ll_loss)
    training_kl_loss = np.array(gpml.epochs_kl_loss)
    testing_loss = np.array(gpml.epochs_validation_total_loss)

    taus_trajectory = np.array(gpml.epochs_taus)
    R_diag_trajectory = np.array(gpml.epochs_R_diags)

    plt.title("VAE Estimator: Elbo Loss Over Epochs")
    plt.plot(training_loss,label="Train Loss",color="black")
    plt.plot(testing_loss,label="Test Loss",color="purple")
    plt.plot(training_ll_loss,label="Train LL Loss",color="red")
    plt.plot(training_kl_loss,label="Train KL Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"./midway_report_figs/{run_save_name}_vae_train_loss.png")
    plt.cla()

    true_taus = dataset["taus"].clone().detach().numpy()
    plt.title("VAE Estimator: Taus Over Epochs")
    plt.plot(taus_trajectory[:,0],color="red",label="tau 1")
    plt.axhline(true_taus[0],color="red",linestyle="--",label="True tau 1")
    plt.plot(taus_trajectory[:,1],color="blue", label="tau 2")
    plt.axhline(true_taus[1],color="blue",linestyle="--",label="True tau 2")
    plt.xlabel("Epochs")
    plt.ylabel("Tau")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"./midway_report_figs/{run_save_name}_vae_taus.png")
    plt.cla()

        
    true_R_diag = dataset["R_diag"]
    plt.title("VAE Estimator: R Over Epochs")
    plt.plot(R_diag_trajectory[:,0],color="red")
    plt.axhline(true_R_diag[0],color="red",linestyle="--")
    plt.plot(R_diag_trajectory[:,1],color="blue")
    plt.axhline(true_R_diag[1],color="blue",linestyle="--")
    plt.plot(R_diag_trajectory[:,2],color="green")
    plt.axhline(true_R_diag[2],color="green",linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Standard Deviation")
    plt.savefig(f"./midway_report_figs/{run_save_name}_vae_R_diag.png")
    plt.cla()

true_taus = [2.5,4.0]
true_R_diag = [0.1,0.1,0.1]
with torch.no_grad():
    train_dataset = two_latent_three_pop_dataset(taus=true_taus,R_diag=true_R_diag,time_divisions=12,samples=10,time_end=10.0)
    with open(f"./midway_models/train_dataset.pkl","wb") as f:
        pkl.dump(train_dataset,f)
    test_dataset = two_latent_three_pop_dataset(taus=true_taus,R_diag=true_R_diag,time_divisions=12,samples=2,time_end=10.0)
    with open(f"./midway_models/test_dataset.pkl","wb") as f:
        pkl.dump(test_dataset,f)

#two_latent_three_pop_Convex(train_dataset,test_dataset,epochs=10000)
#two_latent_three_pop_MC(train_dataset,test_dataset,epochs=10000,batch_size=10,samples=10000,run_save_name="second_latest",layers=[(20,nn.LeakyReLU()),(20,nn.LeakyReLU())])
#graph_MC("second_latest")
    
two_latent_three_pop_VAE(train_dataset,test_dataset,epochs=50000,samples = 1,
                         encoding_model_layers=[(100,nn.LeakyReLU()),(100,nn.LeakyReLU())],
                         decoding_model_layers=[(20,nn.LeakyReLU()),(20,nn.LeakyReLU())],batch_size=10)
#graph_VAE("latest_vae")
#plot_dataset(train_dataset,5,"train_data")