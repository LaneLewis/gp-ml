import torch
from torch import nn
from torch.nn.modules import Module
import math
class LSTM_Encoder(nn.Module):

    def __init__(self,device,latent_dims,observed_dims,
                 data_to_init_hidden_state_size=10,hidden_state_to_posterior_size=10,slack_term=0.00001):
        super().__init__()
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.norm_layer = nn.BatchNorm1d(observed_dims)
        self.data_to_hidden_layer = nn.LSTM(observed_dims,data_to_init_hidden_state_size,batch_first=True)
        self.hidden_layer_to_initial = nn.Linear(data_to_init_hidden_state_size,hidden_state_to_posterior_size)
        self.initial_to_posterior_states = nn.LSTM(1,hidden_state_to_posterior_size,batch_first=True)
        self.to_mean = nn.Linear(hidden_state_to_posterior_size,self.latent_dims)
        self.slack = slack_term
        self.device = device

    def forward(self,X):
        #X has shape [batch_size,timesteps,observed_dims]
        batch_size = X.shape[0]
        timesteps = X.shape[1]
        observed_dims = X.shape[2]
        normed_X = self.norm_layer(X.permute(0,2,1))
        X = normed_X.permute(0,2,1)
        #should have shape [hidden_dim_1]
        hidden_states,(_,_) = self.data_to_hidden_layer(X)
        last_hidden_states = hidden_states[:,-1,:]
        dynamics_initial_cond = self.hidden_layer_to_initial(last_hidden_states)
        dummy_inputs = torch.zeros((batch_size,timesteps,1)).float()
        dummy_initial_cxs =  torch.ones(dynamics_initial_cond.shape).float()
        dynamics_hidden_states,_ = self.initial_to_posterior_states(dummy_inputs,
                                                                    (dynamics_initial_cond.reshape(1,batch_size,dynamics_initial_cond.shape[1])
                                                                    ,dummy_initial_cxs.reshape(1,batch_size,dummy_initial_cxs.shape[1])))
        means = self.to_mean(dynamics_hidden_states)
        return means
    
class FeedforwardPrediction(nn.Module):
    #NOTE this is completely untested!
    def __init__(self,layers_list,latent_dims,observed_dims,timesteps,device="cpu"):
        super().__init__()
        self.timesteps = timesteps
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.flattened_input_dims = observed_dims*timesteps
        self.flattened_output_dims = latent_dims*timesteps
        self.layers = nn.ModuleList()
        self.norm_layer = nn.BatchNorm1d(observed_dims)
        input_size = self.flattened_input_dims
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.layers.append(nn.Linear(input_size,self.flattened_output_dims))
        self.to(device)

    def forward(self,X):
        batch_X = X.permute(0,2,1)
        normed_batch = self.norm_layer(batch_X)
        X = normed_batch.permute(0,2,1)

        batch_size = X.shape[0]
        flattened_X = torch.flatten(X,1)
        input_data = flattened_X
        for layer in self.layers:
            input_data = layer(input_data)
        output_matrix = input_data.reshape(batch_size,self.timesteps,self.latent_dims)
        return output_matrix

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerPrediction(nn.Module):
    def __init__(self,latent_dims,observed_dims,device="cpu",heads=8,transformer_layers=2,embedding_dim=16):
        super().__init__()
        self.latent_dims = latent_dims
        self.observed_dims = observed_dims
        self.norm_layer = nn.BatchNorm1d(observed_dims)
        self.position_encoding = PositionalEncoding(embedding_dim,dropout=0.2)
        self.embedding_layer = nn.Linear(self.observed_dims,embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim,heads,batch_first=True,dim_feedforward=1000)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer,transformer_layers)
        self.decoder = nn.Linear(embedding_dim,self.latent_dims)
    def forward(self,X):
        #X has shape [batch_size, timesteps, observed_dims]
        batch_X = X.permute(0,2,1)
        normed_batch = self.norm_layer(batch_X)
        value = normed_batch.permute(0,2,1)
        value = self.embedding_layer(value)
        value = self.position_encoding(value)
        value = self.encoder_layers(value)
        Zs = self.decoder(value)
        return Zs


