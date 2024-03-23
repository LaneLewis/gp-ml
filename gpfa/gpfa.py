from utils.gaussian_process import sde_kernel_matrices, sde_kernel_matrices_derivatives
import torch
class GPFA():
    "works for a single dataset only"
    def __init__(self,device:str,latent_dims:int,observation_dims:int,times:torch.Tensor,
                 initial_C=None,
                signal_sds:list[float]=None, noise_sds:list[float]=None,
                initial_taus:list[float]=None,initial_R_diag:list[float]=None,
                initial_d:list[float]=None):
        self.latent_dims = latent_dims
        self.observation_dims = observation_dims
        self.timesteps = times.shape[0]
        self.times = times
        self.device = device
        if initial_C == None:
            self.C = torch.rand((observation_dims,latent_dims),requires_grad=False,device=device)
        else:
            assert initial_C.shape[0] == observation_dims
            assert initial_C.shape[1] == latent_dims
            self.C = torch.tensor(initial_C,requires_grad=False,device=device)
    
        if initial_taus == None:
            self.taus = torch.rand((latent_dims),requires_grad=False,device=device)
        else:
            assert len(initial_taus) == latent_dims
            self.taus = torch.tensor(initial_taus,requires_grad=False,device=device)
        #initializes R
        if initial_R_diag== None:
            self.R_diag = torch.rand((self.observation_dims),requires_grad=False,device=device)
        else:
            assert len(initial_R_diag) == observation_dims
            self.R_diag = torch.tensor(initial_R_diag,requires_grad=False,device=device)
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

        if initial_d == None:
            self.d = torch.rand((self.observation_dims),requires_grad=False,device=device)
        else:
            assert len(initial_d) == observation_dims
            self.d = torch.tensor(initial_d,requires_grad=False,device=device)

    def make_K_bar(self,K_tensor):
        "slow but idk if that matters for our use"
        make_block = lambda i,j: torch.diag(K_tensor[i,j,:])
        i_list = []
        for i in range(self.timesteps):
            j_list = []
            for j in range(self.timesteps):
                j_list.append(make_block(i,j))
            block_row = torch.concat(j_list,dim=0)
            i_list.append(block_row)
        return torch.concat(i_list,dim=1)
    
    def make_R_bar(self):
        R = torch.diag(self.R_diag)
        return torch.block_diag(*self.timesteps*[R])
    def make_d_bar(self):
        return torch.concat(self.timesteps*[self.d])
    def make_C_bar(self):
        return torch.block_diag(*self.timesteps*[self.C])
    
    @staticmethod
    def make_y_bar(y):
        return torch.flatten(y.T)
    
    @staticmethod
    def x_bar_given_y_bar(C_bar,d_bar,R_bar,K_bar,y_bar):
        inner_term = torch.inverse(torch.linalg.multi_dot([C_bar,K_bar,C_bar.T]) + R_bar)
        mean_map = torch.linalg.multi_dot([K_bar,C_bar.T,inner_term])
        mean = torch.matmul(mean_map,y_bar-d_bar)
        cov = K_bar - torch.linalg.multi_dot([mean_map,C_bar,K_bar])
        return mean,cov
    
    def get_latent_posterior_moments(self,x_bar_given_y_bar_mean,x_bar_given_y_bar_cov):
        #gets the mean moment split over time
        time_first_moment = [x_bar_given_y_bar_mean[t*self.latent_dims:(t+1)*self.latent_dims] for t in range(self.timesteps)]
        #the total second moment
        second_moment = torch.outer(x_bar_given_y_bar_mean,x_bar_given_y_bar_mean) + x_bar_given_y_bar_cov
        get_t_block = lambda t : second_moment[t*self.latent_dims:(t+1)*self.latent_dims,t*self.latent_dims:(t+1)*self.latent_dims]
        get_l_block = lambda l : second_moment[l::self.latent_dims,l::self.latent_dims]
        time_second_moments = [get_t_block(t) for t in range(self.timesteps)]
        latent_second_moments = [get_l_block(l) for l in range(self.latent_dims)]
        return time_first_moment,time_second_moments,latent_second_moments
    
    def em_update_C_d(self,y,time_first_moments,time_second_moments):
        lh_sum = []
        rh_sum = []
        for t in range(self.timesteps):
            x_shifted = torch.concat([time_first_moments[t],torch.ones(1)])
            lh_sum.append(torch.outer(y[:,t],x_shifted))
            upper_block = torch.concat([time_second_moments[t],time_first_moments[t].unsqueeze(1)],dim=1)
            bottom_block = x_shifted.unsqueeze(0)
            rh_sum.append(torch.concat([upper_block,bottom_block],dim=0))
        C_d = torch.matmul(sum(lh_sum),torch.inverse(sum(rh_sum)))
        C_new = C_d[:,:self.latent_dims]
        d_new = C_d[:,self.latent_dims]
        return C_new,d_new
    
    def em_update_R(self,y,d_new,C_new,time_first_moments):
        lh_sum_term = sum([torch.outer(y[:,t]-d_new,y[:,t]-d_new) for t in range(self.timesteps)])
        rh_sum_term = sum([torch.outer(y[:,t]-d_new,time_first_moments[t]) for t in range(self.timesteps)])
        return torch.diag(lh_sum_term - torch.matmul(rh_sum_term,C_new.T))/self.timesteps
    
    def grad_step_taus(self,K_tensor,latent_second_moments):
        kernel_derivatives = sde_kernel_matrices_derivatives(self.times,self.taus,self.signal_sds)
        tau_derivatives = torch.zeros(self.latent_dims,device=self.device)
        for latent_index in range(self.latent_dims):
            K_inv = torch.inverse(K_tensor[:,:,latent_index])
            loss_K_derivative = -1/2*(-1*K_inv + torch.linalg.multi_dot([K_inv,latent_second_moments[latent_index],K_inv]))
            K_tau_derivative = kernel_derivatives[:,:,latent_index]
            loss_tau_derivative = torch.trace(torch.matmul(loss_K_derivative.T,K_tau_derivative))
            tau_derivatives[latent_index] = loss_tau_derivative
        return tau_derivatives
    
    def em_update(self,y):
        K_tensor = sde_kernel_matrices(self.times,self.taus,self.signal_sds,self.noise_sds)
        K_bar = self.make_K_bar(K_tensor)
        R_bar = self.make_R_bar()
        d_bar = self.make_d_bar()
        C_bar = self.make_C_bar()
        y_bar = self.make_y_bar(y)
        posterior_x_bar_mean,posterior_y_bar_cov = self.x_bar_given_y_bar(C_bar,d_bar,R_bar,K_bar,y_bar)
        time_first_moments,time_second_moments,latent_second_moments = self.get_latent_posterior_moments(posterior_x_bar_mean,posterior_y_bar_cov)
        C_new,d_new = self.em_update_C_d(y,time_first_moments,time_second_moments)
        R_diag_new = self.em_update_R(y,d_new,C_new,time_first_moments)
        tau_derivatives = self.grad_step_taus(K_tensor,latent_second_moments)
        