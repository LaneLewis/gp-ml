import unittest
import torch
from utils.gaussian_process import sde_kernel_matrices
from utils.elbo_loss import kl_divergence

class TestGPML(unittest.TestCase):
    def test_kernel(self):
        times = torch.linspace(0.0,1.0,5)
        tau1 = 1.0
        tau2 = 1.0
        signal_sd1 = 1.0
        signal_sd2 = 1.0
        noise_sd1 = 0.5
        noise_sd2 = 0.5
        taus = torch.tensor([tau1,tau2])
        signal_sds = torch.tensor([signal_sd1,signal_sd2])
        noise_sds = torch.tensor([noise_sd1,noise_sd2])
        kernels = sde_kernel_matrices(times,taus,signal_sds,noise_sds)
        def kernel_indiv_entry(t_1,t_2,tau,signal_sd,noise_sd):
            '''signal_sd^2 * e^(-(t_1 - t_2)^2/(2tau^2)) + nosie_sd^2'''
            if not t_1 == t_2:
                noise_sd = 0.0
            return (signal_sd**2)*torch.exp(-1*((t_1 - t_2)**2)/(2*tau**2)) + (noise_sd)**2
        for kernel_index in range(kernels.shape[0]):
            for t_1_index in range(times.shape[0]):
                for t_2_index in range(times.shape[0]):
                    self.assertEqual(kernels[kernel_index,t_1_index,t_2_index],kernel_indiv_entry(times[t_1_index],times[t_2_index],
                                                                                                  taus[kernel_index],signal_sds[kernel_index],
                                                                                                  noise_sds[kernel_index]))

    def test_kl_loss(self):
        '''   
        two latents, three batches, 20 time steps
        '''
        timesteps = 20
        num_latents = 2
        batches = 3

        mu_1 = torch.concat([torch.ones((timesteps,1)),2*torch.ones((timesteps,1))],dim=1)
        mu_2 = torch.concat([0.5*torch.ones((timesteps,1)),3*torch.ones((timesteps,1))],dim=1)
        mu_3 = torch.concat([0.9*torch.ones((timesteps,1)),4*torch.ones((timesteps,1))],dim=1)

        cov_1 = torch.stack([0.5*torch.diag(torch.ones(timesteps)),1.5*torch.diag(torch.ones(timesteps))],dim=2)
        cov_2 = torch.stack([1.5*torch.diag(torch.ones(timesteps)),1.2*torch.diag(torch.ones(timesteps))],dim=2)
        cov_3 = torch.stack([1.1*torch.diag(torch.ones(timesteps)),0.5*torch.diag(torch.ones(timesteps))],dim=2)

        target_means = torch.zeros((timesteps,num_latents))
        target_sigmas = torch.stack([(i+1)*torch.diag(torch.ones(timesteps)) for i in range(num_latents)]).movedim(0,2)

        target_flat_mean = torch.zeros(timesteps*num_latents)
        target_flat_sigma = torch.block_diag(*[target_sigmas[:,:,i] for i in range(num_latents)])

        batch_0_flat_mu = mu_1.T.reshape(timesteps*num_latents)
        batch_0_flat_sigma = torch.block_diag(*[cov_1[:,:,sigmas_i] for sigmas_i in range(cov_1.shape[2])])

        batch_1_flat_mu = mu_2.T.reshape(timesteps*num_latents)
        batch_1_flat_sigma = torch.block_diag(*[cov_2[:,:,sigmas_i] for sigmas_i in range(cov_2.shape[2])])

        batch_2_flat_mu = mu_3.T.reshape(timesteps*num_latents)
        batch_2_flat_sigma = torch.block_diag(*[cov_3[:,:,sigmas_i] for sigmas_i in range(cov_3.shape[2])])

        target_dist = torch.distributions.MultivariateNormal(target_flat_mean,target_flat_sigma)

        batch_0_dist = torch.distributions.MultivariateNormal(batch_0_flat_mu,batch_0_flat_sigma)
        batch_1_dist = torch.distributions.MultivariateNormal(batch_1_flat_mu,batch_1_flat_sigma)
        batch_2_dist = torch.distributions.MultivariateNormal(batch_2_flat_mu,batch_2_flat_sigma)

        batch_0_kl = torch.distributions.kl.kl_divergence(batch_0_dist,target_dist)
        batch_1_kl = torch.distributions.kl.kl_divergence(batch_1_dist,target_dist)
        batch_2_kl = torch.distributions.kl.kl_divergence(batch_2_dist,target_dist)

        batched_mus = torch.stack([mu_1,mu_2,mu_3])
        batched_covs = torch.stack([cov_1,cov_2,cov_3])
        correct_kl_tensor = torch.stack([batch_0_kl,batch_1_kl,batch_2_kl])
        test_kl_tensor = kl_divergence(batched_mus,batched_covs,target_sigmas)

        self.assertTrue(torch.linalg.norm(correct_kl_tensor-test_kl_tensor) < 0.0001)

if __name__ == '__main__':
    unittest.main()