from gpfa.gpfa import GPFA
import torch
gpfa = GPFA("cpu",2,3,times=torch.linspace(0.0,1.0,4),initial_taus=[1.0,2.0])
y = torch.concat([torch.ones((3,3)),2*torch.ones((3,1))],dim=1)
gpfa.em_update(y)