# Example of pc_info functions
import torch
from pc_info import I_sec, I_spike, I_spike_joint, I_sec_joint

pc1 = torch.zeros(4, 1, 4)
pc1[:,0,0] = torch.tensor([1, 0, 0, 0])
pc1[:,0,1] = torch.tensor([0, 1, 0, 0])
pc1[:,0,2] = torch.tensor([0, 0, 1, 0])
pc1[:,0,3] = torch.tensor([0, 0, 0, 1])

pc2 = torch.zeros(4, 1, 4)
pc2[:,0,0] = torch.tensor([1, 0, 0, 0])
pc2[:,0,1] = torch.tensor([0, 1, 0, 0])
pc2[:,0,2] = torch.tensor([0, 0, 0, 0])
pc2[:,0,3] = torch.tensor([0, 0, 0, 1])


pc3 = torch.zeros(4, 1, 4)
pc3[:,0,0] = torch.tensor([1, 1, 1, 0])
pc3[:,0,1] = torch.tensor([0, 1, 0, 0])
pc3[:,0,2] = torch.tensor([0, 0, 1, 0])
pc3[:,0,3] = torch.tensor([0, 0, 0, 1])


pc4 = torch.zeros(4, 1, 4)
pc4[:,0,0] = torch.tensor([1, 0, 0, 0])
pc4[:,0,1] = torch.tensor([0, 1, 1, 0])
pc4[:,0,2] = torch.tensor([0, 1, 1, 0])
pc4[:,0,3] = torch.tensor([0, 0, 0, 1])


for pc in [pc1, pc2, pc3, pc4]:
    dist = 1/pc.shape[0] * torch.ones(pc.shape[0])
    
    print("infosec:" ,I_sec(pc, dist))
    print("infospike:" ,I_spike(pc, dist))
    print('joint infosec:', I_sec_joint(pc, dist))
    print('joint infospike:', I_spike_joint(pc, dist))
    
    J = I_spike_joint(pc, dist)
    L, V = torch.linalg.eig(J.mean(dim = 0))
    
    idx = torch.argmax(torch.view_as_real(L)[:,0])
    
    print('Largest eigenvalue:', torch.view_as_real(L)[idx,0])
    print('Associated eigenvector:', V[:,idx])
    
    
    
