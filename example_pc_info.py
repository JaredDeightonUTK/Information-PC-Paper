# Example of pc_info functions
import torch
from pc_info import I_sec, I_spike, I_spike_joint, I_sec_joint


# Consider four scenarios with four place cells each. Spatial trajectory of length four. 

# Assume each spatial index is equally likely

dist = 1/4 * torch.ones(4)

pc_scenario_1 = torch.zeros(4, 1, 4)
pc_scenario_1[:,0,0] = torch.tensor([1, 0, 0, 0]) # Neuron fires at 1 spike/s in quadrant A
pc_scenario_1[:,0,1] = torch.tensor([0, 1, 0, 0]) # Neuron fires at 1 spike/s in quadrant B
pc_scenario_1[:,0,2] = torch.tensor([0, 0, 1, 0]) # Neuron fires at 1 spike/s in quadrant C
pc_scenario_1[:,0,3] = torch.tensor([0, 0, 0, 1]) # Neuron fires at 1 spike/s in quadrant D

pc_scenario_2 = torch.zeros(4, 1, 4)
pc_scenario_2[:,0,0] = torch.tensor([1, 0, 0, 0]) # Neuron fires at 1 spike/s in quadrant A
pc_scenario_2[:,0,1] = torch.tensor([0, 1, 0, 0]) # Neuron fires at 1 spike/s in quadrant B
pc_scenario_2[:,0,2] = torch.tensor([0, 0, 0, 0]) # Neuron does not fire
pc_scenario_2[:,0,3] = torch.tensor([0, 0, 0, 1]) # Neuron fires at 1 spike/s in quadrant D


pc_scenario_3 = torch.zeros(4, 1, 4)
pc_scenario_3[:,0,0] = torch.tensor([1, 1, 1, 0]) # Neuron fires at 1 spike/s in quadrant A, B, and C
pc_scenario_3[:,0,1] = torch.tensor([0, 1, 0, 0]) # Neuron fires at 1 spike/s in quadrant B
pc_scenario_3[:,0,2] = torch.tensor([0, 0, 1, 0]) # Neuron fires at 1 spike/s in quadrant C
pc_scenario_3[:,0,3] = torch.tensor([0, 0, 0, 1]) # Neuron fires at 1 spike/s in quadrant D


pc_scenario_4 = torch.zeros(4, 1, 4)
pc_scenario_4[:,0,0] = torch.tensor([1, 0, 0, 0]) # Neuron fires at 1 spike/s in quadrant A
pc_scenario_4[:,0,1] = torch.tensor([0, 1, 1, 0]) # Neuron fires at 1 spike/s in quadrant B and C
pc_scenario_4[:,0,2] = torch.tensor([0, 1, 1, 0]) # Neuron fires at 1 spike/s in quadrant B and C
pc_scenario_4[:,0,3] = torch.tensor([0, 0, 0, 1]) # Neuron fires at 1 spike/s in quadrant D


for (i, pc_scenario) in enumerate([pc_scenario_1, pc_scenario_2, pc_scenario_3, pc_scenario_4]):
    
    print('PLACE CELL SCENARIO', i + 1)
    
    J = I_spike_joint(pc_scenario, dist)
    L, V = torch.linalg.eig(J.mean(dim = 0))
    
    idx = torch.argmax(torch.view_as_real(L)[:,0])
    
    print("infosec:" , I_sec(pc_scenario, dist), '\n')
    print("infospike:" ,I_spike(pc_scenario, dist), '\n')
    print('joint infosec:', I_sec_joint(pc_scenario, dist), '\n')
    print('joint infospike:', I_spike_joint(pc_scenario, dist), '\n')
    print('Largest eigenvalue:', torch.view_as_real(L)[idx,0], '\n')
    print('Associated eigenvector:', V[:,idx], '\n')
    
    
    
