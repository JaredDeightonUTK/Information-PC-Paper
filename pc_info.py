# Containing functions for calculating PC spatial information 
import torch
import torch.nn as nn

relu = nn.ReLU()

eps = 1e-10 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def I_sec(pc, dist):
    ''' Calculate Spatial Information Rate (bits/sec)
    

    Parameters
    ----------
    pc : place cell activations with shape [sequence_length, batch_size, Np]
    dist : probablilities across trajectory of space [sequence_length]

    Returns
    -------
    info_per_cell : information (bits/sec) of each place cell across sequences of shape [batch_size, Np]
        
    '''
    
    pc = torch.relu(pc)
    pc_nonzero_indices = pc.nonzero(as_tuple = False)
    
    info_matrix = torch.zeros(pc.shape)
    
    pc_nonzero = pc[pc_nonzero_indices[:, 0], pc_nonzero_indices[:, 1], pc_nonzero_indices[:, 2]]
    
    dist_nonzero = dist[pc_nonzero_indices[:, 0]]
    spike_rate_per_cell = (pc * dist.view(-1, 1, 1)).sum(dim = 0)

    
    spike_rate_matrix = spike_rate_per_cell.unsqueeze(0).expand(pc.shape[0], -1, -1)
    
    
    l = spike_rate_matrix[pc_nonzero_indices[:, 0], pc_nonzero_indices[:, 1], pc_nonzero_indices[:, 2]]
    
    info = pc_nonzero * torch.log2(pc_nonzero / l + eps) * dist_nonzero
    
    info_matrix[pc_nonzero_indices[:, 0], pc_nonzero_indices[:, 1], pc_nonzero_indices[:, 2]] = info
    
    info_per_cell = info_matrix.sum(dim = 0)
    
    return info_per_cell


def I_spike(pc, dist):
    ''' Calculate Spatial Information Rate (bits/spike)
    
    
    Parameters
    ----------
    pc : place cell activations with shape [sequence_length, batch_size, Np]
    dist : probablilities across trajectory of space [sequence_length]
    
    Returns
    -------
    norm_info : information (bits/spike) of each place cell across sequences of shape [batch_size, Np]
        
    '''
    
    pc = pc.to(device)
    dist = dist.to(device)
   
    pc_nonzero_indices = pc.nonzero(as_tuple = False)

    info_matrix = torch.zeros(pc.shape).to(device)
    norm_info = torch.zeros(pc.shape).to(device)

    pc_nonzero = pc[pc_nonzero_indices[:, 0], pc_nonzero_indices[:, 1], pc_nonzero_indices[:, 2]]

    dist_nonzero = dist[pc_nonzero_indices[:, 0]]

    spike_rate_per_cell = (pc * dist.view(-1, 1, 1)).sum(dim = 0)
    
    spike_rate_matrix = spike_rate_per_cell.unsqueeze(0).expand(pc.shape[0], -1, -1)
    
    l = spike_rate_matrix[pc_nonzero_indices[:, 0], pc_nonzero_indices[:, 1], pc_nonzero_indices[:, 2]]
    
    info = pc_nonzero * torch.log2(pc_nonzero / l + eps) * dist_nonzero
    info_matrix[pc_nonzero_indices[:, 0], pc_nonzero_indices[:, 1], pc_nonzero_indices[:, 2]] = info

    info_per_cell = info_matrix.sum(dim = 0)
    safe_denominator = torch.where(spike_rate_per_cell.abs() > eps, spike_rate_per_cell, torch.ones_like(spike_rate_per_cell) * eps)
    norm_info = info_per_cell / safe_denominator
    
    return norm_info


def I_sec_joint(pc, dist): 
    ''' Calculate Spatial Information Rate of joint distribution defined by neuron pairs (bits/sec)
    
    Parameters
    ----------
    pc : place cell activations with shape [sequence_length, batch_size, Np]
    dist : probablilities across trajectory of space [sequence_length]
    
    Returns
    -------
    J : Information (bits/s) of each joint place cell pair, of shape [batch_size, Np, Np]. J[b, i, j] is joint info between pc_i and pc_j in batch b
        
    '''
    
    Nx, batch_size, Np = pc.shape
    
    pc1 = pc.unsqueeze(-1).expand(-1, -1, Np, Np)
    pc2 = pc.unsqueeze(2).expand(-1, -1, Np, Np)
    
    std_dev = torch.std(pc, dim=0)  # [batch_size, Np]
    std_dev_mask = std_dev == 0  # Boolean mask [batch_size, Np]
    
    
    pc_noise = pc[:,std_dev_mask] + torch.abs(1e-5 *torch.randn_like(pc[:, std_dev_mask]) + eps) 
    pc_new = pc.clone()
    pc_new[:, std_dev_mask] = pc_noise
    
    r_matrix_batch = torch.stack([torch.corrcoef(pc_new[:, obs, :].T) for obs in range(batch_size)])
    
    lab = pc1 * pc2
    
    lab_tilde = (dist.view(Nx, 1, 1, 1) * torch.sqrt(lab + eps)).sum(dim=0)
    la = (dist.view(Nx, 1, 1, 1) * pc1).sum(dim=0)
    lb = (dist.view(Nx, 1, 1, 1) * pc2).sum(dim=0)
    
    r_expanded = r_matrix_batch.unsqueeze(0).repeat(Nx, 1, 1, 1)
   
    log_argument1 = torch.sqrt(lab + eps) / (lab_tilde) 

    info_1 = r_expanded * dist.view(-1, 1, 1, 1) * torch.sqrt(lab + eps) * torch.where(log_argument1 > 0, torch.log2(log_argument1 + eps), torch.tensor(0.)) 
    
    numerator2 = pc1 - r_expanded * torch.sqrt(lab + eps)
    denominator2 = la - r_expanded * lab_tilde + eps
    log_argument2 = (numerator2) / (denominator2 ) ## see above
    info_2 = dist.view(-1, 1, 1, 1) * numerator2 * torch.where(log_argument2 > 0, torch.log2(log_argument2 + eps), torch.tensor(0.))
    
    numerator3 = pc2 - r_expanded * torch.sqrt(lab + eps)
    denominator3 = lb - r_expanded * lab_tilde + eps
    log_argument3 = (numerator3) / (denominator3 ) ## see above
    info_3 = dist.view(-1, 1, 1, 1) * numerator3 * torch.where(log_argument3 > 0, torch.log2(log_argument3 + eps), torch.tensor(0.))
    
    J = info_1.sum(dim=0) + info_2.sum(dim=0) + info_3.sum(dim=0)

    return J 

def I_spike_joint(pc, dist): 
    ''' Calculate Spatial Information Rate of joint distribution defined by neuron pairs (bits/spike)
    
    Parameters
    ----------
    pc : place cell activations with shape [sequence_length, batch_size, Np]
    dist : probablilities across trajectory of space [sequence_length]
    
    Returns
    -------
    J : Information (bits/s) of each joint place cell pair, of shape [batch_size, Np, Np]. J[b, i, j] is joint info between pc_i and pc_j in batch b
        
    '''
    
    pc = pc.to(device)
    dist = dist.to(device)
    
    Nx, batch_size, Np = pc.shape
    
    pc1 = pc.unsqueeze(-1).expand(-1, -1, Np, Np)
    pc2 = pc.unsqueeze(2).expand(-1, -1, Np, Np)
    
    std_dev = torch.std(pc, dim=0)  # [batch_size, Np]
    std_dev_mask = std_dev == 0  # Boolean mask [batch_size, Np]
    
    pc_noise = pc[:,std_dev_mask] + torch.abs(1e-5 *torch.randn_like(pc[:, std_dev_mask]) + eps) 
    pc_new = pc.clone()
    pc_new[:, std_dev_mask] = pc_noise
    
    r_matrix_batch = torch.stack([torch.corrcoef(pc_new[:, obs, :].T) for obs in range(batch_size)])
    
    lab = pc1 * pc2
    
    lab_tilde = (dist.view(Nx, 1, 1, 1) * torch.sqrt(lab + eps)).sum(dim=0)
    la = (dist.view(Nx, 1, 1, 1) * pc1).sum(dim=0)
    lb = (dist.view(Nx, 1, 1, 1) * pc2).sum(dim=0)
    
    alpha = (la + lb) / 2
    
    r_expanded = r_matrix_batch.unsqueeze(0).repeat(Nx, 1, 1, 1)
   
    log_argument1 = torch.sqrt(lab + eps) / (lab_tilde) 

    info_1 = r_expanded * dist.view(-1, 1, 1, 1) * torch.sqrt(lab + eps) * torch.where(log_argument1 > 0, torch.log2(log_argument1 + eps), torch.tensor(0.))
    
    numerator2 = pc1 - r_expanded * torch.sqrt(lab + eps)
    denominator2 = la - r_expanded * lab_tilde + eps
    log_argument2 = (numerator2) / (denominator2) 
    info_2 = dist.view(-1, 1, 1, 1) * numerator2 * torch.where(log_argument2 > 0, torch.log2(log_argument2 + eps), torch.tensor(0.))
    
    numerator3 = pc2 - r_expanded * torch.sqrt(lab + eps)
    denominator3 = lb - r_expanded * lab_tilde + eps
    log_argument3 = (numerator3) / (denominator3) 
    info_3 = dist.view(-1, 1, 1, 1) * numerator3 * torch.where(log_argument3 > 0, torch.log2(log_argument3 + eps), torch.tensor(0.))
    
    J = info_1.sum(dim=0) + info_2.sum(dim=0) + info_3.sum(dim=0)
    
    J[alpha != 0] = 1/alpha[alpha != 0] * J[alpha !=0]

    return J 



