import numpy as np
import torch
import scipy

class GridCells(object):

    def __init__(self, options):
        self.Ng = options.Ng
        # self.sigma_gc = options.gric_cell_rf
        self.box_width = options.box_width
        self.box_height = options.box_height
        # self.is_periodic = options.periodic
        self.device = options.device
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # Grid cell phase offsets
        self.phase_offsets = torch.rand(self.Ng) * 2 * np.pi
        
        max_freq = 0.5*max(options.box_height, options.box_height)
        min_freq = 0.08*min(options.box_height, options.box_height)
        
        self.frequency = torch.rand(self.Ng) * (max_freq - min_freq) + min_freq

        self.phase_offsets = self.phase_offsets.to(self.device)
    
    def get_activation(self, pos):
        '''
        Get grid cell activations for a given position.
    
        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].
    
        Returns:
            outputs: grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
    
        theta = torch.tensor(60, device=self.device).float().mul(np.pi / 180)  # Convert 60 degrees to radians
        batch_size, sequence_length, _ = pos.shape
    
        # Precompute wave vectors
        angles = (self.phase_offsets.unsqueeze(1) + torch.arange(3, device=self.device) * theta).view(-1)
        wave_vectors = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1) / self.frequency.repeat_interleave(3).unsqueeze(1)
    
        # Compute cosine activity
        pos_reshaped = pos.reshape(-1, 2)  # Flatten batch_size and sequence_length
        cos_activities = torch.cos(2 * np.pi * (pos_reshaped @ wave_vectors.T))
    
        # Reshape and sum activities for each orientation
        cos_activities = cos_activities.view(batch_size, sequence_length, self.Ng, 3).sum(dim=-1)
    
        # Apply softmax for normalization
        outputs = self.softmax(cos_activities)
    
        return outputs

    def grid_gc(self, gc_outputs, pos, res=32):
        ''' Interpolate grid cell outputs onto a grid'''
        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        gc_outputs = gc_outputs.reshape(-1, self.Ng)
        pos = pos.reshape(-1, 2)
        T = gc_outputs.shape[0] #T vs transpose? What is T? (dim's?)
        gc = np.zeros([T, res, res])
        
        
        for i in range(self.Ng):
            gridval = scipy.interpolate.griddata(pos, gc_outputs[:,i], grid)
            gc[i] = gridval.reshape([res, res])
        
        return gc

    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of grid cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T

        pos = torch.tensor(pos)
        
        # Put on GPU if available
        pos = pos.to(self.device)

        #Maybe specify dimensions here again?
        gc_outputs = self.get_activation(pos).reshape(-1,self.Ng).cpu()

        C = gc_outputs@gc_outputs.T
        Csquare = C.reshape(res,res,res,res)

        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)

        return Cmean