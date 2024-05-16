import torch
import torch.nn as nn
import pc_info
from scipy.interpolate import griddata

# Define RNN model to be used in main

class RNN(torch.nn.Module):
    def __init__(self, options, gc = None, perfect_pcs = None):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.device = options.device
        self.dist = options.dist
        self.hardcoded_gcs = options.hardcoded_gcs
        self.hardcoded_pcs = options.hardcoded_pcs
        self.perfect_pcs = perfect_pcs
        self.gc = gc
        self.loss_fn = options.loss_fn
        
        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.Ng,
                                nonlinearity=options.activation,
                                batch_first=False,
                                bias=False)
        
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim  = 0)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(0.1) 
        
        self.visited_positions = (options.box_height - options.box_height) *torch.rand(size = (100, 100, 2)) + options.box_height 
        self.place_cell_outputs = torch.rand(size = (100, 100, options.Np))


    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size,  2].

        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        
        v, p0, init_pos = inputs  
        
        if self.hardcoded_gcs:
            # Calculate the cumulative sum of velocities and add to the initial position
            cumulative_velocities = torch.cumsum(v, dim=0)
            visited_positions = init_pos.squeeze(1) + cumulative_velocities
            # Compute grid cell activations
            g = self.gc.get_activation(visited_positions)
            
        else:
            
            init_state = self.encoder(p0)[None]
            init_state = init_state.to(self.device)
            
            g,_ = self.RNN(v, init_state)
           
            g = self.dropout(g)
            
            g = self.relu(g)
       
        return g
    

    def predict(self, inputs, init_pos = None):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [sequence_length, batch_size, Np].
        '''
        
        v, p0, init_pos = inputs   # v is sequence_length x batch_size x 2
        # init_pos is batch_size x 1 x 2
        
        if self.perfect_pcs is not None:
            # Calculate the cumulative sum of velocities and add to the initial position
            cumulative_velocities = torch.cumsum(v, dim=0)
            visited_positions = init_pos.squeeze(1) + cumulative_velocities
            # Compute grid cell activations
            p = self.perfect_pcs.get_activation(visited_positions)
            
        else:
        
            place_preds = self.decoder(self.g(inputs))
            
            p = self.relu(place_preds)
            p = p.to(self.device)
            
            
        return p
    
    def get_activation(self, locations):
        '''
        Get place cell activations at locations
        ----------
        locations : xy coordinates of size [num_locations, 2]
        Returns
        -------
        P : place cell output of size of size [num_locations, Np]

        '''
        
        # Interpolate PC activations from self.visited_positions, self.place_cell_outputs, then get activations at locations
        P = torch.zeros(locations.shape[0], self.Np)
        
        known_P = self.place_cell_outputs.reshape(-1, self.Np).detach().numpy()  
        known_pos = self.visited_positions.reshape(-1, 2).detach().numpy()  
        locations_np = locations.squeeze(1).detach().numpy() 
        
        for i in range(self.Np):
            # Interpolate the i-th place cell activation
            P[:, i] = torch.tensor(griddata(known_pos, known_P[:, i], locations_np, method='nearest'))
        
        return P
    

    
    def I_loss(self, inputs):
        '''

        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].

        Returns: 
            Joint information loss: Avg. joint information loss for this batch
        '''
        
        preds = self.predict(inputs) # sequence_length x batch_size x Np
        Nx, batch_size, Np = preds.shape
        
        eps = torch.finfo(torch.float32).eps
        
        dist = self.dist
        
        J = pc_info.I_spike_joint(preds, dist) 
        
        # J = J * (1 - torch.eye(Np, Np)) # Remove diagonal entries
        
        I_loss = -1/2*torch.mean(J.mean(dim = 0) + eps)
        
        loss =  I_loss 
        
        return loss 

    
    def I_loss_sum(self, inputs):
        '''

        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].

        Returns: 
            Sum of information loss: Avg. joint information loss for this batch
        '''
        
        preds = self.predict(inputs) # sequence_length x batch_size x Np
        Nx, batch_size, Np = preds.shape
        
        eps = torch.finfo(torch.float32).eps
        
        dist = self.dist
        
        J = pc_info.I_spike(preds, dist) 
        
        I_loss = -torch.sum(J + eps)
        
        return I_loss 
    
    def Eigen_I_loss(self, inputs):
        '''

        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].

        Returns: 
            Loss via maximal eigenvalue of J
        '''
        preds = self.predict(inputs)
        
        Nx, batch_size, Np = preds.shape
        
        dist = self.dist.to(self.device)
        
        preds = preds.to(self.device)
        
        J = pc_info.I_spike_joint(preds, dist) 
        
        J = J.mean(dim = 0).to(self.device)
        
        sorted, indices = abs(torch.view_as_real(torch.linalg.eigvals(J))[:,0]).sort(descending=True)
        # sorted, indices = torch.view_as_real(torch.linalg.eigvals(J))[:,0].sort(descending=True)
        
        loss = -1*sorted[0]
        
        return loss
    
    def I_sec_loss(self, inputs):
        '''

        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].

        Returns: 
            Joint information loss: Avg. joint information loss for this batch
        '''
        
        preds = self.predict(inputs) # sequence_length x batch_size x Np
        Nx, batch_size, Np = preds.shape
        
        eps = torch.finfo(torch.float32).eps
        
        dist = self.dist
        
        J = pc_info.I_sec_joint(preds, dist) 
        
        I_loss = -1/2*torch.mean(J.mean(dim = 0) + eps)
        
        loss =  I_loss 
        
        return loss 
    
    def Eigen_I_sec_loss(self, inputs):
        '''

        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].

        Returns: 
            Loss via maximal eigenvalue of J
        '''
        preds = self.predict(inputs)
        
        Nx, batch_size, Np = preds.shape
        
        dist = self.dist
        
        J = pc_info.I_sec_joint(preds, dist)
        
        J = J.mean(dim = 0).to(self.device)
        
        sorted, indices = abs(torch.view_as_real(torch.linalg.eigvals(J))[:,0]).sort(descending=True)
        
        loss = -1*sorted[0]
        
        return loss


    def compute_loss(self, inputs):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. MI_loss for this training batch.
            
        '''
        
        if self.loss_fn == 'Eigen_I_loss':
            loss = self.Eigen_I_loss(inputs)
        elif self.loss_fn == 'Skaggs_loss':
            loss = self.I_loss_sum(inputs)
        elif self.loss_fn == 'I_loss':
            loss = self.I_loss(inputs)
        else:
            print('loss function undetermined')
        
        loss = loss.to(self.device)
        
        # Weight regularization 
        factor = (self.RNN.weight_hh_l0**2).sum()
        factor = factor.to(self.device)

        loss += self.weight_decay * factor
        
        return loss
    
    

    
    