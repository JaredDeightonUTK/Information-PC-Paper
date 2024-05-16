from utils import generate_run_ID
from place_cells import PlaceCells as PC
from trajectory_generator import TrajectoryGenerator 
from model import RNN
from trainer import Trainer
from grid_cells import GridCells as GC
import numpy as np
import torch

# Training options and hyperparameters
class Options:
    pass

options = Options()
num_trials = 10

options.n_epochs = 5          # number of training epochs
options.n_steps = 20    # batches per epoch
options.batch_size = 40      # number of trajectories per batch
options.sequence_length = 100  # number of steps in trajectory
options.learning_rate = 1e-4  # gradient descent learning rate
options.Np = 64              # number of place cells
options.Ng = 1028             # number of grid cells
options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
options.RNN_type = 'RNN'      # RNN or LSTM
options.activation = 'relu'   # recurrent nonlinearity
options.weight_decay = 0  # strength of weight decay on recurrent weights
options.DoG = True            # use difference of gaussians tuning curves
options.periodic = False      # trajectories with periodic boundary conditions
options.box_width = 0.5      # width of training environment
options.box_height = 0.5      # height of training environment
options.device = 'cuda' if torch.cuda.is_available() else 'cpu'
options.dist = 1/options.sequence_length * torch.ones(options.sequence_length)
options.hardcoded_gcs = False
options.hardcoded_pcs = True
options.loss_fn  = 'I_loss'

for i in range(num_trials):
    options.save_dir = 'validation' + '_Np' + str(options.Np) + '/' + options.loss_fn + '/'
    options.run_ID = generate_run_ID(options) + '_' + str(i)

    grid_cells = GC(options)
    model = RNN(options, gc = grid_cells)
    
    model.to(options.device)
    
    place_cells = PC(options)
    
    trajectory_generator = TrajectoryGenerator(options, place_cells, model)
    
    trainer = Trainer(options, model, trajectory_generator)
    
    np.save(options.save_dir + options.run_ID  + '/options.npy', options) 
    torch.save(model.state_dict(), options.save_dir + options.run_ID +'/pre-trained_model.pth' )
    
    model.train()
    
    trainer.train(n_epochs = options.n_epochs, n_steps = options.n_steps)  
    
    print('Model trained!')
    
    if options.device == 'cuda':
        torch.save(model, 'cuda_ModelB.pt')
    
    train_loss = np.array(trainer.loss)
    test_loss = np.array(trainer.val_loss)
    
    np.save(options.save_dir + options.run_ID + '/train_loss.npy', train_loss)
    np.save(options.save_dir + options.run_ID + '/test_loss.npy', test_loss)
    
    torch.save(model.state_dict(), options.save_dir + options.run_ID +'/trained_model.pth' )
    


