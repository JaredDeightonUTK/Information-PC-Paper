# Main script to train (and for now, also evlauate) RNN to do unsupervised place field learning

import torch.cuda

from utils import generate_run_ID
from trajectory_generator import TrajectoryGenerator 
from model import RNN
from trainer import Trainer
from place_cells import PlaceCells as PC
from grid_cells import GridCells as GC


import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='models/',
                    help='directory to save trained models')

parser.add_argument('--n_epochs',
                    default = 5, 
                    help='number of training epochs')

parser.add_argument('--n_steps',
                    default = 20,
                    help='batches per epoch')

parser.add_argument('--batch_size',
                    default = 40,
                    help='number of trajectories per batch')

parser.add_argument('--sequence_length',
                    default = 100, 
                    help='number of steps in trajectory')

parser.add_argument('--learning_rate',
                    default = 1e-4, 
                    help='gradient descent learning rate')

parser.add_argument('--Np',
                    default = 16, 
                    help='number of place cells')

parser.add_argument('--Ng',
                    default = 128, 
                    help='number of grid cells')

parser.add_argument('--place_cell_rf',
                    default = 0.12,
                    help='width of place cell center tuning curve (m)')
parser.add_argument('--surround_scale',
                    default = 2,
                    help='if DoG, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--DoG',
                    default = True, 
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default = False,
                    help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    default= 0.5,
                    help='width of training environment')
parser.add_argument('--box_height',
                    default= 0.5, 
                    help='height of training environment')

parser.add_argument('--RNN_type',
                    default='RNN',
                    help='RNN or LSTM')

parser.add_argument('--activation',
                    default='relu',
                    help='recurrent nonlinearity')

parser.add_argument('--weight_decay',
                    default= 0,
                    help='strength of weight decay on recurrent weights')

parser.add_argument('--hardcoded_pcs',
                    default= True,
                    help='boolean to use hardcoded place cells for initial position embedding')

parser.add_argument('--hardcoded_gcs',
                    default= False,
                    help='boolean to use hardcoded grid cells for hidden layer activations')

parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training')

parser.add_argument('--loss_fn',
                    default= 'Eigen_I_loss',
                    help='loss function to use during training')

options = parser.parse_args()
options.run_ID = generate_run_ID(options)

# Uniform distribtuion across trajectory (that is, across space too)
options.dist = 1/options.sequence_length * torch.ones(options.sequence_length)

print(f'Using device: {options.device}')

grid_cells = GC(options)
place_cells = PC(options)

if options.RNN_type == 'RNN':
    model = RNN(options, gc = grid_cells)
elif options.RNN_type == 'LSTM':
    # model = LSTM(options, place_cells)
    raise NotImplementedError

# Put model on GPU if using GPU
model = model.to(options.device)

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


