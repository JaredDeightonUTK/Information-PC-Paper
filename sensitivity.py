# Sensitivity analysis
from utils import generate_run_ID
from place_cells import PlaceCells as PC
from trajectory_generator import TrajectoryGenerator 
from model import RNN
from trainer import Trainer
from grid_cells import GridCells as GC
import numpy as np
import torch
from visualize import compute_ratemaps_pc
import cv2

class Options:
    pass

torch.manual_seed(14)

print('Cuda:', torch.cuda.is_available())

def run(Np, arena_size, learning_rate, sequence_length, batch_size, num_hidden, loss, num_trials):
    options = Options()

    options.n_epochs = 1          # number of training epochs
    options.n_steps = 50    # batches per epoch
    options.batch_size = batch_size      # number of trajectories per batch
    options.sequence_length = sequence_length  # number of steps in trajectory
    options.learning_rate = learning_rate  # gradient descent learning rate
    options.Np = Np            # number of place cells
    options.Ng = num_hidden             # number of grid cells
    options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
    options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
    options.RNN_type = 'RNN'      # RNN or LSTM
    options.activation = 'relu'   # recurrent nonlinearity
    options.weight_decay = 0  # strength of weight decay on recurrent weights
    options.DoG = True            # use difference of gaussians tuning curves
    options.periodic = False      # trajectories with periodic boundary conditions
    options.box_width = arena_size      # width of training environment
    options.box_height = arena_size      # height of training environment
    options.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    options.dist = 1/options.sequence_length * torch.ones(options.sequence_length)
    options.hardcoded_gcs = False
    options.hardcoded_pcs = True
    options.loss_fn  = loss

    for i in range(num_trials):
        options.save_dir = 'sensitivity_pt2' + '_Np' + str(options.Np) + '/' + options.loss_fn + '/'
        options.run_ID = generate_run_ID(options) + '_' + 'lr_' + str(learning_rate) + '_' + str(i)
        
        folder_name = options.save_dir + options.run_ID
        
        grid_cells = GC(options)
        model = RNN(options, gc = grid_cells)
        pre_trained_model = RNN(options, gc = grid_cells)
        
        model.to(options.device)
        pre_trained_model.to(options.device)
        
        place_cells = PC(options)
        
        trajectory_generator = TrajectoryGenerator(options, place_cells, model)
        trajectory_generator_pretrained = TrajectoryGenerator(options, place_cells, pre_trained_model)
        
        trainer = Trainer(options, model, trajectory_generator, restore = False)
        
        np.save(folder_name  + '/options.npy', options) 
        torch.save(model.state_dict(), folder_name +'/pre-trained_model.pth' )
        
        model.train()
        
        ## NOTICE STOPPING CRITERION ADDITION
        
        trainer.train(n_epochs = options.n_epochs, n_steps = options.n_steps, stopping_criterion = True, save = False)
        
        print('Model trained!')
        
        
        train_loss = np.array(trainer.loss)
        test_loss = np.array(trainer.val_loss)
        
        np.save(folder_name + '/train_loss.npy', train_loss)
        np.save(folder_name + '/test_loss.npy', test_loss)
        
        torch.save(model.state_dict(), folder_name +'/trained_model.pth' )
        
        
        trajectory_generator = TrajectoryGenerator(options, place_cells, model)
        trajectory_generator_pretrained = TrajectoryGenerator(options, place_cells, pre_trained_model)
        
        
        res  = 25

        n_avg = 100
        Np = options.Np
        activations_og, rate_map_og, p_og, pos_og, pcscores_og = compute_ratemaps_pc(pre_trained_model, trajectory_generator_pretrained, options, res = res, n_avg = n_avg, Np = Np)
        activations, rate_map, p, pos, pcscores = compute_ratemaps_pc(model, trajectory_generator, options, res = res, n_avg = n_avg, Np = Np)
        
        smooth_activations = np.zeros_like(activations)

        for i in range(activations.shape[0]):
            im = activations[i]
            smooth_activations[i] = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
            
            
        smooth_p = np.zeros_like(p)

        for i in range(pos.shape[0]):
            x = (pos[i, 0] + options.box_width/2) / (options.box_width) * res
            y = (pos[i, 1] + options.box_height/2) / (options.box_height) * res
            
            if x < 0:
                x = 0
            if x >= res:
                x = res - 1
                
            if y < 0:
                y = 0
            if y >= res:
                y = res - 1
            
            smooth_p[i,:] = smooth_activations[:, int(x), int(y)]
        
        smooth_activations_og = np.zeros_like(activations_og)

        for i in range(activations.shape[0]):
            im = activations_og[i]
            smooth_activations_og[i] = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)

        smooth_p_og = np.zeros_like(p_og)

        for i in range(pos_og.shape[0]):
            x = (pos_og[i, 0] + options.box_width/2) / (options.box_width) * res
            y = (pos_og[i, 1] + options.box_height/2) / (options.box_height) * res
            
            if x < 0:
                x = 0
            if x >= res:
                x = res - 1
                
            if y < 0:
                y = 0
            if y >= res:
                y = res - 1
            
            smooth_p_og[i,:] = smooth_activations_og[:, int(x), int(y)]
            
            
        np.save(folder_name +'/p.npy', p)
        np.save(folder_name + '/smooth_p.npy', smooth_p)
        np.save(folder_name + '/activations.npy', activations)
        np.save(folder_name + '/smooth_activations.npy', smooth_activations)
        np.save(folder_name + '/pcsores.npy', pcscores)
        np.save(folder_name + '/pos.npy', pos)

    
        
        np.save(folder_name + '/p_og.npy', p_og)
        np.save(folder_name + '/smooth_p_og.npy', smooth_p_og)
        np.save(folder_name + '/activations_og.npy', activations_og)
        np.save(folder_name + '/smooth_activations_og.npy', smooth_activations_og)
        np.save(folder_name + '/pcsores_og.npy', pcscores_og)
        np.save(folder_name + '/pos_og.npy', pos_og)
        

# =============================================================================
# num_trained = 0
# Np = 32
# for arena_size in [0.25, 0.75, 1, 1.5]:
#     for learning_rate in [1e-3, 1e-4, 1e-5]:
#         for sequence_length in [40, 80, 120]:
#             for batch_size in [10, 40, 80]:
#                 for num_hidden in [512, 1028, 2048]: #[256, 512, 1028] with Np=32:
#                     for loss in ['Skaggs_loss', 'Eigen_I_loss']:
#                         print('Number of models trained: ', num_trained, '/', 1944)
#                         run(Np, arena_size, learning_rate, sequence_length, batch_size, num_hidden, loss, num_trials = 3)
#                         num_trained += 3
# =============================================================================
                        
num_trained = 0
for Np in [32,64]:
    for arena_size in [0.25, 0.75, 1, 1.5, 3]:
        for learning_rate in [1e-3, 1e-4, 1e-5]:
            for sequence_length in [40, 80, 120, 200]:
                for batch_size in [10, 40, 80]:
                    for num_hidden in [256, 512, 1028, 2048]: #[256, 512, 1028] with Np=32:
                        for loss in ['Skaggs_loss', 'Eigen_I_loss']:
                            print('Parameters:', Np, arena_size, learning_rate, sequence_length, batch_size, num_hidden, loss)
                            print('Number of models trained: ', num_trained, '/', 8640)
                            run(Np, arena_size, learning_rate, sequence_length, batch_size, num_hidden, loss, num_trials = 3)
                            num_trained += 3
                            