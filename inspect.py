from trajectory_generator import TrajectoryGenerator 
from visualize import compute_ratemaps_pc
from place_cells import PlaceCells as PC
import numpy as np
import cv2
from model import RNN
import torch
from grid_cells import GridCells as GC
from visualize import plot_images

class Options:
    pass
options = Options()

# inspect a model


path = 'models/2024-05-16_Eigen_I_loss_Np_16_Ng_128_steps_100_batch_40_size_05'

model_path = path + '/most_recent_model.pth'
pre_trained_model_path = path + '/pre-trained_model.pth'
options = np.load(path + '/options.npy', allow_pickle='TRUE').item()
train_loss = np.load(path + '/train_loss.npy')
test_loss = np.load(path + '/test_loss.npy')
options.device = 'cpu'

place_cells = PC(options)
grid_cells = GC(options)

pre_trained_model  = RNN(options, gc = grid_cells)
pre_trained_model.load_state_dict(torch.load(pre_trained_model_path, map_location=torch.device('cpu')))
pre_trained_model.eval()

model  = RNN(options, gc = grid_cells)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') ))
model.eval()


trajectory_generator = TrajectoryGenerator(options, place_cells, model)
trajectory_generator_pretrained = TrajectoryGenerator(options, place_cells, pre_trained_model)

res  = 25

n_avg = 100
Np = options.Np
activations_og, rate_map_og, p_og, pos_og, pcscores_og = compute_ratemaps_pc(pre_trained_model, trajectory_generator_pretrained, options, res = res, n_avg = n_avg, Np = Np)
activations, rate_map, p, pos, pcscores = compute_ratemaps_pc(model, trajectory_generator, options, res = res, n_avg = n_avg, Np = Np)

smooth_activations = np.zeros_like(activations)

for i in range(activations.shape[0]):
    smooth_activations[i] = cv2.GaussianBlur(activations[i], (3, 3), sigmaX=1, sigmaY=0)

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
    smooth_activations_og[i] = cv2.GaussianBlur(activations_og[i], (3, 3), sigmaX=1, sigmaY=0)

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
    
    
plot_images(activations_og, scores = pcscores_og, num_images = options.Np, save = True, savename = path + '/pre_trained_ratemap.png')
plot_images(activations, scores = pcscores, num_images = options.Np, save  = True, savename = path + '/ratemap.png')
    
np.save(path + '/p.npy', p)
np.save(path + '/smooth_p.npy', smooth_p)
np.save(path + '/activations.npy', activations)
np.save(path + '/smooth_activations.npy', smooth_activations)
np.save(path + '/pcsores.npy', pcscores)
np.save(path + '/pos.npy', pos)


np.save(path + '/p_og.npy', p_og)
np.save(path + '/smooth_p_og.npy', smooth_p_og)
np.save(path+ '/activations_og.npy', activations_og)
np.save(path + '/smooth_activations_og.npy', smooth_activations_og)
np.save(path + '/pcsores_og.npy', pcscores_og)
np.save(path + '/pos_og.npy', pos_og)

