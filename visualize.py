
import numpy as np
import torch
from matplotlib import pyplot as plt
import placescores
# import placescores_torch
from imageio import imsave
import cv2

def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True, regularize = True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    if regularize:
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
        
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth = True, regularize = True, width=16):
    images = [rgb(im, cmap, smooth, regularize) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, Ng=12, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch = trajectory_generator.get_test_batch()
        g_batch = model.g(inputs).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

# =============================================================================
#     for x in range(res):
#         for y in range(res):
#             if counts[x, y] > 0:
#                 activations[:, x, y] /= counts[x, y]
# =============================================================================

    activations = np.nan_to_num(activations/counts)
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)
    

    return activations, rate_map, g, pos

def compute_ratemaps_pc(model, trajectory_generator, options, res=20, n_avg=None, Np=12, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Np)
    idxs = idxs[:Np]

    p = np.zeros([n_avg, options.batch_size * options.sequence_length, Np])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Np, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch = trajectory_generator.get_test_batch()
        p_batch = model.predict(inputs).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        p_batch = p_batch[:,:,idxs].reshape(-1, Np)
        
        p[index] = p_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res
        
        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += p_batch[i, :]

    activations = np.nan_to_num(activations/counts)
                
    p = p.reshape([-1, Np])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Np, -1)
    
    pcscores = []
    for i in range(Np):
        if (np.max(activations[i,:,:]) - np.min(activations[i,:,:])) > 0:
            im = (activations[i,:,:] - np.min(activations[i,:,:])) / (np.max(activations[i,:,:]) - np.min(activations[i,:,:]))
            pcscores.append(placescores.pc_score(im, 0.1))
    
    return activations, rate_map, p, pos, pcscores


# =============================================================================
# def compute_ratemaps_pc_torch(model, trajectory_generator, options, res=20, n_avg=None, Np=12, idxs=None):
#     '''Compute spatial firing fields'''
# 
#     if not n_avg:
#         n_avg = 1000 // options.sequence_length
# 
#     if not np.any(idxs):
#         idxs = np.arange(Np)
#     idxs = idxs[:Np]
#     
#     p = torch.zeros([n_avg, options.batch_size * options.sequence_length, Np]).to(options.device)
#     pos = torch.zeros([n_avg, options.batch_size * options.sequence_length, 2]).to(options.device)
# 
#     activations = torch.zeros([Np, res, res]).to(options.device)
#     counts  = torch.zeros([res, res]).to(options.device)
# 
#     for index in range(n_avg):
#         inputs, pos_batch = trajectory_generator.get_test_batch()
#         
#         
#         p_batch = model.predict(inputs).detach()
#         
#         
#         pos_batch = torch.reshape(pos_batch, [-1, 2])
#         p_batch = p_batch[:,:,idxs].reshape(-1, Np)
#         
#         p[index] = p_batch
#         pos[index] = pos_batch
# 
#         x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
#         y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res
#         
#         for i in range(options.batch_size*options.sequence_length):
#             x = x_batch[i]
#             y = y_batch[i]
#             if x >=0 and x < res and y >=0 and y < res:
#                 counts[int(x), int(y)] += 1
#                 activations[:, int(x), int(y)] += p_batch[i, :]
# 
#     activations = torch.nan_to_num(activations/counts)
#                 
#     p = p.reshape([-1, Np])
#     pos = pos.reshape([-1, 2])
# 
#     # # scipy binned_statistic_2d is slightly slower
#     # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
#     rate_map = activations.reshape(Np, -1)
#     
#     pcscores = []
#     for i in range(Np):
#         im = (activations[i,:,:] - torch.min(activations[i,:,:])) / (torch.max(activations[i,:,:]) - torch.min(activations[i,:,:]))
#         pcscores.append(placescores_torch.pc_score(im, 0.1))
#     
#     return activations, rate_map, p, pos, pcscores
# =============================================================================


def save_ratemaps(model, trajectory_generator, options, step, res=20, n_avg=None):
    if not n_avg:
        n_avg = 1000 // options.sequence_length
    activations, rate_map, g, pos = compute_ratemaps(model, trajectory_generator,
                                                     options, res=res, n_avg=n_avg)
    rm_fig = plot_ratemaps(activations, n_plots=len(activations))
    imdir = options.save_dir + "/" + options.run_ID
    imsave(imdir + "/" + str(step) + ".png", rm_fig)


def plot_images(activations, scores = None, smooth = True, regularize = True, save = False, savename = None, show = True, num_images=64):
    """
    Plot a selection of images from the activations array in a grid format, sorted by scores.

    Parameters:
    activations (numpy.ndarray): An array of images, shape (num_images, height, width)
    scores (list or numpy.ndarray): An array of scores corresponding to each image.
    num_images (int): Number of images to plot, default is 64
    """
    # Ensure that activations has the right dimensions
    if len(activations.shape) != 3:
        raise ValueError("Activations should be a 3D array of shape (num_images, height, width)")

    # Check if there are enough images
    if activations.shape[0] < num_images:
        raise ValueError("There are not enough images to plot. The activations array has fewer images than requested.")

    # Sort images based on scores
    if scores is not None:
        sorted_indices = np.argsort(-np.array(scores))  # Sort and get indices
        activations = activations.copy()[sorted_indices]

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Plot each image
    for i in range(num_images):
        im = activations[i]
        if smooth:
            im = cv2.GaussianBlur(im, (3, 3), sigmaX=1, sigmaY=0)
        if regularize and (np.max(im) - np.min(im)) > 0:
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
        ax = axes[i]
        ax.imshow(im, cmap='jet')
        ax.axis('off')  # Turn off axis

    # Turn off any unused subplots
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    plt.close()

