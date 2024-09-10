import scipy.io
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt
import torch
from pc_info import I_spike, I_spike_joint

IDs = [1,4,5,6,12,13]

pc = torch.zeros((44*9, len(IDs)))
rate_maps = torch.zeros((len(IDs), 44,9))

for i, ID in enumerate(IDs): #range(1,19):
    position_path = 'matData/MonkeyPosition_ID' + str(ID) + '.mat'
    spiketime_path = 'matData/SpikeTimeStamp_ID' + str(ID) + '.mat'
    
    MonkeyPosition = scipy.io.loadmat(position_path)['MonkeyPosition']
    SpikeTimeStamp = scipy.io.loadmat(spiketime_path)['SpTimeStamp']
    
    # Example function to perform data manipulation for firing map
    def compute_firing_map(position_data, spike_times, bin_size=10, speed_threshold=20, smoothing_sigma=1.5, min_occupancy=0.125):
        # Load position and spike data
        positions = pd.DataFrame(position_data, columns=['frame', 'x', 'y'])
        spikes = pd.DataFrame(spike_times, columns=['timestamp'])
        
        # Calculate speed
        positions['x_diff'] = positions['x'].diff()
        positions['y_diff'] = positions['y'].diff()
        positions['time_diff'] = positions['frame'].diff()
        positions['speed'] = np.sqrt(positions['x_diff']**2 + positions['y_diff']**2) / positions['time_diff']
        
        # Filter out immobility
        positions_filtered = positions[positions['speed'] >= speed_threshold]
        
        # Create bins
        x_bins = np.arange(0, 441, bin_size)
        y_bins = np.arange(0, 100, bin_size)
        
        # Initialize arrays for spike counts and occupancy
        spike_counts = np.zeros((len(x_bins)-1, len(y_bins)-1))
        occupancy = np.zeros((len(x_bins)-1, len(y_bins)-1))
        
        # Bin the position data
        positions_filtered['x_bin'] = np.digitize(positions_filtered['x'], x_bins) - 1
        positions_filtered['y_bin'] = np.digitize(positions_filtered['y'], y_bins) - 1
        
        # Calculate occupancy time per bin
        for index, row in positions_filtered.iterrows():
            x_bin = int(row['x_bin'])
            y_bin = int(row['y_bin'])
            if 0 <= x_bin < len(x_bins)-1 and 0 <= y_bin < len(y_bins)-1:
                occupancy[x_bin, y_bin] += row['time_diff']
        
        # Bin the spike data
        for timestamp in spikes['timestamp']:
            frame_idx = positions_filtered['frame'].searchsorted(timestamp)
            if frame_idx < len(positions_filtered):
                x_bin = positions_filtered.iloc[frame_idx]['x_bin']
                y_bin = positions_filtered.iloc[frame_idx]['y_bin']
                if 0 <= x_bin < len(x_bins)-1 and 0 <= y_bin < len(y_bins)-1:
                    spike_counts[int(x_bin), int(y_bin)] += 1
        
        # Apply Gaussian smoothing
        spike_counts_smooth = scipy.ndimage.gaussian_filter(spike_counts, smoothing_sigma)
        occupancy_smooth = scipy.ndimage.gaussian_filter(occupancy, smoothing_sigma)
        
        # Compute firing rate map
        firing_rate_map = np.divide(spike_counts_smooth, occupancy_smooth, where=occupancy_smooth > min_occupancy)
        
        return firing_rate_map, occupancy_smooth, spike_counts_smooth
    
    
    # Compute firing map
    firing_map, occupancy_smooth, spike_counts_smooth = compute_firing_map(MonkeyPosition, SpikeTimeStamp)
    
    rate_maps[i,:,:] = torch.tensor(firing_map)
    pc[:,i] = rate_maps[i,:,:].flatten()
    
pc = pc.unsqueeze(1)

#%%
dist = 1/pc.shape[0] * torch.ones(pc.shape[0])

infos = I_spike(pc, dist).squeeze(0)
J = I_spike_joint(pc, dist).squeeze(0)

# Plot configuration
n_images = rate_maps.shape[0]
n_rows = 2
n_cols = 3

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3, 6))

# Plot each image and corresponding value
for i in range(n_images):
    ax = axes[i // n_cols, i % n_cols]
    ax.imshow(rate_maps[i], cmap='viridis')  # Plot the image
    ax.axis('off')  # Turn off axis

    # Annotate with the corresponding value
    value = infos[i]
    ax.set_title(f"{value:.2f}", fontsize=14)

# Hide any unused subplots
for j in range(i+1, n_rows * n_cols):
    fig.delaxes(axes[j // n_cols, j % n_cols])

plt.tight_layout()
plt.suptitle('Skaggs Info: I(A)', fontsize = 16)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.show()

#%%
from itertools import combinations
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

rate_map_pairs = list(combinations(range(len(rate_maps)), 2))

# Plot configuration
n_pairs = len(rate_map_pairs)
n_rows = 3
n_cols = 5

fig = plt.figure(figsize=(10, 12))
gs = GridSpec(n_rows, n_cols, figure=fig)

# Iterate over each pair of rate maps
for idx, (i, j) in enumerate(rate_map_pairs):
    row = idx // n_cols
    col = idx % n_cols
    
    # Create a new GridSpecFromSubplotSpec for each subplot with an additional row for the title
    inner_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[row, col], height_ratios=[0, 1], wspace=-0.2)

    # Create an axis for the title that spans both columns
    title_ax = fig.add_subplot(inner_gs[0, :])
    title_ax.axis('off')  # Turn off axis for the title area

    # Annotate the title
    pair_value = J[i, j]
    title_ax.set_title(f"{pair_value:.2f}, {infos[i] + infos[j]:.2f}", fontsize=18)

    # Create subplots for each rate map within the grid
    ax1 = fig.add_subplot(inner_gs[1, 0])
    ax2 = fig.add_subplot(inner_gs[1, 1])

    # Plot each rate map in its subplot
    ax1.imshow(rate_maps[i], cmap='viridis')
    ax1.axis('off')  # Turn off axis for the first map

    ax2.imshow(rate_maps[j], cmap='viridis')
    ax2.axis('off')  # Turn off axis for the second map

# Hide any unused subplots
for k in range(idx + 1, n_rows * n_cols):
    fig.add_subplot(gs[k // n_cols, k % n_cols]).axis('off')

L, V = torch.linalg.eig(J)
lambda_1 = torch.max(torch.view_as_real(L)[:, 0])

plt.suptitle(f"I(A,B), I(A) + I(B)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()