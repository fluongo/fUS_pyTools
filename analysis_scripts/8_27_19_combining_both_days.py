#%%
# Analysis example
import glob
import os
import sys
from importlib import reload

sys.path.append('/data/git_repositories_py/SCRAPC/')
sys.path.append('/data/git_repositories_py/fUS_pytools/')

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import scrapc_analysis as scana
import scrapc_io as scio
import scrapc_viz as scviz
import scrapc_npix as scnpix
import scrapc_imfilters as scfilters
from scipy.ndimage import gaussian_filter, median_filter    
from scipy.ndimage.filters import median_filter

from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import ListedColormap

from PIL import Image

dir_aug26 = '/data/fUS_project/data/data_aug26/extras/'
dir_aug27 = '/data/fUS_project/data/data_aug27/extras/'

#comb1 = np.load(dir_aug26 + 'combined_single_trial.npy')
#comb2 = np.load(dir_aug27 + 'combined_single_trial.npy')
comb1 = np.load(dir_aug26 + 'interpolated_trial_average_aug26.npy')
comb2 = np.load(dir_aug27 + 'interpolated_trial_average_aug27.npy')

nX = 128
nY = 52

combined_full = np.zeros([1200, 4*52, 10*nX])
for i in range(10):
    if np.mod(i, 2) == 0:
        combined_full[:, :, nX*i:nX*(i+1)] = comb1[:, :, int((i/2))*nX : int(((i/2 + 1))*nX)]
    else:
        combined_full[:, :, nX*i:nX*(i+1)] = comb2[:, :, int(i/2)*nX : (int(i/2) + 1)*nX]

f0 = combined_full.mean(axis = 0)
for kk in range(1200):
    combined_full[kk, :, :] = combined_full[kk, :, :] - f0;

# MEdian filter before the fft
#combined_full_filt = median_filter(combined_full, size = [8, 8, 6])
combined_full_filt = median_filter(combined_full[range(0, 1200, 20), :, :], size = [8, 8, 6])

#%% Compute the power and phase

out = np.fft.fft(combined_full_filt, axis = 0); # take along time
#out = np.fft.fft(combined_full, axis = 0); # take along time

# Position 1 is stimulus harmonic
plt.figure(figsize = [20, 10])
phase_map = np.angle(out[1, :, :])
power_map = np.abs(out[1, :, :])
phase_map[52:2*52, :] = -phase_map[52:2*52, :]
phase_map[52*3:4*52, :] = -phase_map[52*3:4*52, :]
# Combine phases
phase_average = phase_map.copy()
phase_average[52*0:1*52, :] = (phase_average[52*0:1*52, :] + phase_average[52*1:2*52, :])/2
phase_average[52*1:2*52, :] = (phase_average[52*2:3*52, :] + phase_average[52*3:4*52, :])/2
phase_average = phase_average[0:52*2, :]

plt.figure(figsize = [20, 5]); 
plt.imshow(phase_map, cmap = 'hsv');
plt.figure(figsize = [20, 5]); 
plt.imshow(power_map, cmap = 'binary');
plt.figure(figsize = [20, 5])
plt.imshow(phase_average, cmap = 'hsv'); plt.colorbar()

power_re = power_map.reshape([4, 52, 128*10]).sum(axis = 0)
alpha_map = median_filter(power_re, [15, 15])
# Rescale each slice between 0 and 1
alpha_map_scaled = np.zeros_like(alpha_map)
for xx in range(10):
    tmp = alpha_map[:, xx*nX:(xx+1)*nX]
    alpha_map_scaled[:, xx*nX:(xx+1)*nX] = tmp/np.max(tmp)
alpha_cut = 85;

show_masks = False
if show_masks:
    #alpha_map = gaussian_filter(power_re, 5)
    plt.figure(figsize = [20, 2]); 
    plt.imshow(alpha_map, cmap = 'binary'); plt.title('no scaling')
    plt.figure(figsize = [20, 2]); 
    plt.imshow(alpha_map_scaled, cmap = 'binary'); plt.title('each one rescaled')
    plt.figure(figsize = [20, 2]); 
    plt.imshow(alpha_map, vmax = np.percentile(alpha_map.ravel(), alpha_cut), cmap = 'binary'); 
    plt.title('cut without scaling')
    plt.figure(figsize = [20, 2]); 
    plt.imshow(alpha_map_scaled, vmax = np.percentile(alpha_map_scaled.ravel(), alpha_cut), cmap = 'binary'); plt.title('cut and scaling')

mask_to_use = alpha_map

plt.figure(figsize = [20, 2]); 
plt.subplot(2,1,1)
masked = np.ma.masked_where(mask_to_use < np.percentile(mask_to_use, alpha_cut), phase_average[:52, :])
plt.imshow(masked, cmap = 'hsv'); plt.title('Aximuth'); 

plt.subplot(2,1,2)
masked = np.ma.masked_where(mask_to_use < np.percentile(mask_to_use, alpha_cut), phase_average[52:, :])
plt.imshow(masked, cmap = 'hsv'); plt.title('Elevation'); 


#%% Compute field sign


plt.figure(figsize = [20, 5])
plt.imshow(phase_average, cmap = 'hsv'); plt.colorbar()


mapH = phase_average[:52, :]
mapV = phase_average[52:, :]

plt.figure(figsize = [20, 5])
plt.imshow(mapH, cmap = 'hsv')
plt.figure(figsize = [20, 5])
plt.imshow(mapV, cmap = 'hsv')

dyH, dxH = np.gradient(mapH)
dyV, dxV = np.gradient(mapV)
angleH = (dxH < 0) * np.pi + np.arctan(dyH / dxH);
angleV = (dxV < 0) * np.pi + np.arctan(dyV / dxV);
field_sign = np.sin(angleV - angleH)

plt.figure(figsize = [20, 2])
plt.imshow(gaussian_filter(field_sign, sigma = 1), cmap = 'bwr'); #plt.colorbar()

#%% Delineate borders by visualizing only the reversal

mapH = phase_average[:52, :]
mapV = phase_average[52:, :]
dyH, dxH = np.gradient(mapH)
dyV, dxV = np.gradient(mapV)

# Reshape for beter visualization
#mask_re = np.zeros([2*54, 5*128])
alpha_cut = 50

plt.figure(figsize = [40, 10]); 
plt.subplot(5,1,1); plt.title('Aximuth'); 
masked = np.ma.masked_where(mask_to_use < np.percentile(mask_to_use, alpha_cut), phase_average[:52, :])
plt.imshow(masked, cmap = 'hsv'); 

plt.subplot(5,1,2); plt.title('Elevation');
masked = np.ma.masked_where(mask_to_use < np.percentile(mask_to_use, alpha_cut), phase_average[52:, :])
plt.imshow(masked, cmap = 'hsv');  

plt.subplot(5,1,3); plt.title('masked azimuth X gradient gf1')
masked = np.ma.masked_where(mask_to_use < np.percentile(mask_to_use, alpha_cut), gaussian_filter(dxH, sigma = 1))
plt.imshow(masked, vmin = -0.1, vmax = 0.1, cmap = 'bwr'); 

plt.subplot(5,1,4); plt.title('azimuth X gradient gf1')
plt.imshow(gaussian_filter(dxH, sigma = 1), vmin = -0.1, vmax = 0.1, cmap = 'bwr');

plt.subplot(5,1,5); plt.title('azimuth X gradient gf1')
plt.imshow(np.abs(gaussian_filter(dyV, sigma = 1)) + np.abs(gaussian_filter(dxV, sigma = 1)), vmin = 0, vmax = 0.2, cmap = 'binary')

#%%

nn = 3; # WHich slice.....

plt.figure(figsize = [20, 5]); plt.suptitle('X gradient')
plt.subplot(1,2,1); plt.imshow(dxH[:, nn*128:(nn+1)*128], vmin = -0.1, vmax = 0.1, cmap = 'bwr'); plt.colorbar()
plt.subplot(1,2,2); plt.imshow(dxV[:, nn*128:(nn+1)*128], vmin = -0.1, vmax = 0.1, cmap = 'bwr'); plt.colorbar()
plt.figure(figsize = [20, 5]); plt.suptitle('X gradient median filter')
plt.subplot(1,2,1); plt.imshow(median_filter(dxH[:, nn*128:(nn+1)*128], size = [10, 10]), vmin = -0.1, vmax = 0.1, cmap = 'bwr'); plt.colorbar()
plt.subplot(1,2,2); plt.imshow(median_filter(dxV[:, nn*128:(nn+1)*128], size = [10, 10]), vmin = -0.1, vmax = 0.1, cmap = 'bwr'); plt.colorbar()


plt.figure(figsize = [20, 5]); plt.suptitle('Y gradient')
plt.subplot(1,2,1); plt.imshow(dyH[:, nn*128:(nn+1)*128], vmin = -0.1, vmax = 0.1, cmap = 'bwr'); plt.colorbar()
plt.subplot(1,2,2); plt.imshow(dyV[:, nn*128:(nn+1)*128], vmin = -0.1, vmax = 0.1, cmap = 'bwr'); plt.colorbar()


#plt.imshow(mapH.reshape([52, 10, 128])[:, 0, :])

#%% Load in the other data and try to segment in a similiar way


fullfield_fn = '/data/fUS_project/data/data_aug29/extras/interpolated_trial_average_fullfield.npy'
combined_ff = np.load(fullfield_fn)
combined_ff_filt = median_filter(combined_ff[range(0, 600, 20), :, :], size = [15, 15, 6])

# Do the fft to extract power and phase
out_ff = np.fft.fft(combined_ff_filt, axis = 0); # take along time
phase_map = np.angle(out_ff[1, :, :])
power_map = np.abs(out_ff[1, :, :])

# Position 1 is stimulus harmonic
plt.figure(figsize = [20, 10])
plt.imshow(phase_map)
plt.figure(figsize = [20, 10])
plt.imshow(power_map)

#%%
plt.figure(figsize = [40, 5])

dyP, dxP = np.gradient(power_map)
plt.subplot(4,1, 1); plt.imshow(dxP, vmin = -0.1, vmax = 0.1, cmap = 'bwr');
plt.subplot(4,1, 2); plt.imshow(dyP, vmin = -0.1, vmax = 0.1, cmap = 'bwr');
plt.subplot(4,1, 3); plt.imshow(np.abs(dxP)+ np.abs(dyP), vmin = 0, vmax = 0.2, cmap = 'binary');
plt.subplot(4,1, 4); plt.imshow(gaussian_filter(np.abs(dxP)+ np.abs(dyP), sigma = 3), vmin = 0, vmax = 0.2, cmap = 'binary');


#%% Load the anatomical dataset



