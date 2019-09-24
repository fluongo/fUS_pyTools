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

# 4 slices over 3mm e.g. 750uM apart

##################################################################
# Load the fUS data
##################################################################
fns = sorted(glob.glob('/data/fUS_project/data/data_July18/RT*.mat'))[5:] # Only first 5
timelines = len(fns)*['/data/fUS_project/data/data_July18/timeline_07-19-2019_12-39_retin_ForRev_10reps_20total_600fr.mat']
n_fus = range(len(fns))
n_stim = range(len(fns))

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10;
fus_rate =  2; # THIS EXPEIMENT WAS 2 HZ


#%%

all_exp_dicts = []

# Export tiffs of all experiments
for ii in range(len(all_exps)):
    animal_fn, timeline_fn, exp_number, exp_number_fus, exp_number_stim = all_exps[ii]
    outDir, sub_fn = os.path.split(animal_fn)
    exportDir = outDir + '/extras/'

    if not os.path.exists(exportDir):
        os.mkdir(exportDir)

    # Load data
    m = scio.matloader(); # Instantiate mat object
    m.loadmat_h5(animal_fn); #m.summary()
    data_raw = m.data['Dop'].copy(); # Take the square root...

    # # # Load timeline only on the first experiment
    if ii in [0]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()
        stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 4,  min_number_exp = 100); # Separate the frames for each one...
        fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 3,  min_number_exp = 100); # Separate the frames for each one...

    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus];

    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})
    [nT, nY, nX] = data_resample_at_stim.shape
    # NOTE: nT is 800, which is 20 trials x 40 frames or 20seconds for each trial

    #########################################################
    ########################################################
    #### RETINOITOPUY
    ##########################################################
    ############################################################

    # # Compute the trial for each one
    trials_all = {}
    trials_all['az_LR'] = data_resample_at_stim[:200, :, :]
    trials_all['az_RL'] = data_resample_at_stim[200:400, :, :]
    trials_all['ev_UD'] = data_resample_at_stim[400:600, :, :]
    trials_all['ev_DU'] = data_resample_at_stim[600:800, :, :]
    
    # Compute the fus_stim_time for each frame
    fus_ts_stim = np.zeros_like(fus_ts)
    for ww in range(data_raw.shape[0]):
        fus_ts_stim[ww] = np.argmin(np.abs(stim_ts - fus_ts[ww]));

    curr_dict = {}
    curr_dict['stim_ts'] = stim_ts
    curr_dict['fus_ts'] = fus_ts
    curr_dict['fus_ts_stim'] = fus_ts_stim
    curr_dict['data_raw'] = data_raw
    curr_dict['data_raw_medfilt'] = median_filter(data_raw, size = [8,5,5])
    curr_dict['data_resample_at_stim'] = data_resample_at_stim
    curr_dict['trials_all'] = trials_all

    all_exp_dicts.append(curr_dict)


#%%
# # Export the cumulative tiff for trial averaged
#all_trials = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
#all_trials_dff = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 4*nY, 4*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, 4*nY, 4*nX])

nF_stim = nT;
all_trials_resampled_at_stim = np.zeros([nF_stim, 4*nY, 4*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [4, 4])
    exp_dict = all_exp_dicts[i]

    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]         = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = exp_dict['data_resample_at_stim']

#scio.export_tiffs(all_trials, exportDir + 'cumulative_trials_spots.tiff', dims = {'x':2, 'y':1, 't':0})
#scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_spots_dff.tiff', dims = {'x':2, 'y':1, 't':0})

# Export raw
scio.export_tiffs(all_trials_raw, exportDir + 'RETINOTOPY_cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_raw_medfilt, exportDir + 'RETINOTOPY_cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})


#%%


exp_dict = all_exp_dicts[2]; # Dictionary for current experiment
Y = exp_dict['data_raw_medfilt']
Y = (Y - np.mean(Y))/np.mean(Y);
X = exp_dict['fus_ts_stim']

# trim the first and last frame
idx = np.where(np.logical_and(X>0, X<2999)); # 600 frames for first 5 trials
X = X[idx]; X = np.mod(X, 800)
Y =Y[idx]

plt.scatter(X, Y, s = 5)


#%% POWER ANALYSIS

for xx in range(3):
    exp_dict = all_exp_dicts[xx]; # Dictionary for current experiment
    idx_fft = 5;
    plt.figure(figsize = [20, 5])
    #NON TRIAL BASED

    fourier_results = {}
    alpha_mask = np.zeros([nY, nX]); # For making a mask of the relevant areas
    for pp, stim_key in enumerate(['az_LR', 'az_RL', 'ev_DU', 'ev_UD']):
        data = exp_dict['trials_all'][stim_key]
        data_zs = np.zeros_like(data)

        for ii in range(nY):
            for jj in range(nX):
                data_zs[:, ii, jj] = (data[:, ii, jj] - np.mean(data[:, ii, jj]))/np.std(data[:, ii, jj])

        out = np.fft.fft(data_zs, axis = 0); # take along time

        fourier_results[stim_key] = {}
        if stim_key in ['az_LR', 'ev_DU']: # Flip one of them
            fourier_results[stim_key]['phase'] = -np.angle(out[idx_fft, :, :])
        else:
            fourier_results[stim_key]['phase'] = np.angle(out[idx_fft, :, :])
        fourier_results[stim_key]['power'] =np.abs(out[idx_fft, :, :])

        plt.subplot(2,4, pp+1)
        plt.imshow(fourier_results[stim_key]['power'], cmap = 'binary'); plt.title(stim_key)
        plt.subplot(2,4, pp+5)
        plt.imshow(fourier_results[stim_key]['phase'], cmap = 'hsv'); plt.title(stim_key)
        
        alpha_mask += fourier_results[stim_key]['power']

    # Make the alpha mask
    alpha_mask = gaussian_filter(alpha_mask, 2);
    alpha_mask = alpha_mask/np.max(alpha_mask);
    alpha_mask[alpha_mask>0.5] = 1;
    alpha_mask[alpha_mask<=0.5] = 0;

    # NOW COMBINE THEM INTO A SINGLE MAP
    plt.figure(figsize = [12, 2])
    for ll, (p1, p2) in enumerate([('az_LR', 'az_RL'), ('ev_DU', 'ev_UD')]):
        plt.subplot(1,2,ll+1)
        combined_map = ((fourier_results[p1]['phase'] + fourier_results[p2]['phase']) )/2

        # Set all outside values to zero
        combined_map = (combined_map + np.pi) * alpha_mask
        # Make custom colormap with zero value as black
        hsv_custom = cm.get_cmap('hsv', 256)
        hsv_custom = hsv_custom(np.linspace(0, 1, 256))
        hsv_custom[0, :] = 0; # Turn 0 to black
        hsv_custom = ListedColormap(hsv_custom)

        plt.imshow(combined_map, cmap = hsv_custom); plt.colorbar()
        plt.contour(alpha_mask, [0.5], colors = 'w', linewidths = 2)
        plt.title(p1)

#%%
