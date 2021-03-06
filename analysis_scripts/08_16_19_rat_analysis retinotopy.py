
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

##################################################################
# Load the fUS data
##################################################################
fns = sorted(glob.glob('/data/fUS_project/data/data_aug16/RT*.mat')); # Only first 7 are the retinotopu
timelines = len(fns)*['/data/fUS_project/data/data_aug16/timeline_08-16-2019_12-31.mat']
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
        stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 4,  min_number_exp = 1000); # Separate the frames for each one...
        fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 3,  min_number_exp = 100); # Separate the frames for each one...
        for tmp_ss, tmp_ff in zip(stim_list, fus_list):
            print('Number of frames stim: %d // fus: %d' % (len(tmp_ss), len(tmp_ff)))

    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus];

    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: int(20/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
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
    trials_all['az_LR'] = data_resample_at_stim[:300, :, :]
    trials_all['az_RL'] = data_resample_at_stim[300:600, :, :]
    trials_all['ev_UD'] = data_resample_at_stim[600:900, :, :]
    trials_all['ev_DU'] = data_resample_at_stim[900:1200, :, :]
    
    # Compute the fus_stim_time for each frame
    fus_ts_stim = np.zeros_like(fus_ts)
    for ww in range(data_raw.shape[0]):
        fus_ts_stim[ww] = np.argmin(np.abs(stim_ts - fus_ts[ww]));

    curr_dict = {}
    curr_dict['stim_ts'] = stim_ts
    curr_dict['fus_ts'] = fus_ts
    curr_dict['fus_ts_stim'] = fus_ts_stim
    curr_dict['data_raw'] = data_raw
    curr_dict['data_raw_medfilt'] = median_filter(data_raw, size = [5,5,5])
    curr_dict['data_resample_at_stim'] = data_resample_at_stim
    curr_dict['trials_all'] = trials_all

    all_exp_dicts.append(curr_dict)


#%%
# # Export the cumulative tiff for trial averaged
#all_trials = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
#all_trials_dff = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 2*nY, 4*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, 2*nY, 4*nX])

nF_stim = nT;
all_trials_resampled_at_stim = np.zeros([nF_stim, 2*nY, 4*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [2, 4])
    exp_dict = all_exp_dicts[i]

    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]         = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]   = exp_dict['data_resample_at_stim']

    # TODO Make a version that does the trial averages....

#scio.export_tiffs(all_trials, exportDir + 'cumulative_trials_spots.tiff', dims = {'x':2, 'y':1, 't':0})
#scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_spots_dff.tiff', dims = {'x':2, 'y':1, 't':0})

# Export raw
scio.export_tiffs(all_trials_raw, exportDir + 'RETINOTOPY_cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_raw_medfilt, exportDir + 'RETINOTOPY_cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})


#%%
for xx in range(len(all_exps)):
    exp_dict = all_exp_dicts[xx]; # Dictionary for current experiment
    idx_fft = 5;
    plt.figure(figsize = [20, 4])

    #NON TRIAL BASED
    fourier_results = {}
    alpha_mask = np.zeros([nY, nX]); # For making a mask of the relevant areas
    for pp, stim_key in enumerate(['ev_UD']): # 'az_LR', 'az_RL', 'ev_DU', 
        data = exp_dict['trials_all'][stim_key]
        data_zs = np.zeros_like(data)

        for ii in range(nY):
            for jj in range(nX):
                data_zs[:, ii, jj] = (data[:, ii, jj] - np.mean(data[:, ii, jj]))/np.std(data[:, ii, jj])


        out = np.fft.fft(data_zs, axis = 0); # take along time
        plt.figure(figsize = [9, 9]); plt.suptitle('Slice %d' % xx)
        all_powers = []
        for i in range(49):
            plt.subplot(7, 7, i+1)
            plt.imshow(np.abs(out[i, :, :]))
            all_powers.append(np.sum(np.abs(out[i, :, :])))
            plt.title(str(i))
        plt.figure(); plt.plot(all_powers)

#%% POWER ANALYSIS

for xx in range(len(all_exps)):
    exp_dict = all_exp_dicts[xx]; # Dictionary for current experiment
    idx_fft = 5;
    plt.figure(figsize = [20, 4])
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
    alpha_mask[alpha_mask>0.6] = 1;
    alpha_mask[alpha_mask<=0.6] = 0;
    plt.savefig(outDir + '/slice_%d_parts.pdf' % xx)

    # NOW COMBINE THEM INTO A SINGLE MAP
    plt.figure(figsize = [10, 2])
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
    plt.savefig(outDir + '/slice_%d_masked.pdf' % xx)

#%% GAUSSIAN PROCESS ANALYSIS




# Compute a model for every pixel
import GPy
import seaborn as sns

# Observations
# only one type of tral
plt.figure(figsize = [20, 20])
for xx in range(25):
    idx = np.where((fus_ts_stim>5999) & (fus_ts_stim<=8999))

    cx = 40+xx
    cy = 17
    X = np.mod(fus_ts_stim[idx].reshape(-1, 1), 60)
    #y = all_trials_raw[idx, cy, cx]
    y = all_exp_dicts[6]['data_raw'][idx, cy, cx]

    # Condition y
    y = y-np.mean(y)
    y = y/np.max(y)
    plt.subplot(5,5,xx+1)
    plt.scatter(np.mod(X, 60), y, s=1); plt.title('X: %d Y: %d' %  (cx, cy))
    plt.ylim([np.percentile(y, 1), np.percentile(y, 90)])
    sns.despine()

#%%
kernel = GPy.kern.RBF(input_dim=1, variance = 1., lengthscale= 1.)
m = GPy.models.GPRegression(X, y.reshape(-1,1), kernel)

# the normal way
m.optimize(messages=True)
# with restarts to get better results
m.optimize_restarts(num_restarts = 20)


