
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
from scrapc_imfilters import remove_movement_artifact_from_raw_and_condition

from scipy.ndimage import gaussian_filter, median_filter    
from scipy.ndimage.filters import median_filter

from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import ListedColormap

from PIL import Image

##################################################################
# Load the fUS data
##################################################################
fns = sorted(glob.glob('/data/fUS_project/data/data_aug29/RT*.mat')); # Only first 7 are the retinotopu
timelines = len(fns)*['/data/fUS_project/data/data_aug29/timeline_08-29-2019_11-36.mat']

n_fus = range(len(fns))
n_stim = range(len(fns))

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10;
fus_rate =  2; # THIS EXPEIMENT WAS 2 HZ

# Load in example movies
example_mov_fn = '/data/linuxData/MASTER_STIMULUS_FOLDER/STIMULUS_GENERATION_HERE/Francisco/fUS_3point_retinotopy/fUS_FULLFIELD_Aug29_2019_5_10_5_10Reps.mat'
m = scio.matloader();
m.loadmat_h5(example_mov_fn); #timestamps_m.summary()
moviedata = m.data['moviedata'].copy()

# data_fix = remove_movement_artifact_from_raw_and_condition(data_raw)
# plt.figure(figsize = [20, 3]);
# plt.plot(data_fix[:, :40, :40].mean(axis = -1).mean(axis=-1))
# plt.figure(figsize = [20, 3]);
# plt.plot(data_raw[:, :40, :40].mean(axis = -1).mean(axis=-1))

# # Fix experiment 11 to remove an extra one
# timeline_fn = '/data/fUS_project/data/data_aug29/timeline_08-29-2019_11-36.mat'
# timestamps_m = scio.matloader();
# timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
# ts = timestamps_m.data['timestamps'].copy()
# stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 4,  min_number_exp = 1000); # Separate the frames for each one...
# fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 10,  min_number_exp = 100); # Separate the frames for each one...
# for exp_n, (tmp_ss, tmp_ff) in enumerate(zip(stim_list, fus_list)):
#     print('Number of frames stim: %d // fus: %d // %d' % (len(tmp_ss), len(tmp_ff), exp_n))

# Necessary to skip some steps
ii = 0
animal_fn, timeline_fn, exp_number, exp_number_fus, exp_number_stim = all_exps[ii]
outDir, sub_fn = os.path.split(animal_fn)
exportDir = outDir + '/extras/'

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

    # Artifact removed version of data
    data_fix = scfilters.remove_movement_artifact_from_raw_and_condition(data_raw)

    # # # Load timeline only on the first experiment
    if ii in [0]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()
        stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 4,  min_number_exp = 1000); # Separate the frames for each one...
        fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 10,  min_number_exp = 100); # Separate the frames for each one...
        for tmp_ss, tmp_ff in zip(stim_list, fus_list):
            print('Number of frames stim: %d // fus: %d' % (len(tmp_ss), len(tmp_ff)))

    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus][1:]; # Chop off first
    if ii in [11]: # Chop off last
        fus_ts = fus_ts[:-1]

    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    
    # Now do resample from the artifact cleaned version
    #data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})
    data_resample_at_stim = scana.resample_xyt(data_fix, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})
    
    # Remeber there are 30 sec gray at the beginning // Cut them out early
    print('Chopping off first 30 seconds of gray')
    data_resample_at_stim = data_resample_at_stim[60:, :, :]

    # Size if 460 x 52 x 128, with first 60 frames teh 30 sec of gray    
    [nT, nY, nX] = data_resample_at_stim.shape
    trials_all = data_resample_at_stim

    # Compute the fus_stim_time for each frame
    fus_ts_stim = np.zeros_like(fus_ts)
    for ww in range(data_raw.shape[0]):
        fus_ts_stim[ww] = np.argmin(np.abs(stim_ts - fus_ts[ww]));
    # Turn this into indices of the stimulus, e.g. cut out the first 900
    fus_ts_stim = np.maximum(fus_ts_stim-900, 0);

    curr_dict = {}
    curr_dict['stim_ts'] = stim_ts
    curr_dict['fus_ts'] = fus_ts
    curr_dict['fus_ts_stim'] = fus_ts_stim
    curr_dict['data_raw'] = data_raw
    curr_dict['data_raw_fix'] = data_fix
    curr_dict['data_raw_medfilt'] = median_filter(data_fix, size = [5,5,5])
    curr_dict['data_resample_at_stim'] = data_resample_at_stim
    curr_dict['trials_all'] = trials_all

    all_exp_dicts.append(curr_dict)

np.save(exportDir + 'data_processed_fullfield_aug29.npy', all_exp_dicts)

#%%
# # Export the cumulative tiff for trial averaged

# Size of exported image
#all_exp_dicts = np.load(exportDir + 'data_processed_fullfield_aug29.npy', allow_pickle = True)

[nT, nY, nX] = all_exp_dicts[0]['data_resample_at_stim'].shape
do_save = True

nWide = len(all_exp_dicts);

n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 1*nY, nWide*nX])
all_trials_raw_fix = np.zeros([n_fus_frames, 1*nY, nWide*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, 1*nY, nWide*nX])

nF_stim = nT;
all_trials_resampled_at_stim = np.zeros([nF_stim, 1*nY, nWide*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [1, nWide])
    exp_dict = all_exp_dicts[i]

    all_trials_raw_fix[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw_fix'], [0,2,1])
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                     = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]             = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]       = exp_dict['data_resample_at_stim']

# Export raw
if do_save:
    scio.export_tiffs(all_trials_resampled_at_stim, exportDir + 'FULLFIELD_cumulative_trials_resample_stim.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw, exportDir + 'FULLFIELD_cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_fix, exportDir + 'FULLFIELD_cumulative_trials_RAW_linear_interp_of_artifacts.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_medfilt, exportDir + 'FULLFIELD_cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})

# Make a DFF from median filter
all_trials_medfilt_dff  = all_trials_raw_medfilt.copy()
f0                      = np.median(all_trials_medfilt_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(n_fus_frames)]), [2,0,1])
all_trials_medfilt_dff  = (all_trials_medfilt_dff - f0)/f0

if do_save:
    scio.export_tiffs(all_trials_medfilt_dff, exportDir + 'FULLFIELD_RAW_medfilt_5_5_5_convert_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

# Do the same for the resampled at stim_trials with medfilt
trials_dff              = all_trials_resampled_at_stim.copy()
f0                      = np.median(trials_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(trials_dff.shape[0])]), [2,0,1])
trials_dff              = (trials_dff - f0)/f0
#trials_dff              = median_filter(trials_dff, size = [6,5,5])
if do_save:
    scio.export_tiffs(trials_dff, exportDir + 'FULLFIELD_trial_synchronized_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})


trial_average = trials_dff.reshape([10, 40, 52, nWide*nX]).mean(axis=0)
combined = trial_average

if do_save:
    np.save(exportDir + 'combined_single_trial.npy', combined)
    scio.export_tiffs(combined, exportDir + 'TRIAL_averages_all.tiff', dims = {'x':2, 'y':1, 't':0})
#%%
# Now add the movie
# max_val = np.max(combined)
# min_val = np.max(combined)
# combined_uber = np.zeros([40, 52, (nWide+1)*nX])
sub_movie = moviedata[]

# for cnt, xx in enumerate(['LR', 'RL', 'UD', 'DU']):
#     tmp = mov_examples[xx];
#     tmp = tmp-np.min(tmp); tmp = tmp/np.max(tmp);
#     combined_uber[:, cnt*52 : (cnt+1)*52, nWide*nX : (nWide+1)*nX] = tmp
# combined_uber[:, :, :5*nX] = combined

# if do_save: 
#     scio.export_tiffs(combined, exportDir + 'TRIAL_averages_all.tiff', dims = {'x':2, 'y':1, 't':0})
#     scio.export_tiffs(combined_uber, exportDir + 'TRIAL_averages_all_with_stimulus.tiff', dims = {'x':2, 'y':1, 't':0})

#%% COMBINE INTO MEGA KDS ESTIMATE

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import gaussian_kde

def compute_ts_smooth_kde(x_data, y_data, x_vals, window=60):
    # Computes a sliding window average of the data using a kernel density estimate of the mode
    # x_data and y_data is the function, x_vals is which values to interpolate at, window is the size of the window
    nT = len(x_vals)
    kde_smoothed = []
    for i in range(nT):
        subset_y = y_data[ (x_data> max(0, i-window/2) ) & (x_data<min(nT-2, i+window/2) )]
        kde_smoothed.append(kde_calc(subset_y, plot_hist = False))
    return kde_smoothed

def kde_calc(x, plot_hist = True):
    # Takes as input the x, does a kernel density smooth over time then returns data
    bandwidth = 5;
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    x_grid = np.linspace(-2, 2, 1000)
    kde_plot = kde.evaluate(x_grid)
    if plot_hist:
        plt.figure(); plt.plot(x_grid, kde_plot)
    return x_grid[np.argmax(kde_plot)]

nX = 128
nY = 52
nFrames_per_trial = 600
combined_smoothed = np.zeros([nFrames_per_trial, combined.shape[1], combined.shape[2]])

for exp_n in range(1):
    d = all_exp_dicts[exp_n]
    xx = d['fus_ts_stim'].copy()
    yy = d['data_raw_fix'].copy()
    
    # Only take indices that were during stimulus, exclude first and last frame
    idx = np.argwhere((xx < np.max(xx) ) & (xx> np.min(xx) ) )
    sub_x = np.mod(xx[idx], nFrames_per_trial)

    for xi in tqdm(range(nX)):
        print(xi)
        list_subx   = [sub_x for i in range(nY)]
        list_suby   = [(yy[idx, xi, yi] - np.median(yy[idx, xi, yi])) / np.median(yy[idx, xi, yi]) for yi in range(nY)]
        list_xvals  = [list(np.arange(nFrames_per_trial)) for i in range(nY)]
        results = Parallel(n_jobs=26)(delayed(compute_ts_smooth_kde)(i,j,k) for i,j,k in zip(list_subx,list_suby, list_xvals))
        # Now assign into their place
        for yi in range(nY):
            combined_smoothed[:, yi, exp_n*nX +xi] = results[yi]

#np.save(exportDir + 'interpolated_trial_average.npy', combined_smoothed)
#scio.export_tiffs(combined_smoothed, exportDir + 'TRIAL_SMOOTHED.tiff', dims = {'x':2, 'y':1, 't':0})



#%%

from scipy.ndimage import gaussian_filter

combined = np.load('/data/fUS_project/data/data_aug29/extras/interpolated_trial_average_fullfield.npy')

out = np.fft.fft(combined, axis = 0); # take along time

# Position 1 is stimulus harmonic
plt.figure(figsize = [20, 4])
phase_map = np.angle(out[1, :, :])
power_map = np.abs(out[1, :, :])
plt.subplot(2,1,1); plt.imshow(phase_map, cmap = 'hsv');
plt.subplot(2,1,2); plt.imshow(power_map, cmap = 'binary');

# Scale each experiment
nExp = len(all_exp_dicts)
power_map_scaled = np.zeros_like(power_map)
for i in range(nExp):
    tmp = power_map[:, i*nX:(i+1)*nX]
    tmp = tmp - np.min(tmp); tmp = tmp/np.max(tmp)
    power_map_scaled[:, i*nX:(i+1)*nX] = tmp
    
plt.figure(figsize = [20, 4])
plt.imshow(power_map, cmap = 'binary')
plt.figure(figsize = [20, 4])
plt.imshow(power_map_scaled, cmap = 'binary')

power_map_fold = np.zeros([2*nY, 7*nX])
power_map_fold[:nY, :] = power_map_scaled[:, :7*nX]
power_map_fold[nY:, :] = power_map_scaled[:, 7*nX:]
plt.figure(figsize = [20, 6])
plt.imshow(power_map_fold)
