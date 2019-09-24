#%%

# # Analysis example
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

##################################################################
# Load the fUS data
##################################################################
#%%
fns = sorted(glob.glob('/data/fUS_project/data/data_may03/RT*.mat'))
timelines = len(fns)*['/data/fUS_project/data/data_may03/timeline_05-03-2019_13-01.mat']
n_fus = range(11)
n_stim = [0,1,2,3,5,6,7,8,9,10, 11]

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

do_resample = False
ds_factor = 2;
n_trials = 10;
fus_rate =  2;

corr_list = []
corr_list_zscore = []
corr_im_list = []
trials_list = []; trials_list_dff = []
data_raw_list = []
# Experiment number 4 (5th experiment) of first timeline file is wrong

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
    data_raw = m.data['Dop'].copy()
    data_raw_list.append(data_raw)

    if do_resample:
        data_raw = scana.bin_2d(data_raw, ds_factor, dims = {'x':1, 'y':2, 't':0});
        data_raw = np.transpose(data_raw, (0, 2,1))

    # # Load timeline only on the first experiment
    if ii in [0]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()

    # # Compute the parsed version of the timestamps
    stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 3,  min_number_exp = 700); # Separate the frames for each one...
    
    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus];

    # Check that the number of fus timestamps matches, sometimes there are two extra timeline ttls
    if ii in [3]:
        fus_ts = fus_ts[1:]; # Chop off two
    if ii in [10]:
        fus_ts = fus_ts[2:]; # Chop off two
    

    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})

    # Top right discarding of invalid frames
    do_discard = True
    if do_discard:
        br = data_resample_at_stim[:, :,  -20:].mean(axis = -1).mean(axis = -1)
        invalid_frames = np.where(br > np.mean(br)+2*np.std(br))[0]
        #invalid_frames = np.where(br > np.mean(br))[0]
        
        print('Discarding %d frames' % len(invalid_frames))
        data_resample_at_stim[invalid_frames, : , :] = np.nan

    # Stimuli for each experiment for cross correlation
    stim = np.zeros([3, 600])
    for i in range(n_trials):
        for k in range(3):
            stim[k, i*60 + 20*k : i*60 + 20*k + 10] = 1;
    plt.plot(stim.T)

    # Compute the correlation
    plt.figure(figsize = [20, 3])
    [nT, nY, nX] = data_resample_at_stim.shape

    # Make dff
    f0 = np.nanmean(data_resample_at_stim, axis = 0)
    f0 = np.transpose(np.dstack([f0 for i in range(nT)]), [2,0,1])
    data_dff = (data_resample_at_stim - f0)/f0

    corr_maps = np.zeros([3, nY, nX])
    corr_maps_zscore = np.zeros([3, nY, nX])
    
    for aa in range(3):
        for xx in range(nX):
            for yy in range(nY):
                corr_maps[aa, yy, xx] = scnpix.nan_pearson(data_resample_at_stim[:, yy, xx], stim[aa])
        plt.subplot(1,4,aa+1)
        plt.imshow(corr_maps[aa, :, :], vmin = 0, vmax = 0.3, cmap = 'afmhot')

    corr_list.append(corr_maps)

    # # Compute the trial mean
    [t, y, x] = data_resample_at_stim.shape
    trials = data_resample_at_stim.reshape([n_trials, int(t/n_trials), y, x])
    trials_dff = data_dff.reshape([n_trials, int(t/n_trials), y, x])
    
    trials_export = np.maximum(np.nanmean(trials, axis = 0), 0)
    trials_export = trials_export/np.max(trials_export)
    trials_dff = np.maximum(np.nanmean(trials_dff, axis = 0), 0)
    trials_dff = trials_dff/np.max(trials_dff)
    
    trials_list.append(trials_export)
    trials_list_dff.append(trials_dff)


    plt.savefig(exportDir + sub_fn[:-4]+'_trial_average.pdf')

#%% Now do all of the plots again

from scipy.signal import medfilt

plt.figure(figsize = [25, 30])
for i in range(len(all_exps)):
    tmp_im = np.zeros_like(corr_list[0])
    for aa in range(3):
        plt.subplot(11, 4, 4*i+aa+1)

        tmp_im[aa, :, :] = corr_list[i][aa, :, :]/np.max(corr_list[i][aa, :, :])
        plt.imshow(corr_list[i][aa, :, :], vmin = 0.05, cmap = 'afmhot'); plt.axis('off'); plt.colorbar()
        
    plt.subplot(11, 4, 4*i+4)
    color_tmp = np.transpose(tmp_im, [1,2,0]).copy()
    color_tmp[color_tmp < z_cutoff] = 0;
    color_tmp = np.uint8(255*color_tmp/np.max(color_tmp))
    plt.imshow(color_tmp); plt.axis('off')

plt.savefig(exportDir + 'correlation_maps.pdf')

# SAME AS ABOVE/EXCEPT SMOOTHED

plt.figure(figsize = [25, 30])
# Only do the middle ones, excluding
for i in range(2, len(all_exps)-2):
    smoothed = np.stack(corr_list[i-2:i+3], axis = -1).mean(axis = -1)
    color_smoothed = np.copy(smoothed)
    for aa in range(3):
        plt.subplot(11, 4, 4*i+aa+1)
        plt.imshow(smoothed[aa, :, :], vmin=  0, cmap = 'afmhot'); plt.axis('off'); plt.colorbar()
        color_smoothed[aa, :, :] = color_smoothed[aa, :, :]/np.max(color_smoothed[aa, :, :]); # Normalize to max intensity for each channel
    plt.subplot(11, 4, 4*i+4)
    # Add code for cutting off low vals
    #color_smoothed
    color_tmp = np.transpose(color_smoothed, [1,2,0]).copy()
    color_tmp[color_tmp<0.5] = 0;
    plt.imshow(np.uint8(255*color_tmp)); plt.axis('off')

#plt.savefig(exportDir + 'correlation_maps_smoothed_across_planes.pdf')



#%%
# Export the cumulative tiff for trial averaged
all_trials = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
all_trials_dff = np.zeros([int(30*fus_rate), 4*nY, 4*nX])


n_fus_frames = data_raw_list[0].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 4*nY, 4*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [4, 4])
    all_trials[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = trials_list[i]
    all_trials_dff[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = trials_list_dff[i]
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = np.transpose(data_raw_list[i], [0,2,1])
scio.export_tiffs(all_trials, exportDir + 'cumulative_trials.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_dff.tiff', dims = {'x':2, 'y':1, 't':0})

scio.export_tiffs(all_trials_raw, exportDir + 'cumulative_trials_raw.tiff', dims = {'x':2, 'y':1, 't':0})


