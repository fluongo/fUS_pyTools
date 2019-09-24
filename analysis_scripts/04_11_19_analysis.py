
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
# 13 experiments// all 3 spot
fns = sorted(glob.glob('/data/fUS_project/data/data_apr11/RT*.mat'))
timelines = len(fns)*['/data/fUS_project/data/data_apr11/timeline_04-11-2019_12-14.mat']

all_exps = [[i, j,k] for i,j,k in zip(fns, timelines, range(len(fns)))]

do_resample = False
ds_factor = 2;
n_trials = 20;

corr_list = []
corr_list_zscore = []
corr_im_list = []
trials_list = []
data_raw_list = []

# Experiment number 4 (5th experiment) of first timeline file is wrong

# Export tiffs of all experiments
for ii in range(len(all_exps)):
    animal_fn, timeline_fn, exp_number = all_exps[ii]
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
    fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.1, interval_between_experiments = 3,  min_number_exp = 40); # Separate the frames for each one...
    
    # Recompute at appropriate times
    stim_ts = stim_list[exp_number];
    fus_ts = fus_list[exp_number];

    # Check that the number of fus timestamps matches, sometimes there are two extra timeline ttls
    if ii in [3]:
        fus_ts = fus_ts[2:]; # Chop off two

    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: 30]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
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
            stim[k, i*30 + 10*k : i*30 + 10*k + 5] = 1;
    #plt.plot(stim.T)

    # Compute the correlation
    plt.figure(figsize = [20, 3])
    [nT, nY, nX] = data_resample_at_stim.shape

    # Make dff
    f0 = np.nanmean(data_resample_at_stim, axis = 0)
    f0 = np.transpose(np.dstack([f0 for i in range(nT)]), [2,0,1])
    data_resample_at_stim = (data_resample_at_stim - f0)/f0

    corr_maps = np.zeros([3, nY, nX])
    corr_maps_zscore = np.zeros([3, nY, nX])
    
    for aa in range(3):
        for xx in range(nX):
            for yy in range(nY):
                corr_maps[aa, yy, xx] = scnpix.nan_pearson(data_resample_at_stim[:, yy, xx], stim[aa])
        plt.subplot(1,4,aa+1)
        plt.imshow(corr_maps[aa, :, :], vmin = 0, vmax = 0.3, cmap = 'afmhot')

        # Zscore and median filter
        #corr_maps_zscore[aa, :, :] = np.copy((corr_maps[aa, :, :] - corr_maps[aa, :, :].mean())/corr_maps[aa, :, :].std())
        #plt.imshow(corr_maps[aa, :, :], vmin = z_cutoff, cmap = 'afmhot')
        #plt.imshow(corr_maps[aa, :, :], vmin = corr_cutoff, cmap = 'afmhot')

    # corr_maps_im = np.transpose(corr_maps_zscore, [1,2,0]).copy()
    # corr_maps_im[corr_maps_im < z_cutoff] = 0;
    # corr_maps_im = np.uint8(255*corr_maps_im/np.max(corr_maps_im))
    # plt.subplot(1,4,4); plt.imshow(corr_maps_im)
    # plt.suptitle(animal_fn)
    corr_list.append(corr_maps)

    # # Compute the trial mean
    [t, y, x] = data_resample_at_stim.shape
    trials = data_resample_at_stim.reshape([n_trials, int(t/n_trials), y, x])
    trials_export = np.maximum(np.nanmean(trials, axis = 0), 0)
    trials_export = trials_export/np.max(trials_export)

    trials_list.append(trials_export)
    plt.savefig(exportDir + sub_fn[:-4]+'_trial_average.pdf')

#%% Now do all of the plots again

# plt.figure()
# tmp = np.stack(corr_im_list, axis = -1)
# tmp = np.transpose(tmp[14:20, :, :, :].mean(axis = 0), [2, 0, 1])
# plt.figure
# plt.imshow(tmp,aspect = 10)

from scipy.signal import medfilt

z_cutoff = 0
plt.figure(figsize = [25, 30])
for i in range(len(all_exps)):
    tmp_im = np.zeros_like(corr_list[0])
    for aa in range(3):
        plt.subplot(13, 4, 4*i+aa+1)

        # tmp_im[aa, :, :] = medfilt(corr_list_zscore[i][aa, :, :], kernel_size = 3)
        # plt.imshow(tmp_im[aa, :, :], vmin = z_cutoff, cmap = 'afmhot'); plt.axis('off')
        plt.imshow(corr_maps, vmin = 0,vmax = 0.1, cmap = 'afmhot'); plt.axis('off')

    # plt.subplot(13, 4, 4*i+4)
    # color_tmp = np.transpose(tmp_im, [1,2,0]).copy()
    # color_tmp[color_tmp < z_cutoff] = 0;
    # color_tmp = np.uint8(255*color_tmp/np.max(color_tmp))
    # plt.imshow(color_tmp); plt.axis('off')

#%%
# Export the cumulative tiff for trial averaged
all_trials = np.zeros([30, 4*nY, 4*nX])
n_fus_frames = data_raw_list[0].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 4*nY, 4*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [4, 4])
    all_trials[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = trials_list[i]
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = np.transpose(data_raw_list[i], [0,2,1])
scio.export_tiffs(all_trials, exportDir + 'cumulative_trials.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_raw, exportDir + 'cumulative_trials_raw.tiff', dims = {'x':2, 'y':1, 't':0})


