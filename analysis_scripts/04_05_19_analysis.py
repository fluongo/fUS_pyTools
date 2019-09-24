
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
from scipy.ndimage import gaussian_filter, median_filter    


##################################################################
# Load the fUS data
##################################################################
#%%
# 20 experiments// 10 3 spot // 10 led
fns = sorted(glob.glob('/data/fUS_project/data/data_apr05/RT*.mat'))
timelines = 4*['/data/fUS_project/data/data_apr05/timeline_04-05-2019_11-22.mat'] + 8*['/data/fUS_project/data/data_apr05/timeline_04-05-2019_12-47.mat']


all_exps = [[i, j,k] for i,j,k in zip(fns, timelines, [0,1,2,3,0,1,2,3,4,5,6,7])]

do_resample = False
ds_factor = 2;
save_raw_tiff = False
save_trial_average_tiff = False
n_trials = 20;

corr_list = []
corr_im_list = []
trials_list = []

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
    if save_raw_tiff:
        scio.export_tiffs(data_raw, exportDir + sub_fn[:-4]+'_RAW.tiff', dims = {'x':1, 'y':2, 't':0})

    if do_resample:
        data_raw = scana.bin_2d(data_raw, ds_factor, dims = {'x':1, 'y':2, 't':0});
        data_raw = np.transpose(data_raw, (0, 2,1))

    # # Load timeline
    if ii in [0, 5]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()

    # # Compute the parsed version of the timestamps
    stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.1, interval_between_experiments = 3,  min_number_exp = 40); # Separate the frames for each one...
    
    # Recompute at appropriate times
    stim_ts = stim_list[exp_number];
    fus_ts = fus_list[exp_number];

    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: 30]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})

    # Bottom right discarding of invalid frames
    # do_discard = True
    # if do_discard:
    #     br = data_resample_at_stim[:, -4:,  -4:].mean(axis = -1).mean(axis = -1)
    #     invalid_frames = np.where(br > np.mean(br)+2*np.std(br))[0]
    #     print('Discarding %d frames' % len(invalid_frames))
    #     data_resample_at_stim[invalid_frames, : , :] = np.nan

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
    z_cutoff = 0
    #corr_cutoff = 0
    for aa in range(3):
        for xx in range(nX):
            for yy in range(nY):
                corr_maps[aa, yy, xx] = scnpix.nan_pearson(data_resample_at_stim[:, yy, xx], stim[aa])
        plt.subplot(1,4,aa+1)
        
        # Zscore and median filter
        corr_maps[aa, :, :] = (corr_maps[aa, :, :] - corr_maps[aa, :, :].mean())/corr_maps[aa, :, :].std()
        #corr_maps[aa, :, :] = gaussian_filter(corr_maps[aa, :, :], 1.5)
        plt.imshow(corr_maps[aa, :, :], vmin = z_cutoff, cmap = 'afmhot')

        #plt.imshow(corr_maps[aa, :, :], vmin = corr_cutoff, cmap = 'afmhot')

    corr_maps_im = np.transpose(corr_maps, [1,2,0]).copy()
    corr_maps_im[corr_maps_im < z_cutoff] = 0;
    corr_maps_im = np.uint8(255*corr_maps_im/np.max(corr_maps_im))
    plt.subplot(1,4,4); plt.imshow(corr_maps_im)
    plt.suptitle(animal_fn)

    corr_im_list.append(corr_maps_im)
    corr_list.append(corr_maps)

    # # # # Do the ICA on the raw images
    # ics = scana.perform_ica(data_resample_at_stim, num_comps = 15, dims = {'x':2, 'y':1, 't':0})
    # ics_reshape = ics.reshape([15, nY*nX])
    # data_reshape = data_resample_at_stim.reshape([600, nY*nX])
    # plt.figure(figsize = [20, 8])
    # for i in range(15):
    #     plt.subplot(6,5, i+1); plt.imshow(ics[i, :, :], cmap = 'binary')
    #     plt.subplot(6,5, i+16);
    #     plt.plot(np.dot(ics_reshape[i, :], data_reshape.T))
    # plt.suptitle('ICA')

    # # Compute the trial mean


    [t, y, x] = data_resample_at_stim.shape
    trials = data_resample_at_stim.reshape([n_trials, int(t/n_trials), y, x])
    trials_export = np.maximum(np.nanmean(trials, axis = 0), 0)
    trials_export = trials_export/np.max(trials_export)

    trials_list.append(trials_export)

    if save_trial_average_tiff:
        scio.export_tiffs(trials_export, exportDir + sub_fn[:-4]+'_trial_average.tiff', dims = {'x':2, 'y':1, 't':0})

    # The old phase way....
    #plt.figure()
    #reload(scana)
    #scana.compute_fft(np.nanmean(trials, axis = 0) , dims = {'x':2, 'y':1, 't':0}, doPlot = True, mask_plot = False)
    plt.savefig(exportDir + sub_fn[:-4]+'_trial_average.pdf')

# Collapsing results across experiments

plt.figure()
tmp = np.stack(corr_im_list, axis = -1)
tmp = np.transpose(tmp[14:20, :, :, :].mean(axis = 0), [2, 0, 1])
plt.figure
plt.imshow(tmp,aspect = 10)
