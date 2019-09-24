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
#%% Load in the moviedata
m = scio.matloader(); # Instantiate mat object
mov_fn = '/data/linuxData/MASTER_STIMULUS_FOLDER/STIMULUS_GENERATION_HERE/Francisco/fUS_combined_stim/May23_2019_combined_retinotopy_3point_TS.mat'
m.loadmat_h5(mov_fn); #m.summary()
moviedata = np.transpose(m.data['moviedata'].copy(), [2, 1, 0])

# Parse into example single trial movies
movAz2      = moviedata[:, :, :600]
movAz2      = moviedata[:, :, 600*6:600*7]
movEv1      = moviedata[:, :, 600*12:600*13]
movEv2      = moviedata[:, :, 600*18:600*19]
movSpots    = moviedata[:, :, 144700:144700+900]


#%%

fns = sorted(glob.glob('/data/fUS_project/data/data_may24/RT*.mat'))
timelines = len(fns)*['/data/fUS_project/data/data_may24/timeline_05-24-2019_12-04.mat']
n_fus = range(len(fns))
n_stim = range(len(fns))

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10;
fus_rate =  2;

corr_list = []
corr_list_zscore = []
corr_im_list = []
trials_list = []; trials_list_dff = []
data_raw_list = []
data_spots_list = []
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
    data_raw = m.data['Dop'].copy(); # Take the square root...
    data_raw_list.append(data_raw)

    # # # Load timeline only on the first experiment
    if ii in [0]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()
        stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
        fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 3,  min_number_exp = 700); # Separate the frames for each one...
    
    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus];

    if ii == 2: # Error checking
        fus_ts = fus_ts[1:]

    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})

    ####################################################
    ### PARSE IT INTO THE RETINOTOPY AND THE OTHER STIM
    ####################################################
    # The retin will be the first 1920 (2*24*20) frames
    # The spots will be the last 600 (2*30*10)
    data_retin = data_resample_at_stim[:960, :, :]
    data_spots = data_resample_at_stim[-600:, :, :]


    # # Top right discarding of invalid frames
    # do_discard = False
    # if do_discard:
    #     br = data_resample_at_stim[:, :,  -20:].mean(axis = -1).mean(axis = -1)
    #     invalid_frames = np.where(br > np.mean(br)+2*np.std(br))[0]
    #     #invalid_frames = np.where(br > np.mean(br))[0]
        
    #     print('Discarding %d frames' % len(invalid_frames))
    #     data_resample_at_stim[invalid_frames, : , :] = np.nan

    #########################################################
    ########################################################
    #### 3 SPOTS
    ##########################################################
    ############################################################

    # Stimuli for each experiment for cross correlation
    stim = np.zeros([3, 600])
    for i in range(n_trials):
        for k in range(3):
            stim[k, i*60 + 20*k : i*60 + 20*k + 10] = 1;
    #plt.plot(stim.T)

    # Compute the correlation
    plt.figure(figsize = [20, 3])
    [nT, nY, nX] = data_spots.shape

    # Make dff
    f0 = np.nanmean(data_spots, axis = 0)
    f0 = np.transpose(np.dstack([f0 for i in range(nT)]), [2,0,1])
    data_dff = (data_spots - f0)/f0

    corr_maps = np.zeros([3, nY, nX])
    corr_maps_zscore = np.zeros([3, nY, nX])
    
    for aa in range(3):
        for xx in range(nX):
            for yy in range(nY):
                corr_maps[aa, yy, xx] = scnpix.nan_pearson(data_spots[:, yy, xx], stim[aa])
        plt.subplot(1,4,aa+1)
        plt.imshow(corr_maps[aa, :, :], vmin = 0, vmax = 0.3, cmap = 'afmhot')

    corr_list.append(corr_maps)

    # # Compute the trial mean
    [t, y, x] = data_spots.shape
    trials = data_spots.reshape([n_trials, int(t/n_trials), y, x])
    trials_dff = data_dff.reshape([n_trials, int(t/n_trials), y, x])
    
    trials_export = np.maximum(np.nanmean(trials, axis = 0), 0)
    trials_export = trials_export/np.max(trials_export)
    trials_dff = np.maximum(np.nanmean(trials_dff, axis = 0), 0)
    trials_dff = trials_dff/np.max(trials_dff)
    
    trials_list.append(trials_export)
    trials_list_dff.append(trials_dff)

    data_spots_list.append(data_spots)

    plt.savefig(exportDir + sub_fn[:-4]+'_trial_average_spots.pdf')

# #%% Now do all of the plots again

# from scipy.signal import medfilt

# plt.figure(figsize = [25, 30])
# z_cutoff = 0.05
# for i in range(len(all_exps)):
#     tmp_im = np.zeros_like(corr_list[0])
#     for aa in range(3):
#         plt.subplot(11, 4, 4*i+aa+1)

#         tmp_im[aa, :, :] = corr_list[i][aa, :, :]/np.max(corr_list[i][aa, :, :])
#         plt.imshow(corr_list[i][aa, :, :], vmin = 0.05, cmap = 'afmhot'); plt.axis('off'); plt.colorbar()
        
#     plt.subplot(11, 4, 4*i+4)
#     color_tmp = np.transpose(tmp_im, [1,2,0]).copy()
#     color_tmp[color_tmp < 0.6] = 0;
#     color_tmp = np.uint8(255*color_tmp/np.max(color_tmp))
#     plt.imshow(color_tmp); plt.axis('off')

# plt.savefig(exportDir + 'correlation_maps.pdf')

# # SAME AS ABOVE/EXCEPT SMOOTHED
# #%%
# plt.figure(figsize = [25, 30])
# # Only do the middle ones, excluding
# for i in range(2, len(all_exps)-2):
#     smoothed = np.stack(corr_list[i-2:i+3], axis = -1).mean(axis = -1)
#     color_smoothed = np.copy(smoothed)
#     for aa in range(3):
#         plt.subplot(11, 4, 4*i+aa+1)
#         plt.imshow(smoothed[aa, :, :], vmin=  0, cmap = 'afmhot'); plt.axis('off'); plt.colorbar()
#         color_smoothed[aa, :, :] = color_smoothed[aa, :, :]/np.max(color_smoothed[aa, :, :]); # Normalize to max intensity for each channel
#     plt.subplot(11, 4, 4*i+4)
#     # Add code for cutting off low vals
#     #color_smoothed
#     color_tmp = np.transpose(color_smoothed, [1,2,0]).copy()
#     color_tmp[color_tmp<0.5] = 0;
#     plt.imshow(np.uint8(255*color_tmp)); plt.axis('off')

# #plt.savefig(exportDir + 'correlation_maps_smoothed_across_planes.pdf')



#%%
# # Export the cumulative tiff for trial averaged

nT, nY, nX = data_spots.shape
all_trials = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
all_trials_dff = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
n_fus_frames = data_raw_list[0].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 4*nY, 4*nX])

nF_stim = data_spots.shape[0]
all_trials_resampled_at_stim = np.zeros([nF_stim, 4*nY, 4*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [4, 4])
    all_trials[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = trials_list[i]
    all_trials_dff[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = trials_list_dff[i]
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = np.transpose(data_raw_list[i], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = data_spots_list[i]

scio.export_tiffs(all_trials, exportDir + 'cumulative_trials_spots.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_spots_dff.tiff', dims = {'x':2, 'y':1, 't':0})

# Export raw
scio.export_tiffs(all_trials_raw, exportDir + 'cumulative_trials_RAW_BOTH_EXP_sqrt.tiff', dims = {'x':2, 'y':1, 't':0})


#%%
# from scipy.misc import imresize
# from PIL import Image

# # Load in the movie then add in the lower right corner...
# # Reshape the moviedata into the bottom left
# nF_trial = int(30*fus_rate); # Number of frames per trial
# scale_trials = np.max(all_trials)
# scale_dff = np.max(all_trials_dff)
# for xx in range(nF_trial): # For each frame, draw the movie
#     a, b = np.unravel_index(15, [4, 4])
#     movie_sub = np.array(Image.fromarray(moviedata[int(float(xx/nF_trial)*900), :, :].T).resize([nX, nY]))
#     movie_sub = movie_sub - np.min(movie_sub); movie_sub = movie_sub/np.max(movie_sub);

#     all_trials[xx, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = scale_trials*movie_sub
#     all_trials_dff[xx, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = scale_dff*movie_sub




# if not do_resample:
#     scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_spots_dff.tiff', dims = {'x':2, 'y':1, 't':0})

#     scio.export_tiffs(all_trials_resampled_at_stim, exportDir + 'cumulative_trials_raw_stim_align.tiff', dims = {'x':2, 'y':1, 't':0})


################################################

#%% Do the fourier
################################################
################################################
################################################
################################################
################################################
################################################

fig_size = [25, 9]

#NON TRIAL BASED
data = data_retin[:240, :, :]

idxFFT = 13

out = np.fft.fft(data, axis = 0); # take along time
plt.figure(figsize = fig_size)
plt.imshow(np.abs(out[idxFFT, :, :]), vmax = 5, cmap = 'Greys'); plt.colorbar()
plt.figure(figsize = fig_size)
plt.imshow(np.angle(out[idxFFT, :, :]), cmap = 'RdBu'); plt.colorbar()

#TRIAL BASED
#data = all_trials_resampled_at_stim;
data = all_trials;

idxFFT = 1

out = np.fft.fft(data, axis = 0); # take along time
plt.figure(figsize = fig_size)
plt.imshow(np.abs(out[idxFFT, :, :]), vmax = 1, cmap = 'Greys'); plt.colorbar()

plt.figure(figsize = fig_size)
plt.imshow(np.angle(out[idxFFT, :, :]),cmap = 'RdBu'); plt.colorbar()

#TRIAL DFF
data = all_trials_dff;

idxFFT = 1

out = np.fft.fft(data, axis = 0); # take along time
plt.figure(figsize = fig_size)
plt.imshow(np.abs(out[idxFFT, :, :]), vmax = 3, cmap = 'Greys'); plt.colorbar()

plt.figure(figsize = fig_size)
plt.imshow(np.angle(out[idxFFT, :, :]), cmap = 'RdBu'); plt.colorbar()



#plt.subplot(1,2,2); plt.imshow(np.angle(out[idxFFT, :, :]), cmap = 'seismic')


plt.figure(figsize = [30, 30])
for i in range(30):
    plt.subplot(6,5,i+1)
    plt.imshow(np.abs(out[i, :, :]), cmap = 'binary')
    #plt.colorbar()



#%% Make a cumulative RGB image where RGB relative color is max DFF in L/B/R and Hue is the diff
# rgb_vals = [all_trials_dff[:20, :, :].mean(axis = 0), all_trials_dff[20:40, :, :].mean(axis = 0), all_trials_dff[40:, :, :].mean(axis = 0)]
# rgb_mat = np.stack(rgb_vals, axis = 0) # Create a 3 x nY x nX matrix

# hue_val = rgb_mat.max(axis = 0) - rgb_mat.min(axis = 0)
# hue_val = np.minimum(hue_val/np.percentile(hue_val, 99.5), 1); # Truncate top 5% of values at at 1
# hue_val = np.repeat(np.expand_dims(hue_val, axis = 0), 3, axis = 0)

# # Convert the rgbmat into a 0 min version in RGB
# mins = rgb_mat.min(axis = 0)
# for ii in range(3):
#     rgb_mat[ii, :, :] = rgb_mat[ii, :, :] - mins;
# # And divide by sum such that if something is towering above two then it will get a high pref
# sums = rgb_mat.sum(axis = 0)
# for ii in range(3):
#     rgb_mat[ii, :, :] = rgb_mat[ii, :, :]/sums;
# plt.figure(figsize = [20, 10])
# plt.imshow(np.transpose(np.uint8(255*hue_val*rgb_mat), [1,2,0]))
