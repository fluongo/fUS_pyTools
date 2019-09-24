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
from scipy.ndimage import gaussian_filter, median_filter    


##################################################################
# Load the fUS data
##################################################################
#%%

# 20 experiments// 10 3 spot // 10 led
fns = sorted(glob.glob('/data/fUS_project/data/data_mar28/RT*.mat'))
timelines = 10*['/data/fUS_project/data/data_mar28/timeline_03-28-2019_11-29.mat'] + 10*['/data/fUS_project/data/data_mar28/timeline_03-28-2019_13-12.mat']
all_exps = [[i, j] for i,j in zip(fns, timelines)]

do_resample = False
ds_factor = 2;
save_raw_tiff = False
save_trial_average_tiff = False
n_trials = 10;

corr_list = []
corr_im_list = []

trials_list = []

# Export tiffs of all experiments
for ii in range(10):
    animal_fn, timeline_fn = all_exps[ii]
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
    if ii == 0:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()

    # # Compute the parsed version of the timestamps
    stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.1, interval_between_experiments = 3,  min_number_exp = 40); # Separate the frames for each one...
    
    # Recompute at appropriate times
    stim_ts = stim_list[ii];
    fus_ts = fus_list[ii];
    if ii == 9:
        fus_ts = fus_ts[2:]
        
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: 15]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})


    # Bottom right discarding of invalid frames
    do_discard = True
    if do_discard:
        br = data_resample_at_stim[:, -4:,  -4:].mean(axis = -1).mean(axis = -1)
        invalid_frames = np.where(br > np.mean(br)+2*np.std(br))[0]
        print('Discarding %d frames' % len(invalid_frames))
        data_resample_at_stim[invalid_frames, : , :] = np.nan

    # Stimuli for each experiment for cross correlation
    stim = np.zeros([3, 600])
    for i in range(n_trials):
        for k in range(3):
            stim[k, i*60 + 20*k : i*60 + 20*k + 10] = 1;
    #plt.plot(stim.T)

    # Compute the correlation
    plt.figure(figsize = [20, 3])

    [nT, nY, nX] = data_resample_at_stim.shape

    # Make dff
    f0 = np.nanmean(data_resample_at_stim, axis = 0)
    f0 = np.transpose(np.dstack([f0 for i in range(nT)]), [2,0,1])
    data_resample_at_stim = (data_resample_at_stim - f0)/f0

    corr_maps = np.zeros([3, nY, nX])
    z_cutoff = 1.6
    #corr_cutoff = 0
    for aa in range(3):
        for xx in range(nX):
            for yy in range(nY):
                corr_maps[aa, yy, xx] = scnpix.nan_pearson(data_resample_at_stim[:, yy, xx], stim[aa])
        plt.subplot(1,4,aa+1)
        
        # Zscore and median filter
        corr_maps[aa, :, :] = (corr_maps[aa, :, :] - corr_maps[aa, :, :].mean())/corr_maps[aa, :, :].std()
        corr_maps[aa, :, :] = gaussian_filter(corr_maps[aa, :, :], 1.5)
        plt.imshow(corr_maps[aa, :, :], vmin = z_cutoff, cmap = 'afmhot')

        #plt.imshow(corr_maps[aa, :, :], vmin = corr_cutoff, cmap = 'afmhot')

    corr_maps_im = np.transpose(corr_maps, [1,2,0]).copy()
    corr_maps_im[corr_maps_im < z_cutoff] = 0;
    corr_maps_im = np.uint8(255*corr_maps_im/np.max(corr_maps_im))
    plt.subplot(1,4,4); plt.imshow(corr_maps_im)
    plt.suptitle(animal_fn)

    corr_im_list.append(corr_maps_im)
    corr_list.append(corr_maps)

    # # Do the ICA on the raw images
    # ics = scana.perform_ica(data_resample_at_stim, num_comps = 15, dims = {'x':2, 'y':1, 't':0})
    # ics_reshape = ics.reshape([15, nY*nX])
    # data_reshape = data_resample_at_stim.reshape([600, nY*nX])
    # plt.figure(figsize = [20, 8])
    # for i in range(15):
    #     plt.subplot(6,5, i+1); plt.imshow(ics[i, :, :], cmap = 'binary')
    #     plt.subplot(6,5, i+16);
    #     plt.plot(np.dot(ics_reshape[i, :], data_reshape.T))
    #plt.suptitle('ICA')

    [t, y, x] = data_resample_at_stim.shape
    trials = data_resample_at_stim.reshape([n_trials, int(t/n_trials), y, x])
    trials_export = np.maximum(np.nanmean(trials, axis = 0), 0)
    trials_export = trials_export/np.max(trials_export)

    trials_list.append(trials_export)

    if save_trial_average_tiff:
        scio.export_tiffs(trials_export, exportDir + sub_fn[:-4]+'_trial_average.tiff', dims = {'x':2, 'y':1, 't':0})

#%%
# Collapsing results across experiments

tmp = np.stack(corr_im_list, axis = -1)
tmp = np.transpose(tmp[0:20, :, :, :].mean(axis = 0), [2, 0, 1])
plt.figure
plt.imshow(tmp,aspect = 10)

# Export the cumulative tiff
all_trials = np.zeros([60, 3*52, 4*128])

for i in range(10):
    a, b = np.unravel_index(i, [3, 4])
    all_trials[:, a*52:(a+1)*52, b*128:(b+1)*128] = trials_list[i]

scio.export_tiffs(all_trials, exportDir + 'cumulative_trials.tiff', dims = {'x':2, 'y':1, 't':0})



#%%
#########################################################33
#########################################################33
#########################################################33
#########################################################33
# Code for the light flashing experiment
#########################################################33
#########################################################33
#########################################################33
#########################################################33

save_raw_tiff = False
do_resample = False
ds_factor = 3;

# Analyze the light experiments
for cnt, ii in enumerate(range(10, 20)):
    animal_fn, timeline_fn = all_exps[ii]
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

    # Load timeline
    if cnt == 0:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()

    # # Compute the parsed version of the timestamps
    stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 60); # Separate the frames for each one...
    fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.1, interval_between_experiments = 3,  min_number_exp = 40); # Separate the frames for each one...
    #%%
    stim_ts = stim_list[cnt]
    fus_ts = fus_list[cnt]
    led_ts = np.zeros_like(fus_ts)

    # Find the led times
    # Remember to shift by otherwise it is correlating at end of led
    ledtimes = np.insert(stim_ts[np.where(np.diff(stim_ts)>10)[0]+1], 0, stim_ts[0])
    assert len(ledtimes) == 5
    for t in ledtimes:
        led_ts[(fus_ts>=t)*(fus_ts <= t + 5)] = 1;


    [nT, nX, nY] = data_raw.shape

    # Now correlate each one 
    corr_map = np.zeros([nX, nY])
    for xx in range(nX):
        for yy in range(nY):
            corr_map[xx, yy] = scnpix.numba_corr(data_raw[:, xx, yy], led_ts)
    plt.figure(figsize = [18, 3])
    plt.subplot(1,2,1);plt.imshow(data_raw[0, :, :].T, cmap = 'gray')
    plt.subplot(1,2,2);plt.imshow(corr_map.T, vmin = 0, vmax = np.max(corr_map), cmap = 'Purples')

#data_raw = data_raw[:, 20:100, :]; # Cut into only the brain parts

# if do_resample:
#     data = scana.bin_2d(data_raw, 2, dims = {'x':1, 'y':2, 't':0});
#     data = np.transpose(data, (0, 2,1))
# else:
#     data = data_raw;

# ##################################################################
# # Clean the timestamps
# ##################################################################

# # Loading in the timeline data
# if ii == 0: # Just load on first
#     timestamps_m = scio.matloader();
#     timestamps_m.loadmat_h5(timeline_fn); timestamps_m.summary()
#     ts = timestamps_m.data['timestamps'].copy()

# #exp = 1; # Which experiment number
# stim_ts = stim_list[exp_stim]
# fus_ts = fus_list[exp_fUS]
# # Check that the number of fUS frames matches the experiment
# offset = len(fus_ts) - data.shape[0]
# if offset !=0: # Meaning fus timestamps and fUS dont match
#     fus_ts = fus_ts[offset:] # truncate

# # Upsample the interpolation of each frame to account for stim ttl only getting sent once every 3rd frame
# #    stim_ts = fUS.updample_timestamps(stim_ts, 3)
# fus_rate = 2; # fus_rate in hz

# newx = stim_ts[1:-1: int(stim_rate/(fus_rate))]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....

# data_resample_at_stim = scana.resample_xyt(data, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})

# # Turn into DFF
# f0 = np.repeat(np.expand_dims( data_resample_at_stim.mean(axis=0), axis = 0), len(newx), axis = 0); # Makes a f0 for each timepoint
# data_resample_at_stim = (data_resample_at_stim -f0)/f0

# if is_retinotopy:
#     nF = int(data_resample_at_stim.shape[0]/2/n_trials) # Number of frames for each trial
    
#     # Concatenate across trials and then output to tiff the original as well as the others
#     trials_az = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])
#     trials_el = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])

#     for i in range(n_trials):
#         # Write each trial
#         trials_az[i, :, :, :] = data_resample_at_stim[i*nF:(i+1)*nF, :, :]
#         trials_el[i, :, :, :] = data_resample_at_stim[n_trials*nF+i*nF:n_trials*nF+(i+1)*nF, :, :]

#     ph_ev, pw_ev = scana.compute_fft(trials_el.mean(axis=0), dims = {'x':2, 'y':1, 't':0}, doPlot = False)
#     ph_az, pw_az = scana.compute_fft(trials_az.mean(axis=0), dims = {'x':2, 'y':1, 't':0}, doPlot = False)

#     # Compute a mask by cross validating the fus
#     fs = scana.compute_field_sign(ph_az, ph_ev, pw_az, pw_ev, filt_size=1); 
#     plt.title(sub_fn);

#     pw_ev_list.append(pw_ev);pw_az_list.append(pw_az) 
#     ph_ev_list.append(ph_ev);ph_az_list.append(ph_az) 
#     names_list.append(sub_fn)
    
#     if do_save:
#         add_str = '_re' if do_resample else ''             
#         plt.savefig(outDir + '/' + sub_fn[:-4] + add_str + '.pdf')

#         exportDir = outDir + '/extras/'
#         if not os.path.exists(exportDir):
#             os.mkdir(exportDir)
        
#         scio.export_tiffs(data_resample_at_stim, exportDir + sub_fn[:-4] + add_str + '.tiff', dims = {'x':2, 'y':1, 't':0})
#         scio.export_tiffs(trials_az.mean(axis = 0), exportDir + sub_fn[:-4] + add_str + '_az.tiff', dims = {'x':2, 'y':1, 't':0})
#         scio.export_tiffs(trials_el.mean(axis = 0), exportDir + sub_fn[:-4] + add_str + '_el.tiff', dims = {'x':2, 'y':1, 't':0})

# else:
#     if do_save:
#         exportDir = outDir + '/extras/'


#%% Analysis of the full exp

plt.figure(figsize = [18, 18])
all_az = np.dstack(ph_az_list)
all_az = scana.gaussian_filter_xyt(all_az, sigmas = [0, 0, 0], dims = {'x':1, 'y': 0, 't':2}) 
all_az = np.transpose(all_az, [1,2,0])
for i in range(16):
plt.subplot(4,4,i+1)
plt.imshow(all_az[:, :, i], vmin = np.min(all_az), vmax = np.max(all_az), cmap = 'hsv')

plt.figure(figsize = [18,18])
all_ev = np.dstack(ph_ev_list)
all_ev = scana.gaussian_filter_xyt(all_ev, sigmas = [0, 0, 0], dims = {'x':1, 'y': 0, 't':2}) 
all_ev = np.transpose(all_ev, [1,2,0])
for i in range(16):
plt.subplot(4,4,i+1)
plt.imshow(all_ev[:, :, i], vmin = np.min(all_ev), vmax = np.max(all_ev), cmap = 'hsv')

# Now compute field signs
plt.figure(figsize = [18,18])
for ii in range(16):
plt.subplot(4,4,ii+1)
fs = scana.compute_field_sign(all_az[:, :, ii], all_ev[:, :, ii], pw_az, pw_ev, filt_size=0.1, doPlot = False);
plt.imshow(fs, cmap = 'bwr')
