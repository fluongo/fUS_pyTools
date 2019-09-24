
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

def remove_movement_artifact_from_raw_and_condition(data):
    # Removes the movement artifact from the data that comes in the form of nT, nX, nY
    sub_data = data_raw[:, :40, :40].mean(axis = -1).mean(axis=-1)
    # Exclude all Values greater than 3sd over the median
    pseudo_z = (sub_data - np.median(sub_data))/np.median(sub_data)
    idx_remove = np.argwhere(pseudo_z>2)
    n_timepoints = data.shape[0]
    # Just do a linear interpolation beyween the last two valid points
    data_fix = np.copy(data)
    for xx in idx_remove:
        if xx == 0 or xx == n_timepoints:
            data_fix[xx, :, :] = np.median(data_fix, axis = 0)
        try:
            prev_i = np.max(np.setdiff1d(np.arange(xx),idx_remove)) # Largest value less than the current number but not in list
            post_i = np.min(np.setdiff1d(np.arange(xx+1, n_timepoints),idx_remove)) # Minimum value less than the current number but not in list
            data_fix[xx, :, :] = ( data_fix[prev_i, :, :] + data_fix[post_i, :, :] )/ 2.
        except:
            data_fix[xx, :, :] = np.median(data_fix, axis = 0)

    return data_fix

data_fix = remove_movement_artifact_from_raw_and_condition(data_raw)
plt.figure(figsize = [20, 3]);
plt.plot(data_fix[:, :40, :40].mean(axis = -1).mean(axis=-1))
plt.figure(figsize = [20, 3]);
plt.plot(data_raw[:, :40, :40].mean(axis = -1).mean(axis=-1))


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
    data_fix = remove_movement_artifact_from_raw_and_condition(data_raw)


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
    
    # Now do resample from the artifact cleaned version
    #data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})
    data_resample_at_stim = scana.resample_xyt(data_fix, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})
    
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
    trials_all['ev_DU'] = data_resample_at_stim[900:, :, :]
    

    # Compute the fus_stim_time for each frame
    fus_ts_stim = np.zeros_like(fus_ts)
    for ww in range(data_raw.shape[0]):
        fus_ts_stim[ww] = np.argmin(np.abs(stim_ts - fus_ts[ww]));

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


#%%
# # Export the cumulative tiff for trial averaged
#all_trials = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
#all_trials_dff = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 2*nY, 4*nX])
all_trials_raw_fix = np.zeros([n_fus_frames, 2*nY, 4*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, 2*nY, 4*nX])

nF_stim = nT;
all_trials_resampled_at_stim = np.zeros([nF_stim, 2*nY, 4*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [2, 4])
    exp_dict = all_exp_dicts[i]

    all_trials_raw_fix[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw_fix'], [0,2,1])
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]         = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]   = exp_dict['data_resample_at_stim']

    # TODO Make a version that does the trial averages....

#scio.export_tiffs(all_trials, exportDir + 'cumulative_trials_spots.tiff', dims = {'x':2, 'y':1, 't':0})
#scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_spots_dff.tiff', dims = {'x':2, 'y':1, 't':0})

# Export raw
scio.export_tiffs(all_trials_resampled_at_stim, exportDir + 'RETINOTOPY_cumulative_trials_resample_stim.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_raw, exportDir + 'RETINOTOPY_cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_raw_fix, exportDir + 'RETINOTOPY_cumulative_trials_RAW_linear_interp_of_artifacts.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_raw_medfilt, exportDir + 'RETINOTOPY_cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})

# Make a DFF from median filter
all_trials_medfilt_dff  = all_trials_raw_medfilt.copy()
f0                      = np.median(all_trials_medfilt_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(n_fus_frames)]), [2,0,1])
all_trials_medfilt_dff  = (all_trials_medfilt_dff - f0)/f0
scio.export_tiffs(all_trials_medfilt_dff, exportDir + 'RETINOTOPY_RAW_medfilt_5_5_5_convert_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

# Do the same for the resampled at stim_trials with medfilt
trials_dff              = all_trials_resampled_at_stim.copy()
f0                      = np.median(trials_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(trials_dff.shape[0])]), [2,0,1])
trials_dff              = (trials_dff - f0)/f0
scio.export_tiffs(trials_dff, exportDir + 'RETINOTOPY_trial_synchronized_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

trial_averageLR = trials_dff[:300, :, :].reshape([5, 60, 104, 512]).mean(axis=0)
trial_averageRL = trials_dff[300:600, :, :].reshape([5, 60, 104, 512]).mean(axis=0)
scio.export_tiffs(trial_averageLR, exportDir + 'TRIAL_average_az_LR_all.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(trial_averageRL, exportDir + 'TRIAL_average_az_RL_all.tiff', dims = {'x':2, 'y':1, 't':0})

trial_averageUD = trials_dff[600:900, :, :].reshape([5, 60, 104, 512]).mean(axis=0)
trial_averageDU = trials_dff[900:, :, :].reshape([5, 60, 104, 512]).mean(axis=0)
scio.export_tiffs(trial_averageUD, exportDir + 'TRIAL_average_ev_UD_all.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(trial_averageDU, exportDir + 'TRIAL_average_ev_DU_all.tiff', dims = {'x':2, 'y':1, 't':0})


#%% All plane fourier

def get_and_plot_phase_maps(data1, data2, idx_fft):
    # For making things tidier
    out = np.fft.fft(data1, axis = 0); # take along time
    plt.figure(figsize = [18, 4]); 
    phaseRL = np.angle(out[idx_fft, :, :] )
    plt.imshow(phaseRL, cmap = 'hsv'); plt.colorbar(); plt.title('RL')

    out = np.fft.fft(data2, axis = 0); # take along time
    plt.figure(figsize = [18, 4]);
    phaseLR = np.angle(out[idx_fft, :, :])
    plt.imshow(phaseLR, cmap = 'hsv'); plt.colorbar(); plt.title('LR')

    plt.figure(figsize = [18, 4]);
    plt.imshow((phaseRL-phaseLR)/2, cmap = 'hsv'); plt.colorbar(); 
    
    return phaseRL, phaseLR, (phaseRL-phaseLR)/2

phaseRL, phaseLR, combined = get_and_plot_phase_maps(trials_dff[:300, :, :], trials_dff[300:600, :, :], 5)
plt.title('FULL DATASET')

phaseRL, phaseLR, combined = get_and_plot_phase_maps(trials_dff[600:900, :, :], trials_dff[900:, :, :], 5)
plt.title('FULL DATASET')



#%%

# Compute a model for every pixel
import GPy
import seaborn as sns

exp_no = 6

curr_x = all_exp_dicts[exp_no]['fus_ts_stim']
idx = np.where((curr_x>600*5) & (curr_x<=600*10))
X = np.mod(curr_x[idx].reshape(-1, 1), 600)

plt.figure(figsize = [20, 20])
for cnt, i in enumerate(range(0, 110, 3)):
    plt.subplot(7,7,cnt+1)
    y = all_exp_dicts[exp_no]['data_raw_fix'][idx, i, 15]
    plt.hist2d(np.squeeze(X), np.squeeze(y), bins = [30, 10])
    plt.title(str(i))


#%% GAUSSIAN PROCESS ANALYSIS

# # Compute a model for every pixel
# import GPy
# import seaborn as sns

# x_bounds = np.arange(0, 127-15, 16) # 8
# y_bounds = np.arange(0, 51-10, 10) # 5

# # Observations
# # only one type of tral
# plt.figure(figsize = [20, 20])
# cnt = 0
# exp_no = 1

# curr_x = all_exp_dicts[exp_no]['fus_ts_stim']


# for x_b in x_bounds:
#     for y_b in y_bounds:
#         curr_x = all_exp_dicts[exp_no]['fus_ts_stim']
#         idx = np.where((curr_x>0) & (curr_x<=18000))

#         cx = 40+xx
#         cy = 17
#         X = np.mod(curr_x[idx].reshape(-1, 1), 2*60)
#         #y = all_trials_raw[idx, cy, cx]
#         y = all_exp_dicts[exp_no]['data_raw_fix'][idx, x_b:x_b+16, y_b:y_b+10].mean(axis=-1).mean(axis=-1)

#         # Condition y
#         #y = y-np.mean(y)
#         #y = y/np.max(y)

#         plt.subplot(8,5,cnt+1); cnt = cnt+1;
#         plt.scatter(np.mod(X, 60), y, s=1); plt.title('X: %d Y: %d' %  (x_b, y_b))
#         plt.ylim([np.percentile(y, 1), np.percentile(y, 90)])
#         sns.despine()


exp_no = 3

# MAKE X
curr_x = all_exp_dicts[exp_no]['fus_ts_stim']
idx = np.where((curr_x>600*5) & (curr_x<=600*10))
X = np.mod(curr_x[idx].reshape(-1, 1), 600)
# MAKE Y
y = all_exp_dicts[exp_no]['data_raw_fix'][idx, 66, 15]
y = y/np.mean(y)



# #%%
kernel = GPy.kern.RBF(input_dim=1, variance = 1., lengthscale= 1.)
m = GPy.models.GPRegression(X, y.reshape(-1,1), kernel)

# the normal way
m.optimize(messages=True)
# with restarts to get better results
m.optimize_restarts(num_restarts = 4)
m.plot(); plt.xlim([0,600])

