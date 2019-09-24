
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
fns = sorted(glob.glob('/data/fUS_project/data/data_sep03/RT*.mat')); # Only first 7 are the retinotopu
fns = fns[:3]
timelines = len(fns)*['/data/fUS_project/data/data_sep03/timeline_09-03-2019_12-09.mat']

n_fus = range(len(fns))
n_stim = range(len(fns))

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10;
fus_rate =  2; # THIS EXPEIMENT WAS 2 HZ

'''
# # Fix experiment 11 to remove an extra one...
Loading mat data from /data/fUS_project/data/data_sep03/timeline_09-03-2019_12-09.mat
Finished Loading.
Number of frames stim: 799 // fus: 1699 // 0
Number of frames stim: 800 // fus: 1700 // 1
Number of frames stim: 800 // fus: 1700 // 2
Number of frames stim: 6900 // fus: 550 // 3
Number of frames stim: 6900 // fus: 550 // 4
Number of frames stim: 6900 // fus: 550 // 5
# Necessary to skip some steps
ii = 0
animal_fn, timeline_fn, exp_number, exp_number_fus, exp_number_stim = all_exps[ii]
outDir, sub_fn = os.path.split(animal_fn)
exportDir = outDir + '/extras_object_scram/'
'''

from scipy import signal
def filter_3d_series(data_in, wh = 0.1):
    # wh defined the cutoff of the butter filer, data should be of the form nT x XY/YX
    b, a = signal.butter(2, wh, 'low', analog = False) #first parameter is signal order and the second one refers to limit wh*fs
    nT, nX, nY = data_in.shape
    data_out = np.zeros_like(data_in)
    for ii in range(nX):
        for jj in range(nY):
            data_out[:, ii, jj] = signal.filtfilt(b, a, data_in[:, ii, jj])
    return data_out


#%%
all_exp_dicts = []

# Export tiffs of all experiments
for ii in range(3):
    animal_fn, timeline_fn, exp_number, exp_number_fus, exp_number_stim = all_exps[ii]
    outDir, sub_fn = os.path.split(animal_fn)
    exportDir = outDir + '/extras_object_scram/'

    if not os.path.exists(exportDir):
        os.mkdir(exportDir)

    # Load data
    m = scio.matloader(); # Instantiate mat object
    m.loadmat_h5(animal_fn); #m.summary()
    data_raw = m.data['Dop'].copy(); # Take the square root...

    # Artifact removed version of data
    data_fix = scfilters.remove_movement_artifact_from_raw_and_condition(data_raw)
    data_fix = filter_3d_series(data_fix, wh = 0.1)
    data_fix_filt = filter_3d_series(data_fix, wh = 0.1)

    # # # Load timeline only on the first experiment
    if ii in [0]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()
        stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 4,  min_number_exp = 100); # Separate the frames for each one...
        fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 10,  min_number_exp = 100); # Separate the frames for each one...
        for tmp_ss, tmp_ff in zip(stim_list, fus_list):
            print('Number of frames stim: %d // fus: %d' % (len(tmp_ss), len(tmp_ff)))

    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus]; # Chop off first
    if ii in [1,2]: # Chop off last
        fus_ts = fus_ts[:-1]

    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    if len(stim_ts) == 800:
        newx = stim_ts
    else:
        newx = np.linspace(stim_ts[0], stim_ts[-1], 800)
    
    #newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    
    # Now do resample from the artifact cleaned version
    data_resample_at_stim = scana.resample_xyt(data_fix, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})
    
    # Onnly do this for the full field experiments
    # # Remeber there are 30 sec gray at the beginning // Cut them out early
    # print('Chopping off first 30 seconds of gray')
    # data_resample_at_stim = data_resample_at_stim[60:, :, :]

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
    curr_dict['data_raw_fix'] = data_fix_filt
    curr_dict['data_raw_fix_filt'] = data_fix_filt

    curr_dict['data_raw_medfilt'] = median_filter(data_fix, size = [5,5,5])
    curr_dict['data_resample_at_stim'] = data_resample_at_stim
    
    curr_dict['trials_all'] = trials_all

    all_exp_dicts.append(curr_dict)

np.save(exportDir + 'data_processed_objects_scram_sep03.npy', all_exp_dicts)

#%%
# # Export the cumulative tiff for trial averaged

# Size of exported image
#all_exp_dicts = np.load(exportDir + 'data_processed_fullfield_aug29.npy', allow_pickle = True)

[nT, nY, nX] = all_exp_dicts[0]['data_resample_at_stim'].shape
do_save = True

nWide = 3;

n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 1*nY, nWide*nX])
all_trials_raw_fix = np.zeros([n_fus_frames, 1*nY, nWide*nX])
all_trials_raw_filt = np.zeros([n_fus_frames, 1*nY, nWide*nX])

all_trials_raw_medfilt = np.zeros([n_fus_frames, 1*nY, nWide*nX])

nF_stim = nT;
all_trials_resampled_at_stim = np.zeros([nF_stim, 1*nY, nWide*nX])

for i in range(3):
    a, b = np.unravel_index(i, [1, nWide])
    exp_dict = all_exp_dicts[i]

    all_trials_raw_fix[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw_fix'], [0,2,1])
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                     = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_filt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                = np.transpose(exp_dict['data_raw_fix_filt'], [0,2,1])

    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]             = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]       = exp_dict['data_resample_at_stim']

# Export raw
if do_save:
    scio.export_tiffs(all_trials_resampled_at_stim, exportDir + 'OBJ_SCRAM_cumulative_trials_resample_stim.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw, exportDir + 'OBJ_SCRAM_cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_filt, exportDir + 'OBJ_SCRAM_cumulative_trials_RAW_fixed_and_ButterFilt.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_fix, exportDir + 'OBJ_SCRAM_cumulative_trials_RAW_linear_interp_of_artifacts.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_medfilt, exportDir + 'OBJ_SCRAM_cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})

# # Make a DFF from median filter
all_trials_medfilt_dff  = all_trials_raw_medfilt.copy()
f0                      = np.median(all_trials_medfilt_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(n_fus_frames)]), [2,0,1])
all_trials_medfilt_dff  = (all_trials_medfilt_dff - f0)/f0

if do_save:
    scio.export_tiffs(all_trials_medfilt_dff, exportDir + 'OBJ_SCRAM_RAW_medfilt_5_5_5_convert_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

# # Do the same for the resampled at stim_trials with medfilt
trials_dff              = all_trials_resampled_at_stim.copy()
f0                      = np.median(trials_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(trials_dff.shape[0])]), [2,0,1])
trials_dff              = (trials_dff - f0)/f0
#trials_dff              = median_filter(trials_dff, size = [6,5,5])
if do_save:
    scio.export_tiffs(trials_dff, exportDir + 'OBJ_SCRAM_trial_synchronized_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})


trial_average = trials_dff.reshape([20, 40, 52, nWide*nX]).mean(axis=0)
combined = trial_average

if do_save:
    np.save(exportDir + 'combined_single_trial.npy', combined)
    scio.export_tiffs(combined, exportDir + 'TRIAL_averages_all.tiff', dims = {'x':2, 'y':1, 't':0})

#%%

trials_obj = np.load('/data/fUS_project/data/data_sep03/extras_object_scram/combined_single_trial.npy')
trials_ff = np.load('/data/fUS_project/data/data_sep03/extras_fullfield/combined_single_trial.npy')

plt.figure(figsize = [20, 10])
plt.subplot(3,1,1)
activation_map_obj = trials_obj[10:20, :, :].max(axis = 0) - trials_obj[:5, :, :].mean(axis = 0)
plt.imshow(activation_map_obj, vmin = 0, vmax = 0.5);  plt.colorbar(); plt.title('objects')
plt.subplot(3,1,2)
activation_map_scram = trials_obj[20:30, :, :].max(axis = 0) - trials_obj[:5, :, :].mean(axis = 0)
plt.imshow(activation_map_scram, vmin = 0, vmax = 0.5); plt.colorbar(); plt.title('scrambled')

plt.subplot(3,1,3)
activation_map_ff = trials_ff[15:25, :, :].max(axis = 0) - trials_ff[:5, :, :].mean(axis = 0)
plt.imshow(activation_map_ff, vmin = 0, vmax = 0.5); plt.colorbar(); plt.title('CM noise')

plt.suptitle('DFF max - baseline, for each of 3 conditions')

#%%

plt.figure(figsize = [20, 4])
plt.imshow(2*np.transpose(np.stack([activation_map_obj, activation_map_scram, np.zeros_like(activation_map_ff)]), [1,2,0]))

plt.figure(figsize = [20, 4])
plt.imshow(2*np.transpose(np.stack([activation_map_obj, activation_map_ff, np.zeros_like(activation_map_ff)]), [1,2,0]))


#%%
