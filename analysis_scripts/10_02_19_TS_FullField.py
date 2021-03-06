
#%%

# Analysis example
import glob
import os
import sys
from importlib import reload

sys.path.append('/data/git_repositories_py/SCRAPC/')
sys.path.append('/data/git_repositories_py/fUS_pytools/')
# sys.path.append('/Users/fluongo/Documents/CALTECH/code/Sep23_2019_fus_during_recovery/scrpy')
# sys.path.append('/Users/fluongo/Documents/CALTECH/code/Sep23_2019_fus_during_recovery/fUS_pytools')


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
data_dir = '/data/fUS_project/data/data_Oct02/'
fns = sorted(glob.glob(data_dir + '/RT*.mat'))[9:]; # Only first 9
timelines = len(fns)*[data_dir + '/timeline_10-02-2019_11-24.mat']
n_fus = range(9, 9+len(fns))
n_stim = range(9, 9+len(fns))

all_exps = [[i, j,k, l, m] for i,j,k ,l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10;
fus_rate =  2; # THIS EXPEIMENT WAS 2 HZ

n_exp = len(fns)
#%%
all_exp_dicts = []

# Export tiffs of all experiments
for ii in range(n_exp):
    animal_fn, timeline_fn, exp_number, exp_number_fus, exp_number_stim = all_exps[ii]
    outDir, sub_fn = os.path.split(animal_fn)
    exportDir = outDir + '/extras_fullfield/'

    if not os.path.exists(exportDir):
        os.mkdir(exportDir)

    # Load data
    m = scio.matloader(); # Instantiate mat object
    m.loadmat_h5(animal_fn); #m.summary()
    data_raw = m.data['Dop'].copy(); # Take the square root...

    # Artifact removed version of data
    data_fix = scfilters.remove_movement_artifact_from_raw_and_condition(data_raw, thresh = 1)

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
    fus_ts = fus_list[exp_number_fus][1:]; # Chop off first
    if ii == 0: # add one stim frame
        stim_ts = np.append(stim_ts, stim_ts[-1])
    print('stim_ts has %d frames' % len(stim_ts))
    
    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    #newx=stim_ts
    newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
 
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
    curr_dict['data_raw_fix'] = data_fix
    curr_dict['data_raw_medfilt'] = median_filter(data_fix, size = [5,5,5])
    curr_dict['data_resample_at_stim'] = data_resample_at_stim
    curr_dict['trials_all'] = trials_all

    all_exp_dicts.append(curr_dict)

np.save(exportDir + 'data_processed_fullfield.npy', all_exp_dicts)

#%%
# # Export the cumulative tiff for trial averaged
stim_name = 'FULLFIELD'
[nT, nY, nX] = all_exp_dicts[0]['data_resample_at_stim'].shape
do_save = True

# Size of the output of planes
nWide = 3;
nHigh = 3;

n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, nHigh*nY, nWide*nX])
all_trials_raw_fix = np.zeros([n_fus_frames, nHigh*nY, nWide*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, nHigh*nY, nWide*nX])

nF_stim = nT
all_trials_resampled_at_stim = np.zeros([nF_stim, nHigh*nY, nWide*nX])

for i in range(n_exp):
    a, b = np.unravel_index(i, [nHigh, nWide])
    exp_dict = all_exp_dicts[i]

    all_trials_raw_fix[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw_fix'], [0,2,1])
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                     = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]             = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]       = exp_dict['data_resample_at_stim']

# Export raw
if do_save:
    scio.export_tiffs(all_trials_resampled_at_stim, exportDir + stim_name + '_cumulative_trials_resample_stim.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw, exportDir + stim_name + '_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_fix, exportDir + stim_name + '_RAW_linear_interp_of_artifacts.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_medfilt, exportDir + stim_name + '_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})

# # Make a DFF from median filter
all_trials_medfilt_dff  = all_trials_raw_medfilt.copy()
f0                      = np.median(all_trials_medfilt_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(n_fus_frames)]), [2,0,1])
all_trials_medfilt_dff  = (all_trials_medfilt_dff - f0)/f0

if do_save:
    scio.export_tiffs(all_trials_medfilt_dff, exportDir + stim_name + '_RAW_medfilt_5_5_5_convert_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

# # Do the same for the resampled at stim_trials with medfilt
trials_dff              = all_trials_resampled_at_stim.copy()
f0                      = np.median(trials_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(trials_dff.shape[0])]), [2,0,1])
trials_dff              = (trials_dff - f0)/f0
#trials_dff              = median_filter(trials_dff, size = [6,5,5])
if do_save:
    scio.export_tiffs(trials_dff, exportDir + stim_name + '_trial_synchronized_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

trials_tmp = trials_dff[60:, :, :]# Chop off 15 sec static
trial_average = trials_tmp.reshape([10, 40, nHigh*nY, nWide*nX]).mean(axis=0)
combined = trial_average

if do_save:
    np.save(exportDir + 'combined_single_trial.npy', combined)
    scio.export_tiffs(combined, exportDir + 'TRIAL_averages_all.tiff', dims = {'x':2, 'y':1, 't':0})

#%%

# Compute activation maps from the data at various deltaX e.g. temporal hemodynamic offsets

tmp = trials_dff[60:, :, :]
tmp_re = tmp.reshape([10, 40, nHigh*nY, nWide*nX]).mean(axis=0)

m1  = tmp_re[10:30, :, :].mean(axis = 0)
m2  = tmp_re[:10, :, :].mean(axis = 0)
dm  = m1-m2; m_val = np.percentile(np.abs(dm), 99)
plt.figure(figsize = [10, 10])

plt.imshow(dm, cmap = 'bwr', vmin = -m_val, vmax = m_val)

#plt.imshow(dm[:52, 128*4:128*5], cmap = 'PRGn', vmin = -m_val, vmax = m_val)

#%% P VALUES

from scipy import stats

tmp = trials_dff[60:, :, :]
tmp_re = tmp.reshape([10, 40, nHigh*nY, nWide*nX])

p_vals = np.zeros([nHigh*nY, nWide*nX])
t_vals = np.zeros([nHigh*nY, nWide*nX])

for ii in range(nHigh*nY):
    if ii % 50 == 0:
        print('On iteration %d out of %d' % (ii, nHigh*nY))
    for jj in range(nWide*nX):
        [t_tmp, p_tmp] = stats.ttest_ind(tmp_re[:, 15:20, ii, jj].mean(axis=1), tmp_re[:, 5:10, ii, jj].mean(axis=1))
        p_vals[ii, jj] = p_tmp
        t_vals[ii, jj] = t_tmp
        
p_sign = t_vals.copy()
p_sign[p_sign>0] =1
p_sign[p_sign<0] =-1


plt.figure(figsize = [20, 10])
plt.imshow(np.log10(p_vals), vmin = -6); plt.colorbar()
plt.suptitle('Log p _value with sign ')



#%%
