
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
#
# PINBGLEI LOCALIZER CONTRAST INVERTED
##################################################################
data_dir = '/data/fUS_project/data/data_dec13'
fns = sorted(glob.glob(data_dir + '/RT*.mat')); # NOW DO THE LAST 15
timelines = len(fns)*[data_dir + '/timeline_12-13-2019_10-47.mat']
n_fus = range(len(fns))
n_stim = range(len(fns))

all_exps = [[i, j,k, l, m] for i,j,k ,l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10;
fus_rate =  2; # THIS EXPEIMENT WAS 2 HZ


#%%

# NOTE this stimulus has 5 seconds of gray at the very end of it, so that we can just pull the last 400 and that will
# Be the stimulus frames
all_exp_dicts = []

# Export tiffs of all experiments
for ii in range(len(fns)):
    animal_fn, timeline_fn, exp_number, exp_number_fus, exp_number_stim = all_exps[ii]
    outDir, sub_fn = os.path.split(animal_fn)
    exportDir = outDir + '/extras_obj_pinglei_localizer_inverted/'

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
        stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 1,  min_number_exp = 100); # Separate the frames for each one...
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
    newx = stim_ts; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    
    # Now do resample from the artifact cleaned version
    data_resample_at_stim = scana.resample_xyt(data_fix, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})
    
    # Onnly do this for the full field experiments
    # # Remeber there are 30 sec gray at the beginning // Cut them out early
    # print('Chopping off first 30 seconds of gray')
    # data_resample_at_stim = data_resample_at_stim[60:, :, :]

    # Size if 450 x 52 x 128, with first 60 frames teh 30 sec of gray    
    [nT, nY, nX] = data_resample_at_stim.shape
    # Subsequent trials end with gray
    trials_all = data_resample_at_stim[60:, :, :]

    curr_dict = {}
    curr_dict['stim_ts'] = stim_ts
    curr_dict['fus_ts'] = fus_ts
    curr_dict['data_raw'] = data_raw
    curr_dict['data_raw_fix'] = data_fix
    curr_dict['data_raw_medfilt'] = median_filter(data_fix, size = [5,5,5])
    curr_dict['data_resample_at_stim'] = data_resample_at_stim
    curr_dict['trials_all'] = trials_all

    all_exp_dicts.append(curr_dict)

np.save(exportDir + 'data_processed_obj_PL_localizer_inverted.npy', all_exp_dicts)


#%%

# # Export the cumulative tiff for trial averaged
stim_name = 'PL_localizer'
[nT, nY, nX] = all_exp_dicts[0]['data_resample_at_stim'].shape
do_save = True

# Size of the output of planes
nWide = 3;
nHigh = 1;

n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, nHigh*nY, nWide*nX])
all_trials_raw_fix = np.zeros([n_fus_frames, nHigh*nY, nWide*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, nHigh*nY, nWide*nX])

nF_stim = nT
all_trials_resampled_at_stim = np.zeros([nF_stim, nHigh*nY, nWide*nX])

for i in range(len(fns)):
    a, b = np.unravel_index(i, [nHigh, nWide])
    exp_dict = all_exp_dicts[i]

    all_trials_raw_fix[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw_fix'], [0,2,1])
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                     = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]             = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]       = exp_dict['data_resample_at_stim']

# Export raw
if do_save:
    scio.export_tiffs(all_trials_resampled_at_stim, exportDir + stim_name + '_cumulative_trials_resample_stim.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw, exportDir + '_cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_fix, exportDir + stim_name + '_cumulative_trials_RAW_linear_interp_of_artifacts.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_medfilt, exportDir + stim_name + '_cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})

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

# ADJUST FOR THE DATA
trials_tmp = trials_dff[60:, :, :]# Chop off 30 sec gray
trial_average = trials_tmp.reshape([10, 320, nHigh*nY, nWide*nX]).mean(axis=0)
combined = trial_average

if do_save:
    np.save(exportDir + 'combined_single_trial.npy', combined)
    scio.export_tiffs(combined, exportDir + 'TRIAL_averages_all.tiff', dims = {'x':2, 'y':1, 't':0})

#%%


# First element is the Number of slices, second element is number of masks
tcs = np.zeros([3, 3, 3260])
# y0, y1, x0, x1 / one for each region
#masks = [(13, 23, 2, 12), (8, 18, 25, 35), (10, 20, 70, 80), (15, 25, 85, 95)]
#masks = [(13, 33, 105, 125), (10, 30, 82, 102), (5, 25, 60, 80)]
masks = [(13, 33, 5, 25), (10, 30, 25, 45), (5, 25, 45, 65)]

cnt = -1
for ii in range(1):
    for jj in range(3):
        cnt+=1
        for m_no, (y0, y1, x0, x1) in enumerate(masks):
            tcs[cnt, m_no, :] = trials_dff[:, 52*ii+y0:52*ii+y1, 128*jj+x0:128*jj+x1].mean(-1).mean(-1)


# import seaborn as sns

# sns.set_style('whitegrid')
# plt.figure(figsize=[5, 10])
# for i in range(15):
#     plt.subplot(15, 1, i+1)
#     plt.plot(tcs[i, 0, :], color = 'b')
#     plt.plot(tcs[i, 1, :], color = 'r')

# plt.figure(figsize=[5, 10])
# for i in range(15):
#     plt.subplot(15, 1, i+1)
#     plt.plot(tcs[i, 2, :], color = 'r')
#     plt.plot(tcs[i, 3, :], color = 'b')


# #%%

from matplotlib.patches import Rectangle
plt.figure(figsize = [10, 3])

plt.imshow(trials_dff.mean(axis = 0), cmap = 'binary', vmax = 0.4)

for ii in range(1):
    for jj in range(3):
        for m_no, (y0, y1, x0, x1) in enumerate(masks):
            curr_rect = Rectangle((128*jj+x0, 52*ii+y0), 20, 20, fill = None, color = [0, 0, 0])
            plt.gca().add_patch(curr_rect)


#%%

import seaborn as sns
tmp = tcs[:, :, 60:].reshape([3, 3, 10, 320])

# ORDER: Long Skinny -- > Animate object --> Faces  --> Stubby Inanimate (Boxy)

plt.figure(figsize = [30, 4])
for slice_n in range(3):
    plt.subplot(2,5,slice_n+1)
    sns.tsplot(tmp[slice_n, 0, :, :]+0.1, time = np.linspace(0.5, 160, 320), color = 'r')
    sns.tsplot(tmp[slice_n, 1, :, :]+0.1, time = np.linspace(0.5, 160, 320), color = 'g')
    sns.tsplot(tmp[slice_n, 2, :, :]+0.1, time = np.linspace(0.5, 160, 320), color = 'b')
    
    plt.title('slice_numebr %d' % slice_n)
    plt.ylim([-0.1, 0.5])
    sns.despine()

plt.suptitle('First 4 OBJ // Last 4 SF controls')
# for cnt, slice_n in enumerate(range(9, 12)):
#     plt.subplot(2,3,4+cnt)
#     #sns.tsplot(tmp[slice_n, 2, :, :]+0.2, color = 'r')
#     sns.tsplot(tmp[slice_n, 1, :, :]+0.2, color = or_color)
#     sns.tsplot(tmp[slice_n, 0, :, :]+0.2, color = 'k')
    
#     plt.title('slice_numebr %d' % slice_n)
#     plt.ylim([-0.1, 0.8])

#plt.savefig('/data/fUS_project/visualization/sfn2019_timecourse_for_frank_shifted.pdf');

#%%


# Compute activation maps from the data at various deltaX e.g. temporal hemodynamic offsets
#for x in range(1, 20, 5):


# tmp = trials_dff[60:, :, :]
# tmp_re = tmp.reshape([10, 320, nHigh*nY, nWide*nX]).mean(axis=0)

# m1  = tmp_re[5:15, :, :].mean(axis = 0)
# m2  = tmp_re[25:35, :, :].mean(axis = 0)
# dm  = m1-m2; m_val = np.percentile(np.abs(dm), 99)
# plt.figure(figsize = [10, 3])

# plt.imshow(gaussian_filter(dm, 2), cmap = 'PRGn', vmin = -m_val, vmax = m_val)
# plt.suptitle('OBJECTS-SCRAMBLED E.G. Green is object-prefer')

#%% Do a bunch of t-tests

# from scipy import stats

# # Discard first trial
# tmp = trials_dff[60:, :, :]
# tmp_re = tmp.reshape([10, 320, nHigh*nY, nWide*nX])


# p_vals = np.zeros([nHigh*nY, nWide*nX])
# t_vals = np.zeros([nHigh*nY, nWide*nX])

# for ii in range(nHigh*nY):
#     if ii % 50 == 0:
#         print('On iteration %d out of %d' % (ii, nHigh*nY))
#     for jj in range(nWide*nX):
#         [t_tmp, p_tmp] = stats.ttest_ind(tmp_re[:, 5:15, ii, jj].mean(axis=1), tmp_re[:, 25:35, ii, jj].mean(axis=1))
#         p_vals[ii, jj] = p_tmp
#         t_vals[ii, jj] = t_tmp

# p_sign = t_vals.copy()
# p_sign[p_sign>0] =1
# p_sign[p_sign<0] =-1

# #plt.imshow(np.log(p_vals)); plt.colorbar()
# # With sign 
# p_thresh=0.05
# #p_vals[p_vals>p_thresh] = 1

# p_val_im = p_sign*np.log10(p_vals)

# plt.figure(figsize = [18, 4])
# plt.imshow(p_sign*np.log10(p_vals), cmap = 'bwr', vmin = -3, vmax = 3); plt.colorbar()
# plt.suptitle('Log p _value with sign ')

# plt.figure(figsize = [18, 4])
# vasculature_image = all_trials_raw.mean(axis = 0)
# plt.imshow(vasculature_image, vmax = 3e10, cmap = 'binary_r'); plt.colorbar()

# # Masked version
# plt.figure(figsize = [18, 4])
# mask = np.ma.masked_where(p_vals>0.05, p_sign*np.log10(p_vals))
# #p_vals_mask = np.ma.masked_array(p_vals,mask)
# plt.imshow(vasculature_image, vmax = 3e10, cmap = 'binary_r', aspect = 'equal'); 
# plt.imshow(mask, cmap = 'bwr', vmin =-3, vmax = 3, alpha = 1, aspect = 'equal'); plt.colorbar()


#plt.imshow(dm[:52, 128*4:128*5], cmap = 'PRGn', vmin = -m_val, vmax = m_val)
