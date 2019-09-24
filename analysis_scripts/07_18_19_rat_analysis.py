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
from scipy.ndimage.filters import median_filter

##################################################################
# Load the fUS data
##################################################################

fns = sorted(glob.glob('/data/fUS_project/data/data_july18/RT*.mat'))
timelines = len(fns)*['/data/fUS_project/data/data_july18/timeline_07-18-2019_12-51.mat']
n_fus = range(len(fns))
n_stim = range(len(fns))

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10;
fus_rate =  1; # THIS EXPEIMENT WAS 1 HZ


#%%

# corr_list = []
# corr_list_zscore = []
# corr_im_list = []
# trials_list = []; trials_list_dff = []
# data_raw_list = []
# data_spots_list = []

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

    # # # Load timeline only on the first experiment
    if ii in [0]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()
        stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
        fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 3,  min_number_exp = 100); # Separate the frames for each one...

    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus];

    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})

    #########################################################
    ########################################################
    #### 3 SPOTS
    ##########################################################
    ############################################################

    # Top right discarding of invalid frames
    do_discard = True
    if do_discard:
        br = data_resample_at_stim[:, :, :10].mean(axis = -1).mean(axis = -1)
        invalid_frames = np.where(br > np.mean(br)+1*np.std(br))[0]
        #invalid_frames = np.where(br > np.mean(br))[0]
        
        print('Discarding %d frames' % len(invalid_frames))
        data_resample_at_stim[invalid_frames, : , :] = np.nan

    
    # Stimuli for each experiment for cross correlation
    stim = np.zeros([3, 300])
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
    data_dff = (data_resample_at_stim - f0)/f0

    corr_maps = np.zeros([3, nY, nX])
    
    for aa in range(3):
        for xx in range(nX):
            for yy in range(nY):
                corr_maps[aa, yy, xx] = scnpix.nan_pearson(data_resample_at_stim[:, yy, xx], stim[aa])
        plt.subplot(1,4,aa+1)
        plt.imshow(corr_maps[aa, :, :], cmap = 'afmhot'); plt.colorbar()

    # corr_list.append(corr_maps)

    # # Compute the trial mean
    trials = data_resample_at_stim.reshape([n_trials, int(nT/n_trials), nY, nX])
    trials_dff = data_dff.reshape([n_trials, int(nT/n_trials), nY, nX])
    
    trials_export = np.maximum(np.nanmean(trials, axis = 0), 0)
    trials_export = trials_export/np.max(trials_export)
    trials_dff = np.maximum(np.nanmean(trials_dff, axis = 0), 0)
    trials_dff = trials_dff/np.max(trials_dff)
    
    # trials_list.append(trials_export)
    # trials_list_dff.append(trials_dff)

    # data_spots_list.append(data_resample_at_stim)
    # data_raw_list.append(data_raw)

    # Compute the fus_stim_time for each frame
    fus_ts_stim = np.zeros_like(fus_ts)
    for ww in range(data_raw.shape[0]):
        fus_ts_stim[ww] = np.argmin(np.abs(stim_ts - fus_ts[ww]));

    curr_dict = {}
    curr_dict['stim_ts'] = stim_ts
    curr_dict['fus_ts'] = fus_ts
    curr_dict['fus_ts_stim'] = fus_ts_stim
    curr_dict['data_raw'] = data_raw
    curr_dict['data_raw_medfilt'] = median_filter(data_raw, size = [5,5,5])
    curr_dict['data_resample_at_stim'] = data_resample_at_stim
    curr_dict['data_trials_dff'] = trials
    curr_dict['data_trials_export'] = trials_export
    curr_dict['data_trial_mean_dff'] = trials_dff
    curr_dict['corr_maps'] = corr_maps

    all_exp_dicts.append(curr_dict)

    plt.savefig(exportDir + sub_fn[:-4]+'_trial_average_spots.pdf')


#%%
# # Export the cumulative tiff for trial averaged
all_trials = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
all_trials_dff = np.zeros([int(30*fus_rate), 4*nY, 4*nX])
n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 4*nY, 4*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, 4*nY, 4*nX])

nF_stim = nT;
all_trials_resampled_at_stim = np.zeros([nF_stim, 4*nY, 4*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [4, 4])
    exp_dict = all_exp_dicts[i]

    all_trials[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]             = exp_dict['data_trials_export']
    all_trials_dff[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]         = exp_dict['data_trial_mean_dff']
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]         = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX] = exp_dict['data_resample_at_stim']

scio.export_tiffs(all_trials, exportDir + 'cumulative_trials_spots.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_spots_dff.tiff', dims = {'x':2, 'y':1, 't':0})

# Export raw
scio.export_tiffs(all_trials_raw, exportDir + 'cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
scio.export_tiffs(all_trials_raw_medfilt, exportDir + 'cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})


#%% REGENERATE BUT FROM THE MEDIAN FILTERED RAW
plt.figure(figsize = [10, 20])

for ii in range(len(all_exps)):
    exp_dict = all_exp_dicts[ii]; # Dictionary for current experiment
    newx = exp_dict['stim_ts'][1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    data_resamp_med = scana.resample_xyt(exp_dict['data_raw_medfilt'], exp_dict['fus_ts'], newx, dims = {'x':1, 'y':2, 't':0})
    
    corr_maps = np.zeros([3, nY, nX])
    for aa in range(3):
        for xx in range(nX):
            for yy in range(nY):
                corr_maps[aa, yy, xx] = scnpix.nan_pearson(data_resamp_med[:, yy, xx], stim[aa])
        plt.subplot(7,3,3*ii+aa+1)
        plt.imshow(corr_maps[aa, :, :], cmap = 'afmhot'); plt.colorbar()


#%% EXPERIMENT 3 GAUSSIAN ANALYSIS
exp_dict = all_exp_dicts[3]; # Dictionary for current experiment
Y = exp_dict['data_raw_medfilt'][:, 25, 20]
Y = (Y - np.mean(Y))/np.mean(Y);
X = exp_dict['fus_ts_stim']


plt.scatter(np.mod(X, 900), Y, s = 2)


#%% GAUSSIAN PROCESS
import GPy
import seaborn as sns
sns.set_style('white')
kernel = GPy.kern.RBF(input_dim=1, variance = 1., lengthscale= 10.)
m = GPy.models.GPRegression(np.mod(X, 900).reshape(-1,1), Y.reshape(-1,1), kernel)

# the normal way
m.optimize(messages=True)
# with restarts to get better results
m.optimize_restarts(num_restarts = 5)
m.plot()



#%% Do the fourier
################################################
################################################
################################################
################################################
################################################
################################################

fig_size = [25, 25]

#NON TRIAL BASED
data_resamp_med = scana.resample_xyt(all_exp_dicts[3]['data_raw_medfilt'], exp_dict['fus_ts'], newx, dims = {'x':1, 'y':2, 't':0})
data = data_resamp_med
data_zs = np.zeros_like(data)

for ii in range(nY):
    for jj in range(nX):
        data_zs[:, ii, jj] = (data[:, ii, jj] - np.mean(data[:, ii, jj])) /np.std(data[:, ii, jj])

idxFFT = 13

out = np.fft.fft(data_zs, axis = 0); # take along time
plt.figure(figsize = fig_size)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(np.angle(out[i, :, :]), cmap = 'hsv'); 
    plt.title('phase'); plt.colorbar()
#plt.imshow(np.abs(out[idxFFT, :, :]), vmax = 5, cmap = 'Greys'); plt.colorbar()
#plt.figure(figsize = fig_size)
#plt.imshow(np.angle(out[idxFFT, :, :]), cmap = 'RdBu'); plt.colorbar()


#%%
