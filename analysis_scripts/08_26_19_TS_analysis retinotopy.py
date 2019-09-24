
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

from PIL import Image

##################################################################
# Load the fUS data
##################################################################
fns = sorted(glob.glob('/data/fUS_project/data/data_aug26/RT*.mat')); # Only first 7 are the retinotopu
timelines = len(fns)*['/data/fUS_project/data/data_aug26/timeline_08-26-2019_11-39.mat']


n_fus = range(len(fns))
n_stim = range(len(fns))

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]
# ONly use experiments 1 and 4
#all_exps = [all_exps[0], all_exps[3]]


n_trials = 15;
fus_rate =  2; # THIS EXPEIMENT WAS 2 HZ

# Load in example movies
example_mov_fn = '/data/linuxData/MASTER_STIMULUS_FOLDER/STIMULUS_GENERATION_HERE/Francisco/fUS_retinotopy/fUS_TS_2019_Aug_4_10reps_for_rev_40s_cycle.mat'
m = scio.matloader();
m.loadmat_h5(example_mov_fn); #timestamps_m.summary()

# Extract example movies
mov_examples = {}
mov_examples['LR'] = m.data['moviedata'][np.linspace(0, 1200, 80).astype(int), :, :]
mov_examples['RL'] = m.data['moviedata'][np.linspace(10*1200, 11*1200, 80).astype(int), :, :]
mov_examples['UD'] = m.data['moviedata'][np.linspace(20*1200, 21*1200, 80).astype(int), :, :]
mov_examples['DU'] = m.data['moviedata'][np.linspace(30*1200, 31*1200-1, 80).astype(int), :, :]
# Resize into 52 x 128
for xx in mov_examples.keys():
    new_arr = np.zeros([80, 52, 128])
    for ii in range(80):
        new_arr[ii, :, :] = np.array(Image.fromarray(mov_examples[xx][ii, :, :]).resize([52, 128])).T # Transpose is important for LR
    mov_examples[xx] = new_arr


def remove_movement_artifact_from_raw_and_condition(data):
    # Removes the movement artifact from the data that comes in the form of nT, nX, nY
    sub_data = data_raw[:, :4, :].mean(axis = -1).mean(axis=-1)
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
        except: # ANything goes wrong and just set to median
            data_fix[xx, :, :] = np.median(data_fix, axis = 0)
    
    return data_fix

# data_fix = remove_movement_artifact_from_raw_and_condition(data_raw)
# plt.figure(figsize = [20, 3]);
# plt.plot(data_fix[:, :40, :40].mean(axis = -1).mean(axis=-1))
# plt.figure(figsize = [20, 3]);
# plt.plot(data_raw[:, :40, :40].mean(axis = -1).mean(axis=-1))

# CHECK THE TIMESTAMPS
# timeline_fn = '/data/fUS_project/data/data_aug26/timeline_08-26-2019_11-39.mat'

# timestamps_m = scio.matloader();
# timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
# ts = timestamps_m.data['timestamps'].copy()
# stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 4,  min_number_exp = 1000); # Separate the frames for each one...
# fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 10,  min_number_exp = 100); # Separate the frames for each one...
# for tmp_ss, tmp_ff in zip(stim_list, fus_list):
#     print('Number of frames stim: %d // fus: %d' % (len(tmp_ss), len(tmp_ff)))

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
        fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 10,  min_number_exp = 100); # Separate the frames for each one...
        for tmp_ss, tmp_ff in zip(stim_list, fus_list):
            print('Number of frames stim: %d // fus: %d' % (len(tmp_ss), len(tmp_ff)))

    # Recompute at appropriate times
    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus][1:]; # Chop off first
    if ii == 3: # Chop off last
        fus_ts = fus_ts[:-1]

    # # Compute the parsed version of the timestamps
    playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    
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
    trials_all['az_LR'] = data_resample_at_stim[:800, :, :]
    trials_all['az_RL'] = data_resample_at_stim[800:1600, :, :]
    trials_all['ev_UD'] = data_resample_at_stim[1600:2400, :, :]
    trials_all['ev_DU'] = data_resample_at_stim[2400:3200, :, :]
    
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

np.save(exportDir + 'data_processed.npy', all_exp_dicts)

#%%
# # Export the cumulative tiff for trial averaged

# Size of exported image
all_exp_dicts = np.load(exportDir + 'data_processed.npy', allow_pickle = True)
[nT, nY, nX] = all_exp_dicts[0]['data_resample_at_stim'].shape
do_save = False

nWide = len(all_exp_dicts);

n_fus_frames = all_exp_dicts[0]['data_raw'].shape[0]
all_trials_raw = np.zeros([n_fus_frames, 1*nY, nWide*nX])
all_trials_raw_fix = np.zeros([n_fus_frames, 1*nY, nWide*nX])
all_trials_raw_medfilt = np.zeros([n_fus_frames, 1*nY, nWide*nX])

nF_stim = nT;
all_trials_resampled_at_stim = np.zeros([nF_stim, 1*nY, nWide*nX])

for i in range(len(all_exps)):
    a, b = np.unravel_index(i, [1, nWide])
    exp_dict = all_exp_dicts[i]

    all_trials_raw_fix[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                 = np.transpose(exp_dict['data_raw_fix'], [0,2,1])
    all_trials_raw[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]                     = np.transpose(exp_dict['data_raw'], [0,2,1])
    all_trials_raw_medfilt[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]             = np.transpose(exp_dict['data_raw_medfilt'], [0,2,1])
    all_trials_resampled_at_stim[:, a*nY:(a+1)*nY, b*nX:(b+1)*nX]       = exp_dict['data_resample_at_stim']

    # TODO Make a version that does the trial averages....

#scio.export_tiffs(all_trials, exportDir + 'cumulative_trials_spots.tiff', dims = {'x':2, 'y':1, 't':0})
#scio.export_tiffs(all_trials_dff, exportDir + 'cumulative_trials_spots_dff.tiff', dims = {'x':2, 'y':1, 't':0})

# Export raw
if do_save:
    scio.export_tiffs(all_trials_resampled_at_stim, exportDir + 'RETINOTOPY_cumulative_trials_resample_stim.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw, exportDir + 'RETINOTOPY_cumulative_trials_RAW.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_fix, exportDir + 'RETINOTOPY_cumulative_trials_RAW_linear_interp_of_artifacts.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(all_trials_raw_medfilt, exportDir + 'RETINOTOPY_cumulative_trials_RAW_medfilt_5_5_5.tiff', dims = {'x':2, 'y':1, 't':0})

# Make a DFF from median filter
all_trials_medfilt_dff  = all_trials_raw_medfilt.copy()
f0                      = np.median(all_trials_medfilt_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(n_fus_frames)]), [2,0,1])
all_trials_medfilt_dff  = (all_trials_medfilt_dff - f0)/f0

if do_save:
    scio.export_tiffs(all_trials_medfilt_dff, exportDir + 'RETINOTOPY_RAW_medfilt_5_5_5_convert_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

# Do the same for the resampled at stim_trials with medfilt
trials_dff              = all_trials_resampled_at_stim.copy()
f0                      = np.median(trials_dff, axis= 0)
f0_mat                  = np.transpose(np.dstack([f0 for i in range(trials_dff.shape[0])]), [2,0,1])
trials_dff              = (trials_dff - f0)/f0
#trials_dff              = median_filter(trials_dff, size = [6,5,5])
if do_save:
    scio.export_tiffs(trials_dff, exportDir + 'RETINOTOPY_trial_synchronized_toDFF.tiff', dims = {'x':2, 'y':1, 't':0})

trial_averageLR = trials_dff[:800, :, :].reshape([10, 80, 52, nWide*nX]).mean(axis=0)
trial_averageRL = trials_dff[800:1600, :, :].reshape([10, 80, 52, nWide*nX]).mean(axis=0)
trial_averageUD = trials_dff[1600:2400, :, :].reshape([10, 80, 52, nWide*nX]).mean(axis=0)
trial_averageDU = trials_dff[2400:3200, :, :].reshape([10, 80, 52, nWide*nX]).mean(axis=0)

combined = np.zeros([80, 52*4, nWide*nX])
combined[:, :52, :] = trial_averageLR
combined[:, 52: (2*52), :] = trial_averageRL
combined[:, (2*52) : (3*52), :] = trial_averageUD
combined[:, (3*52) : (4*52), :] = trial_averageDU

if do_save:
    np.save(exportDir + 'combined_single_trial.npy', combined)

# Now add the movie
max_val = np.max(combined)
min_val = np.max(combined)
combined_uber = np.zeros([80, 52*4, (nWide+1)*nX])
for cnt, xx in enumerate(['LR', 'RL', 'UD', 'DU']):
    tmp = mov_examples[xx];
    tmp = tmp-np.min(tmp); tmp = tmp/np.max(tmp);
    combined_uber[:, cnt*52 : (cnt+1)*52, nWide*nX : (nWide+1)*nX] = tmp
combined_uber[:, :, :5*nX] = combined

if do_save: 
    scio.export_tiffs(combined, exportDir + 'TRIAL_averages_all.tiff', dims = {'x':2, 'y':1, 't':0})
    scio.export_tiffs(combined_uber, exportDir + 'TRIAL_averages_all_with_stimulus.tiff', dims = {'x':2, 'y':1, 't':0})

#%% Attempt to combine them all with a simple kernel density estimate and mode to get an estimate at each stimulus frame.....

nn = 0 # Experiment number
d = all_exp_dicts[nn]
print(d.keys())
xx = d['fus_ts_stim'].copy()
yy = d['data_raw_fix'].copy()


plt.figure(figsize = [10, 3])
sub_x = np.mod(xx[idx], 1200)
sub_y = yy[idx, 55, 5]
sub_y = (sub_y - np.median(sub_y)) / np.median(sub_y)


test = sub_y[(sub_x>450) & (sub_x<510)]

#%%
from scipy.stats import gaussian_kde

def compute_ts_smooth_kde(x_data, y_data, x_vals, window=60):
    # Computes a sliding window average of the data using a kernel density estimate of the mode
    # x_data and y_data is the function, x_vals is which values to interpolate at, window is the size of the window
    nT = len(x_vals)
    kde_smoothed = []
    for i in range(nT):
        subset_y = y_data[ (x_data> max(0, i-window/2) ) & (x_data<min(nT-2, i+window/2) )]
        kde_smoothed.append(kde_calc(subset_y, plot_hist = False))
    return kde_smoothed

def kde_calc(x, plot_hist = True):
    # Takes as input the x, does a kernel density smooth over time then returns data
    bandwidth = 5;
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    x_grid = np.linspace(-2, 2, 3000)
    kde_plot = kde.evaluate(x_grid)
    if plot_hist:
        plt.figure(); plt.plot(x_grid, kde_plot)
    return x_grid[np.argmax(kde_plot)]

from joblib import Parallel, delayed
from tqdm import tqdm

combined_smoothed = np.zeros([1200, combined.shape[1], combined.shape[2]])

for exp_n in range(5):
    d = all_exp_dicts[exp_n]
    xx = d['fus_ts_stim'].copy()
    yy = d['data_raw_fix'].copy()
    for condition in range(4):
        # Only take indices pertininent to the condition
        idx = np.argwhere((xx < 1200*10*(condition+1) ) & (xx>1200*10*condition))
        sub_x = np.mod(xx[idx], 1200)
        for xi in tqdm(range(nX)):
            list_subx   = [sub_x for i in range(nY)]
            list_suby   = [(yy[idx, xi, yi] - np.median(yy[idx, xi, yi])) / np.median(yy[idx, xi, yi]) for yi in range(nY)]
            list_xvals  = [list(np.arange(1200)) for i in range(nY)]
            results = Parallel(n_jobs=26)(delayed(compute_ts_smooth_kde)(i,j,k) for i,j,k in zip(list_subx,list_suby, list_xvals))
            # Now assign into their place
            for yi in range(nY):
                combined_smoothed[:, condition*nY + yi, exp_n*nX +xi] = results[yi]

#np.save(exportDir + 'interpolated_trial_average.npy', combined_smoothed)
#scio.export_tiffs(combined_smoothed, exportDir + 'TRIAL_SMOOTHED.tiff', dims = {'x':2, 'y':1, 't':0})

combined_smoothed = np.load(exportDir + 'interpolated_trial_average.npy')



#%% CHECK THE POWER
from scipy.ndimage import gaussian_filter

out = np.fft.fft(combined, axis = 0); # take along time

# Position 1 is stimulus harmonic
plt.figure(figsize = [20, 10])
phase_map = np.angle(out[1, :, :])
power_map = np.abs(out[1, :, :])
phase_map[52:2*52, :] = -phase_map[52:2*52, :]
phase_map[52*3:4*52, :] = -phase_map[52*3:4*52, :]
# Combine phases
phase_average = phase_map.copy()
phase_average[52*0:1*52, :] = (phase_average[52*0:1*52, :] + phase_average[52*1:2*52, :])/2
phase_average[52*1:2*52, :] = (phase_average[52*2:3*52, :] + phase_average[52*3:4*52, :])/2
phase_average = phase_average[0:52*2, :]

plt.subplot(2,2,1); plt.imshow(phase_map, cmap = 'hsv');
plt.subplot(2,2,2); plt.imshow(power_map, cmap = 'binary');
plt.figure(figsize = [20, 5])
plt.imshow(phase_average, cmap = 'hsv'); plt.colorbar()

power_re = power_map.reshape([4, 52, 640]).sum(axis = 0)
alpha_map = median_filter(power_re, [15, 15])
alpha_cut = np.percentile(alpha_map.ravel(), 90)

#alpha_map = gaussian_filter(power_re, 5)
plt.figure(figsize = [20, 5]); 
plt.imshow(alpha_map, cmap = 'binary')
plt.figure(figsize = [20, 5]); 
plt.imshow(alpha_map, vmax = alpha_cut, cmap = 'binary')

plt.figure(figsize = [20, 5]); 

plt.subplot(2,1,1)
masked = np.ma.masked_where(alpha_map < alpha_cut, phase_average[:52, :])
plt.imshow(masked, cmap = 'hsv'); plt.title('Aximuth'); plt.colorbar()

plt.subplot(2,1,2)
masked = np.ma.masked_where(alpha_map < alpha_cut, phase_average[52:, :])
plt.imshow(masked, cmap = 'hsv'); plt.title('Elevation'); plt.colorbar()

#%% All plane fourier

#trials_dff_medfilt6 = median_filter(trials_dff, size = [6,5,5])
#trials_dff_medfilt10 = median_filter(trials_dff, size = [10,5,5])

def get_and_plot_phase_maps(data1, data2, idx_fft):
    # For making things tidier
    plt.figure(figsize = [13, 8]); 
    
    out = np.fft.fft(data1, axis = 0); # take along time
    
    phaseRL = np.angle(out[idx_fft, :, :] )
    plt.subplot(5,1,1); plt.imshow(phaseRL); plt.title('Direction 1');  plt.colorbar();
    powerRL = np.abs(out[idx_fft, :, :] )
    plt.subplot(5,1,2); plt.imshow(powerRL, cmap = 'binary'); plt.title('Power');  plt.colorbar();

    out = np.fft.fft(data2, axis = 0); # take along time
    phaseLR = np.angle(out[idx_fft, :, :])
    plt.subplot(5,1,3); plt.imshow(-phaseLR); plt.title('Direction 2');  plt.colorbar();
    powerLR = np.abs(out[idx_fft, :, :])
    plt.subplot(5,1,4); plt.imshow(powerLR, cmap = 'binary'); plt.title('Power');  plt.colorbar();

    plt.subplot(5,1,5)
    plt.imshow((phaseRL-phaseLR)/2, cmap = 'hsv'); plt.colorbar(); 
    
    return phaseRL, phaseLR, (phaseRL-phaseLR)/2

phaseRL, phaseLR, combined = get_and_plot_phase_maps(trials_dff_medfilt[:800, :, :], trials_dff_medfilt[800:, :, :], 10)
plt.title('FULL DATASET AZIMUTH')

phaseRL, phaseLR, combined = get_and_plot_phase_maps(trials_dff_medfilt[1600:2400, :, :], trials_dff_medfilt[2400:, :, :], 10)
plt.title('FULL DATASET ELEVATION')

#%%


trials_dff_medfilt.shape

plt.figure(figsize = [20, 2]); 
# Plot one direction
pY = 10; # Y position
pX = 180-128
# for cnt, xx in enumerate(range(0, 20, 4)):
#     pX = 180-xx; # X position
#     plt.plot(trials_dff_medfilt[:800, pY, pX],color = [cnt/4.*1, cnt/4.*1, cnt/4.*1])

plt.plot(trials_dff[1600:2400, pY, pX])
plt.plot(trials_dff_medfilt[1600:2400, pY, pX])
plt.plot(trials_dff_medfilt10[1600:2400, pY, pX])

# # Plot the other direction
#plt.plot(trials_dff[:800, pY, pX])
plt.plot(trials_dff_medfilt[2400:3200, pY, pX])
# plt.plot(trials_dff_medfilt10[:800, pY, pX])

#%%

combined.shape


plt.figure(figsize = [20, 2]); 
# Plot one direction
pY = 10+2*52; # Y position
pX = 50+1*128

plt.plot(combined[:, pY, pX])


#%%


for xx in range(len(all_exps)):
    exp_dict = all_exp_dicts[xx]; # Dictionary for current experiment
    idx_fft = 5;
    plt.figure(figsize = [20, 4])

    #NON TRIAL BASED
    fourier_results = {}
    alpha_mask = np.zeros([nY, nX]); # For making a mask of the relevant areas
    for pp, stim_key in enumerate(['az_LR', 'az_LR']): # 'az_LR', 'az_RL', 'ev_DU', 
        data = exp_dict['trials_all'][stim_key]
        data_zs = np.zeros_like(data)

        for ii in range(nY):
            for jj in range(nX):
                data_zs[:, ii, jj] = (data[:, ii, jj] - np.mean(data[:, ii, jj]))/np.std(data[:, ii, jj])

        out = np.fft.fft(data_zs, axis = 0); # take along time
        plt.figure(figsize = [9, 9]); plt.suptitle('Slice %d' % xx)
        all_powers = []
        for i in range(24):
            plt.subplot(6, 4, i+1)
            plt.imshow(np.abs(out[i, :, :]))
            all_powers.append(np.sum(np.abs(out[i, :, :])))
            plt.title(str(i))
        plt.figure(); plt.plot(np.fft.fftfreq(n=800, d = 0.5)[:24], all_powers)






#%% POWER ANALYSIS

for xx in range(len(all_exps)):
    exp_dict = all_exp_dicts[xx]; # Dictionary for current experiment
    idx_fft = 10;
    plt.figure(figsize = [20, 4])
    #NON TRIAL BASED

    fourier_results = {}
    alpha_mask = np.zeros([nY, nX]); # For making a mask of the relevant areas
    for pp, stim_key in enumerate(['az_LR', 'az_RL', 'ev_DU', 'ev_UD']):
        data = exp_dict['trials_all'][stim_key]
        data_zs = np.zeros_like(data)

        for ii in range(nY):
            for jj in range(nX):
                data_zs[:, ii, jj] = (data[:, ii, jj] - np.mean(data[:, ii, jj]))/np.std(data[:, ii, jj])

        out = np.fft.fft(data_zs, axis = 0); # take along time

        fourier_results[stim_key] = {}
        if stim_key in ['az_LR', 'ev_DU']: # Flip one of them
            fourier_results[stim_key]['phase'] = -np.angle(out[idx_fft, :, :])
        else:
            fourier_results[stim_key]['phase'] = np.angle(out[idx_fft, :, :])
        fourier_results[stim_key]['power'] =np.abs(out[idx_fft, :, :])

        plt.subplot(2,4, pp+1)
        plt.imshow(fourier_results[stim_key]['power'], cmap = 'binary'); plt.title(stim_key); plt.colorbar()
        plt.subplot(2,4, pp+5)
        plt.imshow(fourier_results[stim_key]['phase'], cmap = 'hsv'); plt.title(stim_key); plt.colorbar()
        
        alpha_mask += fourier_results[stim_key]['power']

    # Make the alpha mask
    alpha_mask = gaussian_filter(alpha_mask, 2);
    alpha_mask = alpha_mask/np.max(alpha_mask);
    alpha_mask[alpha_mask>0.6] = 1;
    alpha_mask[alpha_mask<=0.6] = 0;
    plt.savefig(outDir + '/extras/slice_%d_parts.pdf' % xx)

    # NOW COMBINE THEM INTO A SINGLE MAP
    plt.figure(figsize = [10, 2])
    for ll, (p1, p2) in enumerate([('az_LR', 'az_RL')]):
        plt.subplot(1,2,ll+1)
        combined_map = ((fourier_results[p1]['phase'] + (-fourier_results[p2]['phase']) )  )/2

        # Set all outside values to zero
        combined_map = (combined_map + np.pi) * alpha_mask
        # Make custom colormap with zero value as black
        hsv_custom = cm.get_cmap('hsv', 256)
        hsv_custom = hsv_custom(np.linspace(0, 1, 256))
        hsv_custom[0, :] = 0; # Turn 0 to black
        hsv_custom = ListedColormap(hsv_custom)

        plt.imshow(combined_map, cmap = hsv_custom); plt.colorbar()
        plt.contour(alpha_mask, [0.5], colors = 'w', linewidths = 2)
        plt.title(p1)
    plt.savefig(outDir + '/extras/slice_%d_masked.pdf' % xx)
