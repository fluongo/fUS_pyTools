# Analysis example

import fUS_tools as fUS
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


##################################################################
# Load the fUS data
##################################################################

all_exps =     [ 	('/data/fUS_project/data/data_feb22/RT1_Acq_161259.mat', '/data/fUS_project/data/data_feb22/animal1_retinotopy.mat', 0, True, 20), 
					('/data/fUS_project/data/data_feb22/RT1_Acq_165438.mat', '/data/fUS_project/data/data_feb22/animal2_retinotopy_and_check.mat', 0, False, 5), 
					('/data/fUS_project/data/data_feb22/RT1_Acq_170246.mat', '/data/fUS_project/data/data_feb22/animal2_retinotopy_and_check.mat', 1, True, 20)]

#for animal_fn, timeline_fn, exp in all_exps:
animal_fn, timeline_fn, exp, is_retinotopy, stim_rate = all_exps[2]

m = fUS.matloader(); # Instantiate mat object
m.loadmat_h5(animal_fn); m.summary()
data = m.data['Dop'].copy()

##################################################################
# Clean the timestamps
##################################################################

# Loading in the timeline data
m = fUS.matloader();
m.loadmat_h5(timeline_fn); m.summary()
ts = m.data['timestamps'].copy()

# Compute the parsed version of the timestamps
stim_list, times = fUS.parse_timestamps(m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
fus_list, times = fUS.parse_timestamps(m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 3); # Separate the frames for each one...

#exp = 1; # Which experiment number
stim_ts = stim_list[exp]
fus_ts = fus_list[exp]

# Upsample the interpolation of each frame to account for stim ttl only getting sent once every 3rd frame
fus_rate = 1; # fus_rate in hz

if is_retinotopy:
	newx = stim_ts[1:-1:stim_rate/(fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
else:
    newx = stim_ts[1:-1:3]; # For the checkerboard stimulus

if data.shape[0] != len(fus_ts):
    fus_ts = fus_ts[:-1]

data_resample_at_stim = fUS.resample_xyt(data, fus_ts, newx, dims = [2,1,0])

if is_retinotopy:
    nF = 30*fus_rate;
    n_trials = 30;
    # Concatenate across trials and then output to tiff the original as well as the others
    trials_az = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])
    trials_el = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])

    for i in range(10):
        # Write each trial
        trials_az[i, :, :, :] = data_resample_at_stim[i*nF:(i+1)*nF, :, :]
        trials_el[i, :, :, :] = data_resample_at_stim[n_trials*nF+i*nF:n_trials*nF+(i+1)*nF, :, :]

    reload(fUS)
    ph_ev, pw_ev = fUS.compute_fft(trials_el.mean(axis=0), dims = [1, 2,0], doPlot = False)
    ph_az, pw_az = fUS.compute_fft(trials_az.mean(axis=0), dims = [1, 2,0], doPlot = False)
    fs = fUS.compute_field_sign(ph_az, ph_ev, pw_az, pw_ev, filt_size=3); 
    # Save the field sign
    plt.title(animal_fn); 
    plt.savefig(animal_fn[:-4]+'.pdf')

    fUS.export_tiffs(data_resample_at_stim, outDir = animal_fn[:-4]+'.tiff', dims = [1,2,0])
    fUS.export_tiffs(trials_az.mean(axis = 0), outDir = animal_fn[:-4]+'_az.tiff', dims = [1,2,0])
    fUS.export_tiffs(trials_el.mean(axis = 0), outDir = animal_fn[:-4]+'_el.tiff', dims = [1,2,0])
else:
    fUS.export_tiffs(data_resample_at_stim, outDir = animal_fn[:-4]+'.tiff', dims = [1,2,0])