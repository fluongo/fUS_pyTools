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

##################################################################
# Load the fUS data
##################################################################

#dirs = sorted(glob.glob('/data/fUS_project/data/data_mar21/RT*.mat'))

all_exps = [   ('/data/fUS_project/data/data_mar21/RT0321_Acq_105306.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 0, 0, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_110911.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 1, 1, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_112345.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 2, 2, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_113841.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 3, 3, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_115259.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 4, 4, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_120722.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 5, 5, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_122443.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 6, 6, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_124046.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 7, 7, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_125518.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 8, 8, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_131043.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 9, 9, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_132540.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 10, 10, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_134049.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 11, 11, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_135601.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 12, 12, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_141217.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 13, 13, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_142643.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 14, 14, True, 24), 
                ('/data/fUS_project/data/data_mar21/RT0321_Acq_144133.mat','/data/fUS_project/data/data_mar21/timeline_03-21-2019_10-53.mat', 15, 15, True, 24)]

#  '/data/fUS_project/data/data_mar21/RT0321_Acq_112345.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_113841.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_115259.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_120722.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_122443.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_124046.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_125518.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_131043.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_132540.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_134049.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_135601.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_141217.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_142643.mat',
#  '/data/fUS_project/data/data_mar21/RT0321_Acq_144133.mat'

#for animal_fn, timeline_fn, exp in all_exps:

do_resample = False
do_save = False
n_trials = 10;

pw_ev_list = []
pw_az_list = []
ph_ev_list = []
ph_az_list = []
names_list = []

# range(len(all_exps))
for ii in range(len(all_exps)):
    
    animal_fn, timeline_fn, exp_stim, exp_fUS, is_retinotopy, stim_rate = all_exps[ii]
    outDir, sub_fn = os.path.split(animal_fn)

    m = scio.matloader(); # Instantiate mat object
    m.loadmat_h5(animal_fn); #m.summary()
    data_raw = m.data['Dop'].copy()
    data_raw = data_raw[:, 20:100, :]; # Cut into only the brain parts

    if do_resample:
        data = scana.bin_2d(data_raw, 2, dims = {'x':1, 'y':2, 't':0});
        data = np.transpose(data, (0, 2,1))
    else:
        data = data_raw;
    
    ##################################################################
    # Clean the timestamps
    ##################################################################

    # Loading in the timeline data
    if ii == 0: # Just load on first
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()

    # Compute the parsed version of the timestamps
    stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.1, interval_between_experiments = 3,  min_number_exp = 40); # Separate the frames for each one...

    #exp = 1; # Which experiment number
    stim_ts = stim_list[exp_stim]
    fus_ts = fus_list[exp_fUS]
    # Check that the number of fUS frames matches the experiment
    offset = len(fus_ts) - data.shape[0]
    if offset !=0: # Meaning fus timestamps and fUS dont match
        fus_ts = fus_ts[offset:] # truncate

    # Upsample the interpolation of each frame to account for stim ttl only getting sent once every 3rd frame
    #    stim_ts = fUS.updample_timestamps(stim_ts, 3)
    fus_rate = 2; # fus_rate in hz

    newx = stim_ts[1:-1: int(stim_rate/(fus_rate))]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....

    data_resample_at_stim = scana.resample_xyt(data, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})

    # Turn into DFF
    f0 = np.repeat(np.expand_dims( data_resample_at_stim.mean(axis=0), axis = 0), len(newx), axis = 0); # Makes a f0 for each timepoint
    data_resample_at_stim = (data_resample_at_stim -f0)/f0

    if is_retinotopy:
        nF = int(data_resample_at_stim.shape[0]/2/n_trials) # Number of frames for each trial
        
        # Concatenate across trials and then output to tiff the original as well as the others
        trials_az = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])
        trials_el = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])

        for i in range(n_trials):
            # Write each trial
            trials_az[i, :, :, :] = data_resample_at_stim[i*nF:(i+1)*nF, :, :]
            trials_el[i, :, :, :] = data_resample_at_stim[n_trials*nF+i*nF:n_trials*nF+(i+1)*nF, :, :]

        ph_ev, pw_ev = scana.compute_fft(trials_el.mean(axis=0), dims = {'x':2, 'y':1, 't':0}, doPlot = False)
        ph_az, pw_az = scana.compute_fft(trials_az.mean(axis=0), dims = {'x':2, 'y':1, 't':0}, doPlot = False)

        # Compute a mask by cross validating the fus
        fs = scana.compute_field_sign(ph_az, ph_ev, pw_az, pw_ev, filt_size=1); 
        plt.title(sub_fn);

        pw_ev_list.append(pw_ev);pw_az_list.append(pw_az) 
        ph_ev_list.append(ph_ev);ph_az_list.append(ph_az) 
        names_list.append(sub_fn)
        
        if do_save:
            add_str = '_re' if do_resample else ''             
            plt.savefig(outDir + '/' + sub_fn[:-4] + add_str + '.pdf')

            exportDir = outDir + '/extras/'
            if not os.path.exists(exportDir):
                os.mkdir(exportDir)
            
            scio.export_tiffs(data_resample_at_stim, exportDir + sub_fn[:-4] + add_str + '.tiff', dims = {'x':2, 'y':1, 't':0})
            scio.export_tiffs(trials_az.mean(axis = 0), exportDir + sub_fn[:-4] + add_str + '_az.tiff', dims = {'x':2, 'y':1, 't':0})
            scio.export_tiffs(trials_el.mean(axis = 0), exportDir + sub_fn[:-4] + add_str + '_el.tiff', dims = {'x':2, 'y':1, 't':0})

    else:
        if do_save:
            exportDir = outDir + '/extras/'
            if not os.path.exists(exportDir):
                os.mkdir(exportDir)
            scana.export_tiffs(data_resample_at_stim, exportDir + sub_fn[:-4]+'.tiff', dims = {'x':2, 'y':1, 't':0})



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