# Analysis example

import fUS_tools as fUS
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from importlib import reload

##################################################################
# Load the fUS data
##################################################################

all_exps =     [   ('/data/fUS_project/data/data_mar08/RT0308_Acq_120258.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 0, 0, True, 24), 
                    
                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_124546.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 2, 1, False, 6), 
                    
                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_125902.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 3, 2, True, 24), 
                    
                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_133933.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 4, 3, False, 6),

                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_135328.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 5, 4, True, 24),

                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_143248.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 6, 5, False, 6), 
                    
                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_145011.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 8, 6, True, 24), 
                    
                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_152912.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 9, 7, False, 6), 
                    
                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_154354.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 10, 8, True, 24),

                    ('/data/fUS_project/data/data_mar08/RT0308_Acq_162312.mat','/data/fUS_project/data/data_mar08/timeline_03-08-2019_12-01.mat', 11, 9, False, 6)]

#for animal_fn, timeline_fn, exp in all_exps:

do_resample = True
do_save = False

pw_ev_list = []
pw_az_list = []
ph_ev_list = []
ph_az_list = []
names_list = []

# range(len(all_exps))
for ii in [0]:
    
    animal_fn, timeline_fn, exp_stim, exp_fUS, is_retinotopy, stim_rate = all_exps[ii]
    outDir, sub_fn = os.path.split(animal_fn)

    m = fUS.matloader(); # Instantiate mat object
    m.loadmat_h5(animal_fn); m.summary()
    data = m.data['Dop'].copy()

    if do_resample:
        data = fUS.bin_2d(data, 2);

    ##################################################################
    # Clean the timestamps
    ##################################################################

    # Loading in the timeline data
    if ii == 0: # Just load on first
        timestamps_m = fUS.matloader();
        timestamps_m.loadmat_h5(timeline_fn); timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()

    # Compute the parsed version of the timestamps
    stim_list, times = fUS.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = fUS.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.1, interval_between_experiments = 3); # Separate the frames for each one...

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

    data_resample_at_stim = fUS.resample_xyt(data, fus_ts, newx, dims = [2,1,0])

    # Turn into DFF
    f0 = np.repeat(np.expand_dims( data_resample_at_stim.mean(axis=0), axis = 0), len(newx), axis = 0); # Makes a f0 for each timepoint
    data_resample_at_stim = (data_resample_at_stim -f0)/f0

    if is_retinotopy:
        n_trials = 30;
        nF = int(data_resample_at_stim.shape[0]/2/n_trials) # Number of frames for each trial
        
        # Concatenate across trials and then output to tiff the original as well as the others
        trials_az = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])
        trials_el = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])

        for i in range(n_trials):
            # Write each trial
            trials_az[i, :, :, :] = data_resample_at_stim[i*nF:(i+1)*nF, :, :]
            trials_el[i, :, :, :] = data_resample_at_stim[n_trials*nF+i*nF:n_trials*nF+(i+1)*nF, :, :]

        ph_ev, pw_ev = fUS.compute_fft(trials_el.mean(axis=0), dims = [1, 2, 0], doPlot = False)
        ph_az, pw_az = fUS.compute_fft(trials_az.mean(axis=0), dims = [1, 2, 0], doPlot = False)
        
        # Compute a mask by cross validating the fus
        
        
        fs = fUS.compute_field_sign(ph_az, ph_ev, pw_az, pw_ev, filt_size=1); 
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
            
            fUS.export_tiffs(data_resample_at_stim, exportDir + sub_fn[:-4] + add_str + '.tiff', dims = [1,2,0])
            fUS.export_tiffs(trials_az.mean(axis = 0), exportDir + sub_fn[:-4] + add_str + '_az.tiff', dims = [1,2,0])
            fUS.export_tiffs(trials_el.mean(axis = 0), exportDir + sub_fn[:-4] + add_str + '_el.tiff', dims = [1,2,0])

    else:
        if do_save:
            exportDir = outDir + '/extras/'
            if not os.path.exists(exportDir):
                os.mkdir(exportDir)
            fUS.export_tiffs(data_resample_at_stim, exportDir + sub_fn[:-4]+'.tiff', dims = [1,2,0])