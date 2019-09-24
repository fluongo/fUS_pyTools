# Analysis example

import fUS_tools as fUS
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.misc import imresize
import os

##################################################################
# Load the fUS data
##################################################################

all_exps =     [   ('/data/fUS_project/data/data_mar01/RT0301_Acq_084323.mat','/data/fUS_project/data/data_mar01/timeline_03-01-2019_08-43_ANIMAL2_PLANE1_BOTHEYES.mat', 0, False, 6), 
                    ('/data/fUS_project/data/data_mar01/RT0301_Acq_085557.mat','/data/fUS_project/data/data_mar01/timeline_03-01-2019_08-43_ANIMAL2_PLANE1_BOTHEYES.mat', 1, True, 20), 
                    ('/data/fUS_project/data/data_mar01/RT0301_Acq_093437.mat','/data/fUS_project/data/data_mar01/timeline_03-01-2019_09-34_animal2_plane2_and_3_botheyes.mat', 0, True, 20), 
                    ('/data/fUS_project/data/data_mar01/RT0301_Acq_095250.mat', '/data/fUS_project/data/data_mar01/timeline_03-01-2019_09-34_animal2_plane2_and_3_botheyes.mat', 1, False, 6), 
                    ('/data/fUS_project/data/data_mar01/RT0301_Acq_100314.mat', '/data/fUS_project/data/data_mar01/timeline_03-01-2019_09-34_animal2_plane2_and_3_botheyes.mat', 2, False, 6),
                    ('/data/fUS_project/data/data_mar01/RT0301_Acq_101246.mat', '/data/fUS_project/data/data_mar01/timeline_03-01-2019_09-34_animal2_plane2_and_3_botheyes.mat', 3, True, 20),
                    
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_141038.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 0, True, 20), 
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_142816.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 1, False, 6) , 
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_143848.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 2, False, 6) ,
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_144856.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 3, True, 20) ,
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_151029.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 4, True, 20) ,
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_152843.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 5, False, 6) ,
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_155109.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 8, True, 20) ,
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_160950.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 9, False, 6) ,
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_162040.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 10, True, 20) , 
                    ('/data/fUS_project/data/data_mar01/animal2/RT0301_Acq_163820.mat', 
                    '/data/fUS_project/data/data_mar01/animal2/timeline_03-01-2019_14-11_final_animal_experiments.mat', 11, False, 6) ]

#for animal_fn, timeline_fn, exp in all_exps:

do_resample = True
do_save = False

pw_ev_list = []
pw_az_list = []
ph_ev_list = []
ph_az_list = []
names_list = []

# range(len(all_exps))
for ii in range(len(all_exps)):
    if ii in [0,1,2,3,4,5,7,8,11,13,15]:
        continue
    
    animal_fn, timeline_fn, exp, is_retinotopy, stim_rate = all_exps[ii]
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
    m = fUS.matloader();
    m.loadmat_h5(timeline_fn); m.summary()
    ts = m.data['timestamps'].copy()

    # Compute the parsed version of the timestamps
    stim_list, times = fUS.parse_timestamps(m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = fUS.parse_timestamps(m.data['data'][:,0], ts, min_isi = 0.1, interval_between_experiments = 3); # Separate the frames for each one...

    #exp = 1; # Which experiment number
    stim_ts = stim_list[exp]
    fus_ts = fus_list[exp]
    # Check that the number of fUS frames matches the experiment
    offset = len(fus_ts) - data.shape[0]
    if offset !=0: # Meaning fus timestamps and fUS dont match
        fus_ts = fus_ts[offset:] # truncate

    # Upsample the interpolation of each frame to account for stim ttl only getting sent once every 3rd frame
    stim_ts = fUS.updample_timestamps(stim_ts, 3)
    fus_rate = 2; # fus_rate in hz

    newx = stim_ts[1:-1:stim_rate/(fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....

    data_resample_at_stim = fUS.resample_xyt(data, fus_ts, newx, dims = [2,1,0])

    # Turn into DFF
    f0 = np.repeat(np.expand_dims( data_resample_at_stim.mean(axis=0), axis = 0), len(newx), axis = 0); # Makes a f0 for each timepoint
    data_resample_at_stim = (data_resample_at_stim -f0)/f0

    if is_retinotopy:
        n_trials = 10;
        nF = data_resample_at_stim.shape[0]/2/n_trials # Number of frames for each trial
        
        # Concatenate across trials and then output to tiff the original as well as the others
        trials_az = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])
        trials_el = np.zeros([n_trials, nF, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])

        for i in range(n_trials):
            # Write each trial
            trials_az[i, :, :, :] = data_resample_at_stim[i*nF:(i+1)*nF, :, :]
            trials_el[i, :, :, :] = data_resample_at_stim[n_trials*nF+i*nF:n_trials*nF+(i+1)*nF, :, :]

        ph_ev, pw_ev = fUS.compute_fft(trials_el.mean(axis=0), dims = [1, 2,0], doPlot = False)
        ph_az, pw_az = fUS.compute_fft(trials_az.mean(axis=0), dims = [1, 2,0], doPlot = False)
        
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