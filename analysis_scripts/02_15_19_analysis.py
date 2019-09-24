# Analysis example

import fUS_tools as fUS
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


##################################################################
# Load the fUS data
##################################################################

all_exps =     [   ('/data/fUS_project/data/data_feb15/RT0_Acq_104833.mat', '/data/fUS_project/data/data_feb15/anima1_timeline_02-15-2019_10-28.mat', 1), 
                    ('/data/fUS_project/data/data_feb15/RT0_Acq_110550.mat', '/data/fUS_project/data/data_feb15/anima1_timeline_02-15-2019_10-28.mat', 2), 
                    ('/data/fUS_project/data/data_feb15/RT0_Acq_112148.mat', '/data/fUS_project/data/data_feb15/anima1_timeline_02-15-2019_10-28.mat', 3),   
                    ('/data/fUS_project/data/data_feb15/RT0_Acq_120437.mat', '/data/fUS_project/data/data_feb15/animal2_timeline_02-15-2019_11-47.mat', 1),
                    ('/data/fUS_project/data/data_feb15/RT0_Acq_121935.mat', '/data/fUS_project/data/data_feb15/animal2_timeline_02-15-2019_11-47.mat', 2) ]

for animal_fn, timeline_fn, exp in all_exps:

    m = fUS.matloader(); # Instantiate mat object
    #animal_fn = '/data/fUS_project/data/data_feb15/RT0_Acq_104833.mat'
    m.loadmat_h5(animal_fn); m.summary()
    data = m.data['Dop'].copy()

    ##################################################################
    # Clean the timestamps
    ##################################################################

    # Loading in the timeline data
    #timeline_fn = '/data/fUS_project/data/data_feb15/animal2_timeline_02-15-2019_11-47.mat'
    m = fUS.matloader();
    m.loadmat_h5(timeline_fn); m.summary()
    ts = m.data['timestamps'].copy()

    # Compute the parsed version of the timestamps
    stim_list, times = fUS.parse_timestamps(m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = fUS.parse_timestamps(m.data['data'][:,0], ts, min_isi = 0.5, interval_between_experiments = 3); # Separate the frames for each one...

    #exp = 1; # Which experiment number
    stim_ts = stim_list[exp]
    fus_ts = fus_list[exp]

    newx = stim_ts[1:-1:20]; # Interpolate the value at each frame.....
    data_resample_at_stim = fUS.resample_xyt(data, fus_ts, newx, dims = [2,1,0])

    # Concatenate across trials and then output to tiff the original as well as the others
    trials_az = np.zeros([30, 10, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])
    trials_el = np.zeros([30, 10, data_resample_at_stim.shape[1], data_resample_at_stim.shape[2]])

    for i in range(10):
        trials_az[:, i, :, :] = data_resample_at_stim[i*30:(i+1)*30, :, :]
        trials_el[:, i, :, :] = data_resample_at_stim[300+i*30:300+(i+1)*30, :, :]
        
    reload(fUS)
    ph_ev, pw_ev = fUS.compute_fft(trials_el.mean(axis=1), dims = [1, 2,0], doPlot = False)
    ph_az, pw_az = fUS.compute_fft(trials_az.mean(axis=1), dims = [1, 2,0], doPlot = False)
    fs = fUS.compute_field_sign(ph_az, ph_ev, filt_size=3); 
    # Save the field sign
    plt.title(animal_fn); plt.savefig(animal_fn[:-4]+'.pdf')


# fUS.export_tiffs(data_resample_at_stim, outDir = animal_fn[:-4]+'.tiff', dims = [1,2,0])
# fUS.export_tiffs(trials_az.mean(axis = 1), outDir = animal_fn[:-4]+'_az.tiff', dims = [1,2,0])
# fUS.export_tiffs(trials_el.mean(axis = 1), outDir = animal_fn[:-4]+'_el.tiff', dims = [1,2,0])
