# Analysis example

import fUS_tools as fUS
import numpy as np
import scipy.signal as signal
##################################################################
# Load the fUS data
##################################################################

m = fUS.matloader(); # Instantiate mat object
m.loadmat_h5('/data/fUS_project/data/data_feb15/RT0_Acq_110550.mat'); m.summary()
data = m.data['Dop'].copy()
fUS.export_tiffs(data, outDir = m.filename[:-4]+'.tiff', dims = [2,1,0])
fUS.compute_fft(data, dims = [2,1,0]); # Will display naive phase maps...

##################################################################
# Clean the timestamps
##################################################################

# Loading in the timeline data
timeline_fn = '/data/fUS_project/data/data_feb15/anima1_timeline_02-15-2019_10-28.mat'
m = fUS.matloader();
m.loadmat_h5(timeline_fn); m.summary()
ts = m.data['timestamps'].copy()

# Compute the parsed version of 
#reload(fUS)
stim_list, times = fUS.parse_timestamps(m.data['data'][:,1], ts, min_isi = 0.01, interval_between_experiments = 20); # Separate the frames for each one...
fus_list, times = fUS.parse_timestamps(m.data['data'][:,0], ts, min_isi = 0.5, interval_between_experiments = 3); # Separate the frames for each one...

exp = 2; # Which experiment number
stim_ts = stim_list[exp]
fus_ts = fus_list[exp]

valid_fus = np.logical_and(fus_ts<stim_ts[-1], fus_ts > stim_ts[0]); # Which frames are within the stimulus presentation period


reload(fUS)

# Subsample the fUS experiment in time and space...
sub_fus = data[valid_fus, 30:100 , :]
fUS.compute_fft(sub_fus, dims = [2,1,0]); # Will display naive phase maps...



nT = sum(valid_fus)


plt.figure()
for i in range(1,4):
    signal_upsampled = signal.resample(sub_fus[:,40,20], nT*i)
    plt.plot(np.linspace(0, 1, signal_upsampled), signal_upsampled)
    