#%% LOAD IN SOME DATA FROM 05_03 sessions
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

import seaborn as sns

#%%
fns = sorted(glob.glob('/data/fUS_project/data/data_may03/RT*.mat'))
timelines = len(fns)*['/data/fUS_project/data/data_may03/timeline_05-03-2019_13-01.mat']
n_fus = range(11)
n_stim = [0,1,2,3,5,6,7,8,9,10, 11]

all_exps = [[i, j,k, l, m] for i,j,k , l, m in zip(fns, timelines, range(len(fns)), n_fus, n_stim )]

n_trials = 10; fus_rate =  2;

# Experiment number 4 (5th experiment) of first timeline file is wrong
exp_list = []

# Export tiffs of all experiments
for ii in range(3):#range(len(all_exps)):
    animal_fn, timeline_fn, exp_number, exp_number_fus, exp_number_stim = all_exps[ii]

    # Load data
    m = scio.matloader(); # Instantiate mat object
    m.loadmat_h5(animal_fn); #m.summary()
    data_raw = m.data['Dop'].copy()

    # # Load timeline only on the first experiment
    if ii in [0]:
        timestamps_m = scio.matloader();
        timestamps_m.loadmat_h5(timeline_fn); #timestamps_m.summary()
        ts = timestamps_m.data['timestamps'].copy()

    # # Compute the parsed version of the timestamps
    stim_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,1], ts, min_isi = 0, interval_between_experiments = 20); # Separate the frames for each one...
    fus_list, times = scana.parse_timestamps(timestamps_m.data['data'][:,0], ts, min_isi = 0.2, interval_between_experiments = 3,  min_number_exp = 700); # Separate the frames for each one...

    stim_ts = stim_list[exp_number_stim];
    fus_ts = fus_list[exp_number_fus];

    if ii in [3]:
        fus_ts = fus_ts[1:]; # Chop off two
    if ii in [10]:
        fus_ts = fus_ts[2:]; # Chop off two3:

    # Find the corresponding closest stimulus frame
    fus_ts_stim = np.zeros_like(fus_ts)
    for ww in range(data_raw.shape[0]):
        fus_ts_stim[ww] = np.argmin(np.abs(stim_ts - fus_ts[ww]));
    
    curr_dict = {}
    curr_dict['raw'] = data_raw;
    curr_dict['stim_ts'] = stim_ts;
    curr_dict['fus_ts'] = fus_ts;
    curr_dict['fus_ts_stim'] = fus_ts_stim; # Rounded to nearest stimulus frame
    
    exp_list.append(curr_dict)

data_tmp = exp_list[1]['raw'][:, 10:30, 20:40].mean(axis=-1).mean(axis=-1);

sns.set_style('white')
plt.figure(figsize = [10, 5])
plt.subplot(1,2,1); plt.scatter(range(749), data_tmp, s=1)

X = np.mod(exp_list[1]['fus_ts_stim'], 900).reshape(-1,1)
y = data_tmp
plt.subplot(1,2,2); plt.scatter(X, y, s=1)



# for ii in range(11):
#     print(ii)
#     print(len(data_raw_list[ii]))
#     print(len(fus_list[ii]))


    # # Recompute at appropriate times



    

    # playbackHz = 1/np.mean(np.diff(stim_ts)); # Estimate the playback speed
    # newx = stim_ts[1:-1: int(30/fus_rate)]; # Interpolate the value at roughly the rate of the imaging, eg 500 ms so about 12 frames for 25 hz.....
    # data_resample_at_stim = scana.resample_xyt(data_raw, fus_ts, newx, dims = {'x':1, 'y':2, 't':0})




#%%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

# #  First the noiseless case
# nT = data_raw.shape[0];
# X = np.arange(nT).reshape(-1,1)
# X = X;

# # Observations
# y = data_raw[:, 50, 10]
# y = y-np.mean(y)

# # Train on a subset


# # Mesh the input space for evaluations of the real function, the prediction and its MSE
x = np.atleast_2d(np.linspace(0, 900, 1000)).T

# # Instantiate a Gaussian Process model

kernel = 1.0 * RBF(length_scale=0.01, length_scale_bounds=(1e-8, 1e2)) \
     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# # Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# # Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# # Plot the function, the prediction and the 95% confidence interval based on the MSE
plt.figure(figsize = [10, 2])
plt.plot(X, y, 'k.', markersize=2, label='Observations')
plt.plot(x, y_pred, 'y-', label='Prediction')

#%%

import GPy

kernel = GPy.kern.RBF(input_dim=1, variance = 1., lengthscale= 1.)
m = GPy.models.GPRegression(X, y.reshape(-1,1), kernel)

# the normal way
m.optimize(messages=True)
# with restarts to get better results
m.optimize_restarts(num_restarts = 20)