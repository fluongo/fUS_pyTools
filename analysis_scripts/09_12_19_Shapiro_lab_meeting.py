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
from matplotlib.patches import Rectangle

import seaborn as sns
from PIL import Image

sns.set_style('white')
sns.set_style("ticks")

dir_aug26 = '/data/fUS_project/data/data_aug26/extras/'
dir_aug27 = '/data/fUS_project/data/data_aug27/extras/'

d1 = np.load(dir_aug26 + 'data_processed.npy')
d2 = np.load(dir_aug27 + 'data_processed.npy')

comb1 = np.load(dir_aug26 + 'interpolated_trial_average_aug26.npy')
comb2 = np.load(dir_aug27 + 'interpolated_trial_average_aug27.npy')

nX = 128
nY = 52

combined_full = np.zeros([1200, 4*52, 10*nX])
for i in range(10):
    if np.mod(i, 2) == 0:
        combined_full[:, :, nX*i:nX*(i+1)] = comb1[:, :, int((i/2))*nX : int(((i/2 + 1))*nX)]
    else:
        combined_full[:, :, nX*i:nX*(i+1)] = comb2[:, :, int(i/2)*nX : (int(i/2) + 1)*nX]

f0 = combined_full.mean(axis = 0)
for kk in range(1200):
    combined_full[kk, :, :] = combined_full[kk, :, :] - f0;

# plt.figure(figsize = [12, 4])
# plt.imshow(combined_full[600, :52, 128*4:128*8])
# plt.grid(True)


#% Combined for each trial type
combined_resamp_stim = np.zeros([10, 80, 4*52, 10*nX])
for sl in range(10):
    for cnt, stim_type in enumerate(['az_LR', 'az_RL', 'ev_UD', 'ev_DU']):
        if np.mod(sl, 2) == 0:
            tmp = d1[int(sl/2)]['trials_all'][stim_type]
        else:
            tmp = d2[int(sl/2)]['trials_all'][stim_type]
        for trN in range(10):
            tmp2    = tmp[trN*80:(trN+1)*80, :, :].copy();
            f0      = tmp2[:6, :, :].mean(axis = 0)
            tmp2    = (tmp2-f0)/f0 # DFF
            combined_resamp_stim[trN, :, cnt*52:(cnt+1)*52, sl*128:(sl+1)*128] = tmp2

plt.figure(figsize = [20, 5])
plt.imshow(combined_resamp_stim.mean(axis=0)[40, :, :])

# Load anatimical
raw_fn = '/data/fUS_project/data/data_aug27/extras/RETINOTOPY_cumulative_trials_RAW_linear_interp_of_artifacts.tiff'
raw_data = scio.load_multitiff(raw_fn, first_n=100)

shapiro_out_dir = '/data/fUS_project/visualization/shapiro_lab_meeting'

#%% First make progressive grids

def gen_masks(step=8, n_masks=7, y_offset = 5, x_offset=5, im = np.zeros([52, 128]), vlim = [0, 1.5e10]):
    # Generates N masks with size step spaced over a certain range        
    plt.imshow(im, vmin = vlim[0], vmax = vlim[1], cmap = 'binary'); #plt.colorbar(); 
    plt.axis('off')
    masks = []
    colors = plt.cm.Reds(np.linspace(0.2,1,n_masks))

    r = n_masks-1; # For computing Y spacing which is overlapping
    for i in range(n_masks):
        #reg_mask = np.zeros([52, 128])
        y_start = int(y_offset + (r-i)*20/r)
        y_end   = int(y_start + step);
        x_start = int(x_offset + i*step);
        x_end   = int(x_start + step)
        curr_rect = Rectangle((x_start, y_start), step, step, fill = None, color = colors[i])
        plt.gca().add_patch(curr_rect)
        masks.append((y_start, y_end, x_start, x_end, curr_rect, colors[i]))
        print(y_start, y_end, x_start, x_end)
    return masks, colors


im_in = raw_data[:, 128:256, :].mean(axis=-1)
masks, colors = gen_masks(step=8, n_masks=7, y_offset = 5, x_offset=5, im = im_in, vlim =[0, 1.5e10])


#%% Now go through and for each one extract a timecourse from the first type of trial
n_masks = len(masks)

timecourse = np.zeros([n_masks, 10, 4, 1200])
for sl in range(10):
    for exp in range(4):
        for ii in range(n_masks):
            y0, y1, x0, x1, _, c = masks[ii]
            timecourse[ii, sl, exp, :] = combined_full[:, 52*exp+y0:52*exp+y1, 128*sl+x0:128*sl+x1].mean(-1).mean(-1)

# Plots
plt.figure(figsize = [20, 40])
for sl in range(10):
    for exp in range(4):
        plt.subplot(10,4,4*sl+exp+1)
        for ii in range(n_masks):
            plt.plot(timecourse[ii, sl, exp, :], color = colors[ii])
            sns.despine()
            plt.ylim([-0.2, 1])

#%% Now with the resamp at stim

comb_stim = combined_resamp_stim.mean(axis=0)

timecourse2 = np.zeros([n_masks, 10, 4, 80])
for sl in range(10):
    for exp in range(4):
        for ii in range(n_masks):
            y0, y1, x0, x1, _, c = masks[ii]
            timecourse2[ii, sl, exp, :] = comb_stim[:, 52*exp+y0:52*exp+y1, 128*sl+x0:128*sl+x1].mean(-1).mean(-1)

# Plots
plt.figure(figsize = [20, 40])
for sl in range(10):
    for exp in range(4):
        plt.subplot(10,4,4*sl+exp+1)
        for ii in range(n_masks):
            plt.plot(timecourse2[ii, sl, exp, :], color = colors[ii])


#%% Now with individual trials but only do this for a single slice.....
#%% TIMECOURSES OF RETINOTOPY
#######################################
#######################################
#######################################

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(shapiro_out_dir + '/slices_timecourse_from_retinotopy.pdf') as pdf_pointer:
    im_in = raw_data[:, 128:256, :].mean(axis=-1)
    masks, colors = gen_masks(step=8, n_masks=7, y_offset = 5, x_offset=5, im = im_in, vlim =[0, 1.5e10])
    pdf_pointer.savefig(plt.gcf())
    for slice_no in range(10):
        comb_stim = combined_resamp_stim[:, :, :, 128*slice_no:128*(slice_no+1)]
        timecourse_trials = np.zeros([n_masks, 4, 10, 80]) # Now the 10 is nTrials instead of nSlices
        for exp in range(4):
            for ii in range(n_masks):
                y0, y1, x0, x1, _, c = masks[ii]
                tmp = comb_stim[:, :, 52*exp+y0:52*exp+y1, x0:x1].mean(axis=-1).mean(axis=-1)
                timecourse_trials[ii, exp, :, :] = tmp

        # # Plots
        plt.figure(figsize = [20, 10]); plt.suptitle('slice %d'% slice_no)
        for ii in range(n_masks):
            for exp in range(4):
                plt.subplot(4,n_masks, exp*n_masks+ii+1)
                sns.tsplot(timecourse_trials[ii, exp, :, :], time = np.linspace(0, 40, 80), color = colors[ii])
                sns.despine(); plt.ylim([-0.2, 1])
                if exp == 3:
                    plt.xlabel('time (sec)')
                if ii == 0:
                    plt.ylabel('% signal change')

        pdf_pointer.savefig(plt.gcf(), facecolor= 'white')  # or you can pass a Figure object to pdf.savefig
        plt.close()           

#%%
###################################
####################################
#% FULL FIELD% 
# NOW DO FULL FIELD ANALYSIS

data_fn = '/data/fUS_project/data/data_aug29/extras/combined_single_trial.npy'
data_full_fn = '/data/fUS_project/data/data_aug29/extras/data_processed_fullfield_aug29.npy'

test = np.load(data_fn)
d_FF = np.load(data_full_fn)
nSlices = 14
comb_stim = test


rep_im = d_FF[7]['data_raw_fix'].mean(0).T
masks, colors = gen_masks(step=8, n_masks=7, y_offset = 10, x_offset=10, im = rep_im, vlim =[0, 8e9])

#%%
# 40 frames per trial, 10 trials
#d_FF[0]['data_resample_at_stim'].reshape([40, 10, 52, 128])


timecourseFF = np.zeros([n_masks, nSlices, 10, 40])
for sl in range(nSlices):
    for ii in range(n_masks):
        y0, y1, x0, x1, _, c = masks[ii]
        tmp = d_FF[sl]['data_resample_at_stim'].reshape([10, 40, 52, 128])
        tmp = tmp[:, :, y0:y1, x0:x1].mean(-1).mean(-1)
        f0 = tmp[:, :5].mean(-1)
        tmp = (tmp.T - f0)/f0
        timecourseFF[ii, sl, :, :] =tmp.T;


#         timecourseFF[ii, sl, :, :] = comb_stim[:, y0:y1, 128*sl+x0:128*sl+x1].mean(-1).mean(-1)
#         timecourseFF[ii, sl, :, :] = timecourseFF[ii, sl, :] - timecourseFF[ii, sl, :10].mean()

# # Plots
plt.figure(figsize = [20, 40])
for sl in range(nSlices):
    for ii in range(n_masks):
        plt.subplot(14,7,7*sl+ii+1)
        sns.tsplot(timecourseFF[ii, sl, :, :], time = np.linspace(0, 20, 40), color = colors[ii])
        sns.despine(); plt.ylim([-0.2, 1])
        if sl == 13:
            plt.xlabel('time (sec)')
        if ii == 0:
            plt.ylabel('% signal change')

with PdfPages(shapiro_out_dir + '/FULLFIELD_perslice_timecourse.pdf') as pdf_pointer:
    pdf_pointer.savefig(plt.gcf(), facecolor= 'white')
    plt.figure()
    masks, colors = gen_masks(step=8, n_masks=7, y_offset = 10, x_offset=10, im = rep_im, vlim =[0, 8e9])
    pdf_pointer.savefig(plt.gcf(), facecolor= 'white')

###################################3
#%% NOW DO OBJECTS VS SCRAMBLED
###################################3###################################3
###################################3###################################3
###################################3###################################3


data_fn = '/data/fUS_project/data/data_sep03/extras_fullfield/data_processed_fullfield_sep13.npy'
d_OS_FF = np.load(data_fn)

data_fn = '/data/fUS_project/data/data_sep03/extras_object_scram/data_processed_objects_scram_sep03.npy'
d_OS = np.load(data_fn)

rep_im = d_OS_FF[2]['data_raw_fix'].mean(0).T
masks, colors = gen_masks(step=8, n_masks=7, y_offset = 10, x_offset=10, im = rep_im, vlim =[0, 8e9])
n_masks = 7


#%% Plot the full field

nSlices = 3
timecourseFF = np.zeros([n_masks, nSlices, 10, 40])
for sl in range(nSlices):
    for ii in range(n_masks):
        y0, y1, x0, x1, _, c = masks[ii]
        # Cut out first 30 seconds
        tmp = d_OS_FF[sl]['data_resample_at_stim'][60:, :, :] # Make suyre it is the one from obj-scram
        tmp = tmp.reshape([10, 40, 52, 128])
        tmp = tmp[:, :, y0:y1, x0:x1].mean(-1).mean(-1)
        f0 = tmp[:, :5].mean(-1)
        tmp = (tmp.T - f0)/f0
        timecourseFF[ii, sl, :, :] =tmp.T;
#% Plot the obj and scram conditions
timecourseOS = np.zeros([n_masks, nSlices, 10, 80])
for sl in range(nSlices):
    for ii in range(n_masks):
        y0, y1, x0, x1, _, c = masks[ii]
        # Cut out first 30 seconds
        tmp = d_OS[sl]['data_resample_at_stim'] # Make suyre it is the one from obj-scram
        tmp = tmp.reshape([10, 80, 52, 128])
        tmp = tmp[:, :, y0:y1, x0:x1].mean(-1).mean(-1)
        f0 = tmp[:, :5].mean(-1)
        tmp = (tmp.T - f0)/f0
        timecourseOS[ii, sl, :, :] =tmp.T;

#%%

with PdfPages(shapiro_out_dir + '/OBJ_SCRAM_3slices_sweepMasks.pdf') as pdf_pointer:
    rep_im = d_OS_FF[2]['data_raw_fix'].mean(0).T
    for current_y in [3,6,9,12]:
        plt.figure()
        masks, colors = gen_masks(step=8, n_masks=7, y_offset = current_y, x_offset=10, im = rep_im, vlim =[0, 8e9])
        plt.title('Y offset: %d and size: %d' % (current_y, 8))
        pdf_pointer.savefig(plt.gcf(), facecolor= 'white')

        # # Plots
        plt.figure(figsize = [20, 40])
        for sl in range(nSlices):
            for ii in range(n_masks):
                plt.subplot(14,7,7*sl+ii+1)
                sns.tsplot(timecourseFF[ii, sl, :, :], time = np.linspace(0, 20, 40), color = colors[ii])
                sns.despine(); plt.ylim([-0.2, 0.8])
                if sl == 2:
                    plt.xlabel('time (sec)')
                if ii == 0:
                    plt.ylabel('% signal change')
        #plt.suptitle('3 slices full field')
        pdf_pointer.savefig(plt.gcf(), facecolor= 'white')

        # # Plots
        plt.figure(figsize = [20, 40])
        for sl in range(nSlices):
            for ii in range(n_masks):
                plt.subplot(14,7,7*sl+ii+1)
                sns.tsplot(timecourseOS[ii, sl, :, :], time = np.linspace(0, 40, 80), color = colors[ii])
                sns.despine(); plt.ylim([-0.2, 0.8])
                if sl == 2:
                    plt.xlabel('time (sec)')
                if ii == 0:
                    plt.ylabel('% signal change')

        #plt.suptitle('objects_v_scrambled')
        pdf_pointer.savefig(plt.gcf(), facecolor= 'white')

        plt.figure(figsize = [20, 40])
        for sl in range(nSlices):
            for ii in range(n_masks):
                plt.subplot(14,7,7*sl+ii+1)
                sns.tsplot(timecourseOS[ii, sl, :, :40], time = np.linspace(0, 20, 40), color = [0.5, 0, 0])
                sns.tsplot(timecourseOS[ii, sl, :, 40:], time = np.linspace(0, 20, 40), color = [0, 0.5, 0])
                
                sns.despine(); plt.ylim([-0.2, 0.8])
                if sl == 2:
                    plt.xlabel('time (sec)')
                if ii == 0:
                    plt.ylabel('% signal change')
        pdf_pointer.savefig(plt.gcf(), facecolor= 'white')
        
        plt.figure(figsize = [20, 40])
        for sl in range(nSlices):
            for ii in range(n_masks):
                plt.subplot(14,7,7*sl+ii+1)
                sns.tsplot(timecourseOS[ii, sl, :, :40], time = np.linspace(0, 20, 40), color = [0.5, 0, 0])
                sns.tsplot(timecourseOS[ii, sl, :, 40:], time = np.linspace(0, 20, 40), color = [0, 0.5, 0])
                
                sns.despine(); plt.ylim([-0.2, 0.8])
                if sl == 2:
                    plt.xlabel('time (sec)')
                if ii == 0:
                    plt.ylabel('% signal change')

