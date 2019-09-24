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
from sklearn import preprocessing

from PIL import Image

def imresize(im, sz):
    return np.array(Image.fromarray(im).resize(sz))

dir_aug26 = '/data/fUS_project/data/data_aug26/extras/'
dir_aug27 = '/data/fUS_project/data/data_aug27/extras/'

anatomical_fn = '/data/morpheus/raw_data/fUS_rig/08_16_19_TS_waluigi_100um_step.tiff'
anatomical_data = scio.load_multitiff(anatomical_fn)

# Load the raw data
print('loading day 1')
day1_retin = scio.load_multitiff(dir_aug26 + 'RETINOTOPY_cumulative_trials_RAW.tiff', first_n=100)
print('loading day 2')
day2_retin = scio.load_multitiff(dir_aug27 + 'RETINOTOPY_cumulative_trials_RAW.tiff', first_n=100)

# Parse them into slices // Each spaced by 1500uM
day1 = [day1_retin.mean(axis = -1)[:, kk*128:(kk+1)*128] for kk in range(5)]
day2 = [day2_retin.mean(axis = -1)[:, kk*128:(kk+1)*128] for kk in range(5)]

#%% Using the eyeball test
# I think Day 2 slice 1 corresponds to slice 1 from the anatomical, whereas Day 1 slice 1 is ~750uM back from slice 1



for offset in range(7):
    nY = 250
    plt.figure(figsize = [20, 5]); plt.suptitle('Offset: %d' % offset)
    for ii in range(4):
        plt.subplot(2,4,ii+1); plt.imshow(imresize(anatomical_data[:200, :, offset+15*ii], [128, 52]))
        plt.subplot(2,4,ii+5); plt.imshow(preprocessing.scale(day1[ii]))


#%% USING PYSTACK REG

from pystackreg import StackReg
from scipy import stats

do_plot = False

for kk in range(5):
    all_corrs = []
    for ii in range(71):
        ref = np.array(Image.fromarray(anatomical_data[:200, :, ii]).resize([128, 52]))
        ref = preprocessing.scale(ref)
        mov = day1[kk]
        mov = preprocessing.scale(mov)
        
        # #Scaled Rotation transformation
        sr = StackReg(StackReg.SCALED_ROTATION)
        out_sca = sr.register_transform(ref, mov)
        if do_plot:
            plt.figure(figsize = [20, 5])
            plt.subplot(1,3,1); plt.imshow(ref)
            plt.subplot(1,3,2); plt.imshow(mov)
            plt.subplot(1,3,3); plt.imshow(out_sca)

        corr_val = stats.pearsonr(out_sca.ravel(), ref.ravel())[0]
        all_corrs.append(corr_val)
    plt.plot(all_corrs/np.max(all_corrs), label = str(kk))

#%%
###################################
#######  AIRLAB
###################################

#fixed_image = al.read_image_as_tensor("./data/affine_test_image_2d_fixed.png", dtype=dtype, device=device)
#moving_image = al.read_image_as_tensor("./data/affine_test_image_2d_moving.png", dtype=dtype, device=device)


import sys
import os
import time

import matplotlib.pyplot as plt
import torch as th
import airlab as al

start = time.time()

# set the used data type
dtype = th.float32

# set the device for the computaion to CPU
#device = th.device("cpu")
device = th.device("cuda:1")

# load the image data and normalize to [0, 1]
#im1 = preprocessing.scale(day2[1]);
#im2 = preprocessing.scale(imresize(anatomical_data[:200, :, 14], [128, 52]));

#for pp, qq in zip([0,1,2,3], [0,14,29,44]):
im2 = preprocessing.scale(day2[0]);
im1 = preprocessing.scale(imresize(anatomical_data[30:200, :, 0], [128, 52]));

# FIXED IMAGE IS THE ANATOMICAL
tensor_image = th.from_numpy(im1).to(dtype=dtype, device=device)
fixed_image = al.utils.image.Image(tensor_image, im1.shape, 0, (0, 0)); # Make anatomical fixed

# WARPING THE ATLAS ONTO THAT IMAGE.....
tensor_image = th.from_numpy(im2).to(dtype=dtype, device=device)
moving_image = al.utils.image.Image(tensor_image, im2.shape, 0, (0, 0)); # Warp the retin onto this one

fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

# convert intensities so that the object intensities are 1 and the background 0. This is important in order to
# calculate the center of mass of the object
fixed_image.image = 1 - fixed_image.image
moving_image.image = 1 - moving_image.image

# create pairwise registration object
registration = al.PairwiseRegistration()

# choose the affine transformation model
#transformation = al.transformation.pairwise.SimilarityTransformation(moving_image, opt_cm=True)
transformation = al.transformation.pairwise.AffineTransformation(moving_image, opt_cm=True)
#transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=True)

# initialize the translation with the center of mass of the fixed image
transformation.init_translation(fixed_image)
registration.set_transformation(transformation)

# choose the Mean Squared Error as image loss
image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
registration.set_image_loss([image_loss])

# choose the Adam optimizer to minimize the objective
optimizer = th.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)
registration.set_optimizer(optimizer)
registration.set_number_of_iterations(100)

# start the registration
registration.start(EarlyStopping=False)

# set the intensities back to the original for the visualisation
fixed_image.image = 1 - fixed_image.image
moving_image.image = 1 - moving_image.image

# warp the moving image with the final transformation result
displacement = transformation.get_displacement()
warped_image = al.transformation.utils.warp_image(moving_image, displacement)

end = time.time()

print("=================================================================")

print("Registration done in:", end - start, "s")
print("Result parameters:")
transformation.print()

plt.figure(figsize = [20, 10])
# plot the results
plt.subplot(131)
plt.imshow(fixed_image.numpy(), cmap='gray')
plt.title('Fixed Image')

plt.subplot(132)
plt.imshow(moving_image.numpy(), cmap='gray')
plt.title('Moving Image')

plt.subplot(133)
plt.imshow(warped_image.numpy(), cmap='gray')
plt.title('Warped Moving Image')

plt.figure(figsize = [20, 5])
for ii in range(2):
    plt.subplot(1,2,ii+1)
    plt.imshow(displacement.cpu()[:, : ,ii]); plt.colorbar()

#%%###################################################
###################  ITERATE THROUGH
###############################################
import scipy.stats as stats


def compute_mse_registration(im_fixed, im_moving, dtype=th.float32, device = th.device("cuda:1"), doPlot = False):
    # Takes as input a fixed image and a moving image and then return the displacement and 
    # warped images
    
    # FIXED IMAGE IS USUALLY ANATOMICAL
    tensor_image = th.from_numpy(preprocessing.scale(im_fixed)).to(dtype=dtype, device=device)
    fixed_image = al.utils.image.Image(tensor_image, im_fixed.shape, 0, (0, 0));

    # MOVING IMAGE TO BE WARPED
    tensor_image = th.from_numpy(preprocessing.scale(im_moving)).to(dtype=dtype, device=device)
    moving_image = al.utils.image.Image(tensor_image, im_moving.shape, 0, (0, 0)); # Warp the retin onto this one

    # Normalize and flip BG
    fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)
    fixed_image.image = 1 - fixed_image.image; moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    #transformation = al.transformation.pairwise.SimilarityTransformation(moving_image, opt_cm=True)
    transformation = al.transformation.pairwise.AffineTransformation(moving_image, opt_cm=True)
    #transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=True)

    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)
    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.01, amsgrad=True)
    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(100)

    # start the registration
    registration.start(EarlyStopping=False)

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)
    # Print transform
    transformation.print()

    if doPlot:
        plt.figure(figsize = [20, 4])
        # plot the results
        plt.subplot(131)
        plt.imshow(fixed_image.numpy(), cmap='gray')
        plt.title('Fixed Image')

        plt.subplot(132)
        plt.imshow(moving_image.numpy(), cmap='gray')
        plt.title('Moving Image')

        plt.subplot(133)
        plt.imshow(warped_image.numpy(), cmap='gray')
        plt.title('Warped Moving Image')

        # plt.figure(figsize = [20, 5])
        # for ii in range(2):
        #     plt.subplot(1,2,ii+1)
        #     plt.imshow(displacement.cpu()[:, : ,ii]); plt.colorbar()

    return warped_image.numpy(), displacement, transformation

# anatomical_resize = np.zeros([52, 128, 71])
# for xx in range(71):
#     anatomical_resize[:, :, xx] = imresize(anatomical_data[30:200, :, xx], [128, 52]);

# anatomical_smoothed = median_filter(anatomical_resize, size = 5)



for offset in range(0,10,2):
    for xx in range(3):
        im_moving = day1[xx+1]
        im_fixed = anatomical_resize[:, :, 15*xx + offset];
        im_warped, disp, trans = compute_mse_registration(im_fixed, im_moving, dtype=th.float32, device = th.device("cuda:1"), doPlot = True)
        stats.pearsonr(im_fixed.ravel(), im_warped.ravel())[0]
        plt.suptitle('Offset %d and slice %d' % (offset, 15*xx + offset))

# TOTAL ALIGNMENTS FOR WALUIGI
# 8-26: Day 1 retinotopy    {5 slices 1.5mm spacing}    is offset -7 (slice 8 seems to match 2nd slice)
# 8-27: Day 2 retinotopy    {5 slices 1.5mm spacing}    is offset 0  (slice 0 seems to match 1st slice)
# 8-29: Full-field          {14 slices, 750 um spacing} is offset 
# 9-03: Obj_Scram           {3 slices, 750 um spacing}  is offset ~40
# 9-03: Full field          {3 slices, 750 um spacing}  is offset ~40



#%% Apply a median filter to each image


