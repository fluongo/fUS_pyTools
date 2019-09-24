
from scipy import ndimage


all = []
for i in range(50):
    idx = np.random.permutation(10)
    ph_az, pw_az = fUS.compute_fft(trials_az[idx[:5], :, :, :].mean(axis=0), dims = [1, 2,0], doPlot = False)
    ph_az2, pw_az = fUS.compute_fft(trials_az[idx[5:], :, :, :].mean(axis=0), dims = [1, 2,0], doPlot = False)
    all.append(ph_az - ph_az2)
    plt.subplot(10,5,i+1); plt.imshow(ph_az-ph_az2, cmap = 'bwr')

plt.imshow(np.var(np.dstack(all), axis = -1), cmap = 'bwr')




# Median filter the image with kernal of size 3
ph_az_list_med = []
ph_ev_list_med = []

for az, ev in zip(ph_az_list, ph_ev_list):
    ph_az_list_med.append(ndimage.median_filter(az, 1))
    ph_ev_list_med.append(ndimage.median_filter(ev, 1))
    

d_sub_az = np.dstack(ph_az_list)[:20, :, :]
d_sub_ev = np.dstack(ph_ev_list)[:20, :, :]

plt.figure()
for i in range(5):
    plt.subplot(5,4,4*i+1)
    plt.imshow(d_sub_az[:, :, i], vmin =-pi, vmax = pi, cmap = 'hsv')
    plt.subplot(5,4,4*i+2)
    plt.imshow(np.expand_dims(d_sub_az[:, :, i].mean(axis=0), 0), vmin =-pi, vmax = pi, cmap = 'hsv')
    
    plt.subplot(5,4,4*i+3)
    plt.imshow(d_sub_ev[:, :, i], vmin =-pi, vmax = pi, cmap = 'hsv')
    plt.subplot(5,4,4*i+4)
    plt.imshow(np.expand_dims(d_sub_ev[:, :, i].mean(axis=0), 0), vmin =-pi, vmax = pi, cmap = 'hsv')
plt.colorbar()


# for i in range(5):
#     plt.subplot(5,1,i+1);
#     plt.imshow(ndimage.median_filter(ph_az_list[3], i+1), vmin = -pi, vmax = 0, cmap = 'jet')
from scipy.stats import pearsonr

# Code for computing a corelation map between a given stimulus and a time series

for zz in range(3,10):
    d = fUS.bin_2d(data_resample_at_stim, 2);
    nt, ny, nx = d.shape;
    corrs = np.zeros([ny, nx])
    # The stimulus to convolve

    stim_arr = np.zeros(50)
    stim_arr[-zz:] = 1;
    stim_arr = np.hstack([stim_arr for i in range(10)])
    for yy in range(ny):
        for xx in range(nx):
            corrs[yy, xx] = pearsonr(stim_arr, d[:, yy, xx])[0]
    plt.subplot(2,4, zz-2)
    plt.imshow(corrs); plt.title(str(zz))




out = fUS.gaussian_filter_xyt(data_resample_at_stim, sigmas = [2,2,2]);
test = fUS.perform_ica(out, num_comps = 2)
for i in range(2):
    plt.subplot(1,2,i+1); 
    plt.imshow(test[i, :, :])




data_old = m.data['Dop'].copy()
plt.subplot(1,3,1); plt.imshow(data_old[0, :, :])

data_reshape = data_old.reshape(1199, 64, 2, 26, 2).mean(-1).mean(2)
plt.subplot(1,3,2); plt.imshow(data_reshape[0, :, :])

plt.subplot(1,3,3);
reload(scana)
plt.imshow(scana.bin_2d(data_old, 2, dims = {'x':1, 'y':2, 't':0})[0, :, ])



all_az = np.zeros([26, 40])
for ii in range(10):
    ph_az, pw_az = scana.compute_fft(trials_az[np.random.randint(0, high = 9, size = 5) , :, :, :].mean(axis=0), dims = {'x':2, 'y':1, 't':0}, doPlot = False)
    all_az +=ph_az

plt.figure()
plt.imshow(ph_az/5., cmap = 'hsv')