import numpy as np
import sys
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tifffile
import scipy
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors



# TODO: Make something that imports the data from individual tiffs
# Make something that imports the timeline files from mat
# Make something that computes the fourier and plots the retinoptic maps
# Maybe reuse the allen institute data...

class matloader:

    def __init__(self):
        self.data = {}
        self.filename = None;
        self.loaded = False


    def _add_dtype_name(self, f, name):
        """Keep track of all dtypes and names in the HDF5 file using it."""
        global dtypes
        dtype = f.dtype
        if dtypes.has_key(dtype.name):
            dtypes[dtype.name].add(name)
        else:
            dtypes[dtype.name] = set([name])
        return

    def _string(self, seq):
        """Convert a sequence of integers into a single string."""
        out = ''.join([chr(a) for a in seq])
        return out

    def _recursive_dict(self, f, root=None, name='root'):
        """This function recursively navigates the HDF5 structure from
        node 'f' and tries to unpack the data structure by guessing their
        content from dtype, shape etc.. It returns a dictionary of
        strings, arrays and some leftovers. 'root' is the root node of the
        HDF5 structure, i.e. what h5py.File() returns.
        Note that this function works well on the Matlab7.3 datasets on
        which it was tested, but in general it might be wrong and it might
        crash. The motivation is that it has to guess the content of
        substructures so it might fail. One source of headache seems to be
        Matlab7.3 format that represents strings as array of 'uint16' so
        not using the string datatype. For this reason it is not possible
        to discriminate strings from arrays of integers without using
        heuristics.
        """
        
        global global_excluded_variables

        if root is None: root = f
        if hasattr(f, 'keys'):
            a = dict(f)
            if u'#refs#' in a.keys(): # we don't want to keep this
                del(a[u'#refs#'])
            for k in a.keys():
                #print k
                # FL: I added this in to skip moviedata to make loading faster...
                if k in global_excluded_variables:
                    continue
                else:
                    a[k] = self._recursive_dict(f[k], root, name=name+'->'+k)
            return a
        
        elif hasattr(f, 'shape'):
            if f.dtype.name not in ['object', 'uint16']: # this is a numpy array
                # Check shape to assess whether it can fit in memory
                # or not. If not recast to a smaller dtype!
                self._add_dtype_name(f, name)
                dtype = f.dtype
                if (np.prod(f.shape)*f.dtype.itemsize) > 2e9:
                    print("WARNING: The array", name, "requires > 2Gb")
                    if f.dtype.char=='d':
                        print("\t Recasting", dtype, "to float32")
                        dtype = np.float32
    #				else:
    #					raise MemoryError
                return np.array(f, dtype=dtype).squeeze()
            elif f.dtype.name in ['uint16']: # this may be a string for Matlab
                self._add_dtype_name(f, name)
                try:
                    return self._string(f)
                except ValueError: # it wasn't...
                    print("WARNING:", name, ":")
                    print("\t", f)
                    print("\t CONVERSION TO STRING FAILED, USING ARRAY!")
                    tmp = np.array(f).squeeze()
                    print("\t", tmp)
                    return tmp
                pass
            elif f.dtype.name=='object': # this is a 2D array of HDF5 object references or just objects
                self._add_dtype_name(f, name)
                container = []
                for i in range(f.shape[0]):
                    for j in range(f.shape[1]):
                        if str(f[i][j])=='<HDF5 object reference>': # reference follow it:
                            container.append(self._recursive_dict(root[f[i][j]], root, name=name))
                        else:
                            container.append(np.array(f[i][j]).squeeze())
                try:
                    return np.array(container).squeeze()
                except ValueError:
                    print("WARNING:", name, ":")
                    print("\t", container)
                    print("\t CANNOT CONVERT INTO NON-OBJECT ARRAY")
                    return np.array(container, dtype=np.object).squeeze()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return

    def loadmat_h5(self, filename, excluded_variables = []):
        '''
        'Master code for loading int the .mat filename'
        Can not load certain variables by including a list under the keyword argument, excluded)_variables, useful when certain variables are very large and take too long to load
        '''

        print('Loading mat data from %s' %filename)

        f = h5py.File(filename, mode='r')
        global dtypes, global_excluded_variables
        global_excluded_variables = excluded_variables
        dtypes = {}; # Reset dtypes each time...
        
        #print outputName
        
        # Output everything from the structure
        self.data = self._recursive_dict(f)
        self.filename = filename
        self.loaded = True
        print('Finished Loading.')


    def summary(self):
        '''Lists a summary of the given outputs'''
        #print("+++++++++++++++++++++++++++++++++++++++")
        #%print("Variable name                     Type")
        print('\n'); print('Summary of loaded data'); print('\n')
        print('{:<40s}{:<30s}{:<0s}'.format('Variable name','Type', 'Shape'))
        print('+'*100);
        for x in self.data.keys():

            if isinstance(self.data[x], np.ndarray):
                print('{:<40s}{:<30s}{:<0s}'.format(x,type(self.data[x]),  str(self.data[x].shape) ) )
            else:
                print('{:<40s}{:<30s}'.format(x,type(self.data[x]) ) )

            #print('%s   %s' % (x, type(self.data[x]) ) )

def export_tiffs(data, outDir='', dims = [0,1,2]):
    ''' dims takes as inpute the x y t dimensions'''

    if outDir == '':
        raise NotADirectoryError
    else:
        tifffile.imsave(outDir, np.transpose(data, [dims[2], dims[0], dims[1]]).astype('single') )

def updample_timestamps(ts, factor):
    ''' In the event that the data is only ttl every 3rd frame, you can re-upsample to each frame'''
    return np.interp(np.linspace(0, 1, len(ts)*factor), np.linspace(0, 1, len(ts)), ts)

# def compute_pval_map(data, dims = [0,1,2]):
#     ''' Code to compute the correlation map between a given pixel timeseries and a stimulus'''


def resample_xyt(data, oldx, newx, dims = [0,1,2]):
    '''Code will resample every pixel in the data at the newx, given the oldx and the dims of teh array'''
    data_reshaped = np.transpose(data, [dims[2], dims[0], dims[1]]); # Transpose to y, x, t
    sz = data_reshaped.shape
    data_resampled = np.zeros([len(newx), sz[1], sz[2]])

    print(data_reshaped.shape)
    print(data_resampled.shape)

    for xx in range(sz[1]):
        for yy in range(sz[2]):
            data_resampled[:,xx,yy] = np.interp(newx, oldx, data_reshaped[:,xx, yy])

    return data_resampled

def compute_fft(data , dims = [0,1,2], doPlot = False):
    ''' Compute the fourier transform and plots the first 100 power and phase maps'''
    reshaped = np.transpose(data, [dims[2], dims[0], dims[1]]); # Transpose to t, y, x
    out = np.fft.fft(reshaped, axis = 0); # take along time
    
    power_map = np.abs(out[0, :, :])
    phase_map = np.angle(out[1, :, :])

    if doPlot:
        plt.figure(figsize = [10, 4])
        plt.subplot(1,2,1); plt.imshow(power_map, cmap = 'binary'); plt.title('power'); plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(phase_map, cmap = 'gist_rainbow'); plt.title('phase'); plt.colorbar()

    return phase_map, power_map

def bin_2d(data, factor):
    '''Takes as input a 3d array T x w x h and bins by a scalar factor
    '''
    t, y, x = data.shape;
    if (y % factor != 0) or (x % factor != 0):
        if (y % factor != 0) and (x % factor != 0):
            data = data[:, :-(y % factor) , : -(x % factor)]
        elif (y % factor != 0):
            data = data[:, :-(y % factor) , :]
        elif (x % factor != 0):
            data = data[:, : , : -(x % factor)]
        t, y, x = data.shape;

    out = data.reshape([t, y/factor, factor, x/factor, factor]).mean(-1).mean(2)
    return out




def compute_field_sign(ph_az, ph_ev, pw_az, pw_ev, filt_size = 1):
    '''Compute field sign map from the phase maps of azimuth and elevation
    
    Arguments:
        ph_az {[type]} -- phase map azimuth
        ph_ev {[type]} -- phase map elevation
    '''
    #[dXev, dYev]= np.gradient(gaussian_filter(ph_ev, filt_size) )
    #[dXaz, dYaz]= np.gradient(gaussian_filter(ph_az, filt_size) )

    [dXev, dYev]= np.gradient(ph_ev)
    [dXaz, dYaz]= np.gradient(ph_az )

    angleEV = (dXev < 0) * np.pi + np.arctan(dYev / dXev);
    angleAZ = (dXaz < 0) * np.pi + np.arctan(dYaz / dXaz);

    field_sign = np.sin(angleEV - angleAZ);

    plt.figure(figsize = [10, 10])

    # Make a new double sided colormap.....
    colors1 = plt.cm.gist_rainbow(np.linspace(0., 1, 128))
    colors2 = plt.cm.gist_rainbow_r(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))  # combine them and build a new colormap
    double_side_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    plt.subplot(3,2,1); plt.imshow(ph_az, cmap = double_side_cmap); plt.title('azimuth 2_sided_cmap'); plt.colorbar()
    plt.subplot(3,2,2); plt.imshow(ph_az, cmap = 'gist_rainbow'); plt.title('azimuth'); plt.colorbar()
    
    plt.subplot(3,2,3); plt.imshow(ph_ev, cmap = 'gist_rainbow'); plt.title('elevation'); plt.colorbar()
    plt.subplot(3,2,4); plt.imshow(field_sign, cmap = 'bwr'); plt.title('field sign')

    plt.subplot(3,2,5); plt.imshow(pw_az); plt.title('power azimuth')
    plt.subplot(3,2,6); plt.imshow(pw_ev); plt.title('power elevation')


    return field_sign



def parse_timestamps(signal, timestamps, thresh = 2.5, interval_between_experiments = 2, min_isi = 0.01, min_number_exp = 20):
    ''' Takes as input a signal and a given timestamp array and will return a list of parsed experiments
    with timestamsp corresponding to each experiment, interval_between sets the interval in between
    each experiment to call a new experiment
    '''

    # Find timestamps of all stimulus frames..
    times = timestamps[1:][np.logical_and(signal[1:]>thresh, signal[:-1]<=thresh)]
    # Throw out ones that dont pass a mini isi
    discard_idx = np.where(np.diff(times) < min_isi)[0] + 1;
    times = np.delete(times, discard_idx)
    
    # Parse experiments
    cuts = np.where(np.diff(times) > interval_between_experiments)[0]; 
    # Anywhere it is inter-interval greater than interval represents a new experiment
    
    if len(cuts) > 0: # Condition where there is more than one experiment
        exp_list = []; 
        exp_list.append(times[:cuts[0]])
        for i in range(len(cuts)):
            if i == len(cuts)-1:
                exp_list.append(times[cuts[i]+1: ] )
            else:
                exp_list.append(times[cuts[i]+1:cuts[i+1]+1 ] )

        # Lastly throw out experiments that dont have the minimum number
        out_list = [s for s in exp_list if len(s) >= min_number_exp]      
    else:
        out_list = []; out_list.append(times)
	
    return out_list, times


