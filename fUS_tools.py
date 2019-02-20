import numpy as np
import sys
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tifffile
import scipy

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
                raise NotImplemented
        else:
            raise NotImplemented
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


def play_movie(data):
    ''' Code for quickly playing a movie
    need to specify the x, y, and T dimensions'''

    if isinstance(data, np.ndarray):
        images = []
        for i in range(data.shape[0]):
            images.append(np.squeeze(data[i, :, : ]))
    fig = plt.figure()
    animation.ArtistAnimation(fig, images, interval=500, blit=True, repeat_delay=1000)
    #plt.show()

def export_tiffs(data, outDir='', dims = [0,1,2]):
    ''' dims takes as inpute the x y t dimensions'''

    if outDir == '':
        raise NotADirectoryError
    else:
        tifffile.imsave(outDir, np.transpose(data, [dims[2], dims[0], dims[1]]).astype('single') )

def upsample_xyt(data, factor = 2):
    '''Code will upsample an array of xyt by a given scaling factor'''



def compute_fft(data , dims = [0,1,2]):
    ''' Compute the fourier transform and plots the first 100 power and phase maps'''
    reshaped = np.transpose(data, [dims[2], dims[0], dims[1]]); # Transpose to y, x, t
    out = np.fft.fft(reshaped, axis = 0)
    plt.figure()

    for i in range(100):
        plt.subplot(10,10,i+1); plt.imshow(np.abs(out[i,:,:]),cmap = 'gist_rainbow'); plt.axis('off')

    plt.show()


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
    cuts = np.where(np.diff(times) > interval_between_experiments)[0]; # Anywhere it is inter-interval greater than interval represents a new experiment
    exp_list = []; 
    exp_list.append(times[:cuts[0]])
    for i in range(len(cuts)):
        if i == len(cuts)-1:
            exp_list.append(times[cuts[i]+1: ] )
        else:
            exp_list.append(times[cuts[i]+1:cuts[i+1]+1 ] )

    # Lastly throw out experiments that dont have the minimum number
    out_list = [s for s in exp_list if len(s) >= min_number_exp]        

    return out_list, times