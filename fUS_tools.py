import numpy as np
import sys
import glob

# TODO: Make something that imports the data from individual tiffs
# Make something that imports the timeline files from mat
# Make something that computes the fourier and plots the retinoptic maps
# Maybe reuse the allen institute data...

class dataLoading:
    """Class for 
    
    Returns:
        [type] -- [description]
    """
        
    def __init__(self):
        self.data = []

    def loadTiffDir(self):
        """Loads all tiff images in a given file
        
        Returns:
            [nd-array] -- nPixels_X x nPixels_Y x nT
        """

        return data
