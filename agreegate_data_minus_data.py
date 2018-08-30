from __future__ import print_function
from six.moves import range
import numpy as np
import scipy as sp
import random
random.seed(2001)
import scipy.io
import matplotlib.pyplot as plt
import glob as glob

def fetch_data(files, slice_range):
    #randomly pick test_percent of folders
    num = len(files)
    XY_Data = list()
    
    for f in files:
        Xy_tr = np.load(f)
        trimed_data = None
        if slice_range != 0:                    
            num_of_slices = Xy_tr[-1, 785] #number of slices
            start_slice =np.floor(num_of_slices/2) - np.floor(slice_range/2) 
            end_slice = start_slice + slice_range
            start_indices = Xy_tr[:, 785] >= start_slice
            end_indices = Xy_tr[:, 785] <= end_slice
            intercept = start_indices & end_indices        
            trimed_data = Xy_tr[intercept]
        else:
            trimed_data = Xy_tr        
        if len(XY_Data) == 0:
            XY_Data = trimed_data
        else:
            XY_Data = np.append(XY_Data, trimed_data, axis =0)
    return XY_Data


def fetch_data_for_minus(files, slice_range):
    #randomly pick test_percent of folders
    num = len(files)
    XY_Data = list()
    
    for f in files:
        Xy_tr = np.load(f)
        trimed_data = None
        if slice_range != 0:                    
            num_of_slices = Xy_tr[-1, 785] #number of slices
            start_slice =np.floor(num_of_slices/2) - np.floor(slice_range/2) 
            end_slice = start_slice + slice_range
            start_indices = Xy_tr[:, 785] >= start_slice
            end_indices = Xy_tr[:, 785] <= end_slice
            intercept = start_indices & end_indices        
            trimed_data = Xy_tr[intercept]
        else:
            trimed_data = Xy_tr        
        if len(XY_Data) == 0:
            XY_Data = trimed_data
        else:
            XY_Data = np.append(XY_Data, trimed_data, axis =0)
    return XY_Data

data_files = './minus_data/rescaled_minus/'
all_files = glob.glob(data_files+'/*.npy')
all_files = sorted(all_files)  #load the number of folders indicated in the slice.... loading all will require more memory
print (len(all_files))


XY_Data_All_with_minus= fetch_data_for_minus(all_files, 0)


print (XY_Data_All_with_minus.shape)


np.save("./minus_data/Processed_data/XY_Dataset_28_28_with_minus_1", XY_Data_All_with_minus)

