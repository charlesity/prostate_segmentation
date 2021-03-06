from __future__ import print_function
from six.moves import range
import numpy as np
import scipy as sp
import random
random.seed(2001)
import scipy.io
import matplotlib.pyplot as plt
import glob as glob
from tqdm import tqdm
from sklearn.preprocessing import Normalizer


XY_Data = np.load("./Processed_data/XY_Dataset_28_28_with_minus_1.npy")
slice_list = np.unique(XY_Data[:, XY_Data.shape[1]-1])
# print (slice_list)
save_location = './Processed_data/slices/'
# print (slice_list)
#
for s in slice_list:
    location = XY_Data[:, XY_Data.shape[1]-1] == s
    data = XY_Data[location]
    data[:, :(28**2)] = Normalizer().fit_transform(data[:, :(28**2)] + 1)

    # print (data[0, :50], data[:5].shape, data[0, :(28**2)].min())
    # break
    np.save(save_location+'pslice_'+str(int(s)), data)
