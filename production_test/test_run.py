'''Original Source from https://github.com/Riashat
'''

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K
import random
random.seed(2001)
import scipy.io
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit

import glob as glob
from active_deep_utilities import *
from model_utilities import *
import os
import argparse


def run():
    currentScript = os.path.basename(__file__)  #to collect performance data
    save_location = './Results/' + currentScript
    data_files = './dataset/'
    all_files = glob.glob(data_files + '/*.npz')
    all_files = all_files  #load the number of folders indicated in the slice.... loading all will require more memory

    batch_size = 128
    nb_classes = 2

    # input image dimensions
    img_rows, img_cols = 60, 60

    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 3
    # convolution kernel size
    nb_conv = 4

    nb_epoch = 50

    acquisition_iterations = 30 # number of aquisitions from unlabeled samples

    dropout_iterations = 20  # number of dropout ROUNDS for uncertainty estimation

    active_query_batch = 60  # number to added to the training data after active score evaluation
    # All unlabeled samples could be considered

    X_Train_percent = .2  # initial train percent from the entire training set
    x_val_percent = .5  # of leftovers

    pool_batch_samples = 600  #Number to sample from the Pool for dropout evaluation

    img_dim = img_rows * img_cols  #flattened image dimension
    # all_files = all_files[:3]

    XY_Data = fetch_data(all_files, slice_range, img_dim = (img_rows, img_cols))

    # print (XY_Data.shape)

    X = XY_Data[:, :img_dim]
    y = XY_Data[:, img_dim]



    sss = StratifiedShuffleSplit(y, n_experiments, test_size=0.33, random_state=0)

    # Number of times to perform experiments... Note this is different from the epoch
    e = 0 #starting experiment number
    for train_index, test_index in sss:
        # the data, split between train and test sets
        X_Train_all, X_Test = X[train_index], X[test_index]
        Y_Train_all, Y_Test = y[train_index], y[test_index]

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-sr", "--slice_range", help="Number of subset to consider", default=6, type = int)
        parser.add_argument("-nexp", "--num_exp", help="Number of experiments", default=3, type = int)
        args = parser.parse_args()
        slice_range = args.slice_range
        n_experiments = args.num_exp
        run()
