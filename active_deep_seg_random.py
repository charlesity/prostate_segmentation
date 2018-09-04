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



currentScript = os.path.splitext(__file__)[0]  #to collect performance data
data_files = './dataset/'
all_files = glob.glob(data_files + '/*.npz')
all_files = all_files  #load the number of folders indicated in the slice.... loading all will require more memory

n_experiments = 3  # number of experiments
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

active_query_batch = 20  # number to added to the training data after active score evaluation
# All unlabeled samples could be considered

X_Train_percent = .2  # initial train percent from the entire training set
x_val_percent = .5  # of leftovers

pool_batch_samples = 100  #Number to sample from the Pool for dropout evaluation

img_dim = img_rows * img_cols  #flattened image dimension
# all_files = all_files[:3]
XY_Data = fetch_data(all_files, 0)


X = XY_Data[:, :img_dim]
y = XY_Data[:, img_dim]

sss = StratifiedShuffleSplit(y, n_experiments, test_size=0.33, random_state=0)

# Number of times to perform experiments... Note this is different from the epoch
e = 0 #starting experiment number
for train_index, test_index in sss:
    # the data, split between train and test sets
    X_Train_all, X_Test = X[train_index], X[test_index]
    Y_Train_all, Y_Test = y[train_index], y[test_index]

	# if K.image_data_format() == 'channels_first':
	#reshape to appropriate backend format

    X_Train_all = X_Train_all.reshape(X_Train_all.shape[0], 1, img_rows,
                                      img_cols)


    X_Test = X_Test.reshape(X_Test.shape[0], 1, img_rows, img_cols)
    Y_Test = np_utils.to_categorical(Y_Test, nb_classes)	#one hot encode Y_Test
    input_shape = (1, img_rows, img_cols)

    #split train set into train, val, and unlabeled pool
    X_Train, Y_Train, X_Valid, Y_Valid, X_Pool, Y_Pool = split_train_ratio_based(X_Train_all, Y_Train_all, img_rows = img_rows, img_cols =img_cols, nb_classes= nb_classes,
     X_Train_percent = X_Train_percent, val_percent =x_val_percent)


    #performance evaluation metric for each experiment
    All_auc = list()  #Receiver Operator Characteristic data
    All_pre = list()
    All_rec = list()
    All_ap = list()
    All_recall_score = list()
    All_precision_score = list()
    X_Pool_All = np.zeros(shape=(1))  #store all the pooled indices

    model = build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, X_Train.shape[0], c_param = 3.5)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(
        X_Train,
        Y_Train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        show_accuracy=True,
        verbose=1,
        validation_data=(X_Valid, Y_Valid))

    # model.save_weights("./saved_models/"+currentScript+"model_0.h5")
    #collect statistics of performance
    y_predicted = model.predict(X_Test, batch_size=batch_size)
    y_reversed = np.argmax(Y_Test, axis=1)
    y_score = np.argmax(y_predicted, axis =1)

    fpr = dict()
    tpr = dict()
    auc = dict()
    #collect statistics for the two classes
    for ci in range(nb_classes):
        fpr[ci], tpr[ci], _ =  metrics.roc_curve(Y_Test[:, ci], y_predicted[:, ci])
        auc[ci] = metrics.auc(fpr[ci], tpr[ci])

    precision_score = metrics.precision_score(y_reversed, y_score)
    recall_score = metrics.recall_score(y_reversed, y_score)
    precision, recall, _ = metrics.precision_recall_curve(y_reversed, y_score, pos_label = 1)
    average_precision = metrics.average_precision_score(y_reversed, y_score)
    print ("Experiment ", e, "acquisition ", 0)
    print('Average Precision', average_precision, "precision score", precision_score, "recall score ", recall_score)
    print ('AUC', auc)
    All_auc.append(auc)
    All_pre.append(precision)
    All_rec.append(recall)
    All_ap.append(average_precision)
    All_recall_score.append(recall_score)
    All_precision_score.append(precision_score)

    print('Starting Active Learning in Experiment ', e)

    for i in range(acquisition_iterations):
        print('POOLING ITERATION', i)
        X_Pool_index = np.asarray(random.sample(range(0, pool_batch_samples), active_query_batch))

        Pooled_X = X_Pool[X_Pool_index, :, :, :]
        Pooled_Y = Y_Pool[X_Pool_index]

        # Delete the pool set from X_Pool
        X_Pool = np.delete(X_Pool, (X_Pool_index), axis=0)
        Y_Pool = np.delete(Y_Pool, (X_Pool_index), axis=0)


        X_Train = np.concatenate((X_Train, Pooled_X), axis=0)
        Y_Train = np.concatenate((Y_Train, Pooled_Y), axis=0)

        # convert class vectors to binary class matrices
        Y_Train = np_utils.to_categorical(Y_Train, nb_classes)

        model = build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, X_Train.shape[0], c_param = 3.5)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(
            X_Train,
            Y_Train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            show_accuracy=True,
            verbose=1,
            validation_data=(X_Valid, Y_Valid))

        # model.save_weights("./saved_models/"+currentScript+"model_"+str(i+1)+".h5")
        #collect statistics of performance
        y_predicted = model.predict(X_Test, batch_size=batch_size)
        y_reversed = np.argmax(Y_Test, axis=1)
        y_score = np.argmax(y_predicted, axis =1)

        fpr = dict()
        tpr = dict()
        auc = dict()
        #collect statistics for the two classes
        for ci in range(nb_classes):
            fpr[ci], tpr[ci], _ =  metrics.roc_curve(Y_Test[:, ci], y_predicted[:, ci])
            auc[ci] = metrics.auc(fpr[ci], tpr[ci])

        precision_score = metrics.precision_score(y_reversed, y_score)
        recall_score = metrics.recall_score(y_reversed, y_score)
        precision, recall, _ = metrics.precision_recall_curve(y_reversed, y_score, pos_label = 1)
        average_precision = metrics.average_precision_score(y_reversed, y_score)
        print ("Experiment ", e, "acquisition ", i)
        print('Average Precision', average_precision, "precision score", precision_score, "recall score ", recall_score)
        print ('AUC', auc)

        All_auc.append(auc)
        All_pre.append(precision)
        All_rec.append(recall)
        All_ap.append(average_precision)
        All_recall_score.append(recall_score)
        All_precision_score.append(precision_score)

    print('Saving Results Per Experiment')

    np.save('./Results/' + currentScript + '_AUC_Experiment_' + str(e) +
            '.npy', All_auc)
    np.save('./Results/' + currentScript + '_PRE_Experiment_' + str(e) +
            '.npy', All_pre)
    np.save('./Results/' + currentScript + '_REC_Experiment_' + str(e) +
            '.npy', All_rec)
    np.save(
        './Results/' + currentScript+'_AVG_pre_' + str(e) + '.npy',
        All_ap)
    np.save(
        './Results/' + currentScript+'_recall_score_' + str(e) + '.npy',
        All_recall_score)
    np.save('./Results/' + currentScript+'_precision_score_' + str(e) + '.npy',
        All_precision_score)
    print ("===================== Experiment number ",e+1, " completed======================== " )
    e += 1
    if (e >= n_experiments ):
        break
