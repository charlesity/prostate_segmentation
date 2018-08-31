'''Original Source from https://github.com/Riashat
'''
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils, Sequence
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K
import random
random.seed(2001)
import scipy.io
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

import glob as glob
from active_deep_utilities import *
from net_active_with_gen import *
# from model_utilities import *
import os
from imblearn.over_sampling import SMOTE



class DataGenerator(keras.utils.Sequence):
    def __init__(self, slice_IDs, XY_Data, slice_number_index, oversampler = None, batch_size = 16, dim = (28,28),
                 n_channels = 1, nb_classes = 10, shuffle=True):
        'initialization'
        self.dim = dim
        self.batch_size =batch_size
        self.slice_IDs = slice_IDs
        self.XY_Data = XY_Data
        self.slice_number_index = slice_number_index
        self.n_channels = n_channels
        self.nb_classes = nb_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.oversampler = oversampler

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.slice_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        selected_slices = self.slice_IDs[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__build_dataset(selected_slices)
        return X, keras.utils.to_categorical(y, num_classes=self.nb_classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.slice_IDs)

    def __build_dataset(self, s_list):
        image_s = self.dim[0]**2
        theX = list()
        theY = list()
        for l in s_list:
            id = XY_Data[:, self.slice_number_index] == l
            if len(theX) == 0:
                theX = XY_Data[id, :image_s]
                theY = XY_Data[id, image_s]
            else:
                tempX = self.XY_Data[id, :image_s]
                theX = np.vstack((theX, tempX))
                theY  = np.hstack((theY, self.XY_Data[id, image_s]))
        # over sample using smote
        if self.oversampler != None:
            theX, theY = self.oversampler.fit_sample(theX, theY)
        return theX.reshape(theX.shape[0], self.n_channels, *self.dim), theY

currentScript = os.path.splitext(__file__)[0]  #to collect performance data

n_experiments = 2  # number of experiments
batch_size = 128
nb_classes = 2

# input image dimensions
img_rows, img_cols = 28, 28

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 3
# convolution kernel size
nb_conv = 4

nb_epoch = 100

acquisition_iterations = 50 # number of aquisitions from unlabeled samples

dropout_iterations = 20  # number of dropout ROUNDS for uncertainty estimation

active_query_batch = 2  # number to added to the training data after active score evaluation
# All unlabeled samples could be considered

X_Train_percent = .1  # initial train percent from the entire training set
x_val_percent = .5  # of leftovers
total_train = .7
pool_batch_samples = 100  #Number to sample from the Pool for dropout evaluation

img_dim = img_rows * img_cols  #flattened image dimension

# XY_Data = fetch_data(all_files, 0)
XY_Data = np.load("./Processed_data/XY_Dataset_28_28.npy")
slice_list = np.unique(XY_Data[:, XY_Data.shape[1]-1])

for experiment_index in range(n_experiments):
    random.shuffle(slice_list)
    total_train_num = int(total_train *len(slice_list))
    train_slices = slice_list[:total_train_num]
    test_slices = slice_list[total_train_num:]

    initial_labeled_training_num = int(X_Train_percent* total_train_num)
    initial_labeled_slices = train_slices[:initial_labeled_training_num]
    print ('Total number of training slices', len(train_slices))
    print ('Total number of test slices', len(test_slices))
    print ('Number of slices to consider initially ', len(initial_labeled_slices))


    params = {'slice_number_index' : XY_Data.shape[1]-1,
                'dim': (img_rows, img_cols),
              'batch_size': 32,
              'nb_classes': 2,
              'shuffle':True}

    input_shape = (1, img_rows, img_cols)

    training_Generator = DataGenerator(initial_labeled_slices, XY_Data, oversampler=SMOTE(random_state=0), **params)
    testingg_Generator = DataGenerator(test_slices, XY_Data, oversampler=SMOTE(random_state=0), **params)
    model = net(input_shape, n_inputs=XY_Data.shape[0], filters=None, kernel_size=None, maxpool=None)
    model.fit_myGenerator(training_Generator, nb_epochs=50)
    All_auc = list()  #Receiver Operator Characteristic data
    # All_fpr = list()  # all false positive rates
    # All_tpr = list()  # all true positive rates
    All_pre = list()
    All_rec = list()
    All_ap = list()
    All_recall_score = list()
    All_precision_score = list()
    # y_predicted = model.predict_gen(testingg_Generator)
    # y_reversed = np.argmax(Y_Test, axis=1)
    # y_score = np.argmax(y_predicted, axis =1)
    #
    # fpr = dict()
    # tpr = dict()
    # auc = dict()
    # #collect statistics for the two classes
    # for ci in range(nb_classes):
    #     fpr[ci], tpr[ci], _ =  metrics.roc_curve(Y_Test[:, ci], y_predicted[:, ci])
    #     auc[ci] = metrics.auc(fpr[ci], tpr[ci])
    #
    # precision_score = metrics.precision_score(y_reversed, y_score)
    # recall_score = metrics.recall_score(y_reversed, y_score)
    # precision, recall, _ = metrics.precision_recall_curve(y_reversed, y_score, pos_label = 1)
    # average_precision = metrics.average_precision_score(y_reversed, y_score)
    # print ("Experiment ", e, "acquisition ", 0)
    # print('Average Precision', average_precision, "precision score", precision_score, "recall score ", recall_score)
    #
    # All_auc.append(auc)
    # All_fpr.append(fpr)
    # All_tpr.append(tpr)
    # All_pre.append(precision)
    # All_rec.append(recall)
    # All_ap.append(average_precision)
    # All_recall_score.append(recall_score)
    # All_precision_score.append(precision_score)
    # print('Starting Active Learning in Experiment ', e)

    # for i in range(acquisition_iterations):
    #     print('POOLING ITERATION', i)
    #
    #     # pool_subset = len(X_Pool) # the remaining unlabeled set
    #     pool_subset = pool_batch_samples  # sample just the given number.... usually we should sample from all the remaining unlabeled set
    #     pool_subset_dropout = np.asarray(
    #         random.sample(range(0, X_Pool.shape[0]),
    #                       pool_subset))  # sample a subset of unlabeled set
    #     X_Pool_Dropout = X_Pool[
    #         pool_subset_dropout, :, :, :]  #sample a subset of unlabeled set
    #     Y_Pool_Dropout = Y_Pool[
    #         pool_subset_dropout]  #sample a subset of unlabeled set
    #
    #     score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))
    #
    #     for d in range(dropout_iterations):
    #         dropout_score = model.predict_stochastic(
    #             X_Pool_Dropout, batch_size=batch_size, verbose=1)
    #
    #         #np.save(''+'Dropout_Score_'+str(d)+'Experiment_' + str(e)+'.npy',dropout_score)
    #         score_All = score_All + dropout_score
    #
    #     # average out the probabilities .... ensembled based on dropout paper
    #     Avg_Pi = np.divide(score_All, dropout_iterations)
    #     # take log of the average
    #     Log_Avg_Pi = np.log2(Avg_Pi)
    #     #multply the average with their repective log probabilities -- to calculate entropy
    #     Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
    #     Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
    #
    #     U_X = Entropy_Average_Pi
    #
    #     # THIS FINDS THE MINIMUM INDEX
    #     # a_1d = U_X.flatten()
    #     # X_Pool_index = a_1d.argsort()[-active_query_batch:]
    #
    #     a_1d = U_X.flatten()
    #     X_Pool_index = U_X.argsort()[-active_query_batch:][::-1]
    #
    #     # keep track of the pooled indices
    #     X_Pool_All = np.append(X_Pool_All, X_Pool_index)
    #     # print (X_Pool_index)
    #
    #     # saving pooled images
    #     # for im in range(X_Pool_index[0:2].shape[0]):
    #     #   Image = X_Pool[X_Pool_index[im], :, :, :]
    #     #   img = Image.reshape((28,28))
    #     #   sp.misc.imsave('./Pooled_Images/Max_Entropy'+ 'Exp_'+str(e) + 'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)
    #
    #     Pooled_X = X_Pool_Dropout[X_Pool_index, :, :, ]
    #     Pooled_Y = Y_Pool_Dropout[X_Pool_index]
    #
    #
    #     #first delete the random subset used for test time dropout from X_Pool
    #     #Delete the pooled point from this pool set (this random subset)
    #     #then add back the random pool subset with pooled points deleted back to the X_Pool set
    #     # delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
    #     # delete_Pool_Y = np.delete(Y_Pool, (pool_subset_dropout), axis=0)
    #
    #     X_Pool = np.delete(X_Pool, (pool_subset_dropout), axis=0)
    #     Y_Pool = np.delete(Y_Pool, (pool_subset_dropout), axis=0)
    #
    #     #delete from selected items from the dropout pool
    #     X_Pool_Dropout = np.delete(
    #         X_Pool_Dropout, (X_Pool_index), axis=0)
    #     Y_Pool_Dropout = np.delete(
    #         Y_Pool_Dropout, (X_Pool_index), axis=0)
    #
	# 	# delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (X_Pool_index), axis=0)
  	# 	# delete_Pool_Y_Dropout = np.delete(Y_Pool_Dropout, (X_Pool_index), axis=0)
    #
    #
    #
    #     #add whats left to the back to the pool
    #     X_Pool = np.concatenate((X_Pool, X_Pool_Dropout), axis=0)
    #     Y_Pool = np.concatenate((Y_Pool, Y_Pool_Dropout), axis=0)
    #
    #     X_Train = np.concatenate((X_Train, Pooled_X), axis=0)
    #     Y_Train = np.concatenate((Y_Train, Pooled_Y), axis=0)
    #
    #     # convert class vectors to binary class matrices
    #     Y_Train = np_utils.to_categorical(Y_Train, nb_classes)
    #
    #     model = build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, X_Train.shape[0], c_param = 3.5)
    #     model.compile(loss='categorical_crossentropy', optimizer='adam')
    #     model.fit(
    #         X_Train,
    #         Y_Train,
    #         batch_size=batch_size,
    #         nb_epoch=nb_epoch,
    #         show_accuracy=True,
    #         verbose=1,
    #         validation_data=(X_Valid, Y_Valid))
    #
    #     #collect statistics of performance
    #     y_predicted = model.predict(X_Test, batch_size=batch_size)
    #     y_reversed = np.argmax(Y_Test, axis=1)
    #     y_score = np.argmax(y_predicted, axis =1)
    #
    #     fpr = dict()
    #     tpr = dict()
    #     auc = dict()
    #     #collect statistics for the two classes
    #     for ci in range(nb_classes):
    #         fpr[ci], tpr[ci], _ =  metrics.roc_curve(Y_Test[:, ci], y_predicted[:, ci])
    #         auc[ci] = metrics.auc(fpr[ci], tpr[ci])
    #
    #     precision_score = metrics.precision_score(y_reversed, y_score)
    #     recall_score = metrics.recall_score(y_reversed, y_score)
    #     precision, recall, _ = metrics.precision_recall_curve(y_reversed, y_score, pos_label = 1)
    #     average_precision = metrics.average_precision_score(y_reversed, y_score)
    #     print ("Experiment ", e, "acquisition ", i)
    #     print('Average Precision', average_precision, "precision score", precision_score, "recall score ", recall_score)
    #
    #     All_auc.append(auc)
    #     All_fpr.append(fpr)
    #     All_tpr.append(tpr)
    #     All_pre.append(precision)
    #     All_rec.append(recall)
    #     All_ap.append(average_precision)
    #     All_recall_score.append(recall_score)
    #     All_precision_score.append(precision_score)
    #
    # print('Saving Results Per Experiment')
    #
    # np.save('./Results/' + currentScript + '_AUC_Experiment_' + str(e) +
    #         '.npy', All_auc)
    # np.save('./Results/' + currentScript + '_FPR_Experiment_' + str(e) +
    #         '.npy', All_fpr)
    # np.save('./Results/' + currentScript + '_TPR_Experiment_' + str(e) +
    #         '.npy', All_tpr)
    # np.save('./Results/' + currentScript + '_PRE_Experiment_' + str(e) +
    #         '.npy', All_pre)
    # np.save('./Results/' + currentScript + '_REC_Experiment_' + str(e) +
    #         '.npy', All_rec)
    # np.save(
    #     './Results/' + currentScript+'_AVG_pre_' + str(e) + '.npy',
    #     All_ap)
    # np.save(
    #     './Results/' + currentScript+'_recall_score_' + str(e) + '.npy',
    #     All_recall_score)
    # np.save('./Results/' + currentScript+'_precision_score_' + str(e) + '.npy',
    #     All_precision_score)
    # print ("===================== Experiment number ",e+1, " completed======================== " )
    # e += 1
    # if (e >= n_experiments ):
    #     break
