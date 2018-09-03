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
    def __init__(self, slice_IDs, oversampler = None, batch_size = 16, dim = (28,28),
                 n_channels = 1, nb_classes = 10, shuffle=True):
        'initialization'
        self.dim = dim
        if len(slice_IDs) < batch_size:
            self.batch_size = 1
        else:
            self.batch_size = batch_size
        self.slice_IDs = slice_IDs
        self.n_channels = n_channels
        self.nb_classes = nb_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.oversampler = oversampler


    def __len__(self):
        'Denotes the number of batches per epoch'
        b = int(np.floor(len(self.slice_IDs) / self.batch_size))
        return b

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        selected_slices = self.slice_IDs[index*self.batch_size:(index+1)*self.batch_size]
        # print ("Our of  {} slices, selected {}".format(self.slice_IDs, selected_slices))
        # print (self.slice_IDs[0:1])
        # Generate data
        X, y = self.__build_dataset(selected_slices)
        return X, keras.utils.to_categorical(y, num_classes=self.nb_classes)
        # return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.slice_IDs)

    def __get_channel_shape(self, X, y):
        if K.image_data_format() == 'channels_first':
            return X.reshape(X.shape[0], self.n_channels, *self.dim), y
        else:
            return X.reshape(X.shape[0], *self.dim, self.n_channels), y

    def __build_dataset(self, s_list):
        image_s = int(self.dim[0]**2)
        theX = list()
        theY = list()
        for l in s_list:
            n = int(l)
            d = np.load(slices_location+"pslice_"+str(n)+".npy")
            if len(theX) == 0:
                theX = d[:, :image_s]
                theY = d[:, image_s]
            else:
                tempX = d[:, :image_s]
                theX = np.vstack((theX, tempX))
                theY  = np.hstack((theY, d[:, image_s]))
        # over sample using smote
        if self.oversampler != None:
            try:
                theX, theY = self.oversampler.fit_sample(theX, theY)
                return self.__get_channel_shape(theX, theY)
            except:
                return self.__get_channel_shape(theX, theY)
        else:

            return self.__get_channel_shape(theX, theY)


def delete_sub_array(A, B):
    for i in range(len(B)):
        for j in range(len(A)):
            if B[i] == A[j]:
                A[j] = -1
    return A[A != -1]

def get_Xy_generator_data(generator):
    no_batches = 0
    Y = list()
    X = list()
    for d in generator:
        if len(Y) == 0:
            Y = d[1]
            X = d[0]
        else:
            Y = np.vstack((Y, d[1]))
            X = np.vstack((X, d[0]))
        no_batches +=1
        if no_batches >= generator.__len__():
            break
    return X, Y

currentScript = os.path.splitext(__file__)[0]  #to collect performance data

n_experiments = 1  # number of experiments
batch_size = 16
nb_classes = 2

# input image dimensions
img_rows, img_cols = 28, 28

nb_filters = 32
# size of pooling area for max pooling
pool_size = 3
# convolution kernel size
kernel_size = 4

nb_epoch = 100

acquisition_iterations = 20 # number of aquisitions from unlabeled samples

dropout_iterations = 30  # number of dropout ROUNDS for uncertainty estimation

active_query_batch = 6  # number to added to the training data after active score evaluation
# All unlabeled samples could be considered

X_Train_percent = .01  # initial train percent from the entire training set
total_train = .7
pool_batch_samples = 10  #Number to sample from the Pool for dropout evaluation

img_dim = img_rows * img_cols  #flattened image dimension

# XY_Data = fetch_data(all_files, 0)
num_slices  = 548

slices_location = "./Processed_data/slices/"
for experiment_index in range(n_experiments):
    slice_list = np.arange(num_slices)
    random.shuffle(slice_list)
    total_train_num = int(total_train *len(slice_list))
    train_slices = slice_list[:total_train_num]
    test_slices = slice_list[total_train_num:]

    initial_labeled_training_num = int(X_Train_percent* total_train_num)
    initial_labeled_slices = train_slices[:initial_labeled_training_num]
    unlabeled_slices = train_slices[initial_labeled_training_num:]
    print ('Total number of training slices', len(train_slices))
    print ('Total number of test slices', len(test_slices))
    print ('Number of slices to consider initially ', len(initial_labeled_slices))
    print ('Number of unlabeled slices ', len(unlabeled_slices))

    train_params = {'dim': (img_rows, img_cols),
              'batch_size': batch_size,
              'nb_classes': nb_classes,
              'shuffle':True}


    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)


    training_Generator = DataGenerator(initial_labeled_slices, oversampler=SMOTE(random_state=0), **train_params)

    test_params = {'dim': (img_rows, img_cols),
              'batch_size': batch_size,
              'nb_classes': nb_classes,
              'shuffle':True}
    testing_Generator = DataGenerator(test_slices, oversampler=None, **test_params)
    model = net(input_shape, filters=nb_filters
                , kernel_size=kernel_size, maxpool=pool_size)

    history = model.fit_myGenerator(training_Generator, nb_epochs=nb_epoch)
    #performance evaluation metric for each experiment
    All_auc = list()  #Receiver Operator Characteristic data
    All_fpr = list()  # all false positive rates
    All_tpr = list()  # all true positive rates
    All_pre = list()
    All_rec = list()
    All_ap = list()
    All_recall_score = list()
    All_precision_score = list()

    testing_Generator = DataGenerator(test_slices, oversampler=None, **test_params)
    X_Test, Y_Test = get_Xy_generator_data(testing_Generator)
    Y_Predicted = model.pred(X_Test, batch_size=128)

    #collect statistics of performance
    y_reversed = np.argmax(Y_Test, axis=1)
    y_score = np.argmax(Y_Predicted, axis =1)

    fpr = dict()
    tpr = dict()
    auc = dict()
    #collect statistics for the two classes
    for ci in range(nb_classes):
        fpr[ci], tpr[ci], _ =  metrics.roc_curve(Y_Test[:, ci], Y_Predicted[:, ci])
        auc[ci] = metrics.auc(fpr[ci], tpr[ci])


    precision_score = metrics.precision_score(y_reversed, y_score)
    recall_score = metrics.recall_score(y_reversed, y_score)
    precision, recall, _ = metrics.precision_recall_curve(y_reversed, y_score, pos_label = 1)
    average_precision = metrics.average_precision_score(y_reversed, y_score)
    print ("Experiment ", experiment_index, "acquisition ", 0)
    print('Average Precision = {} \
          Precision score = {} Recall Score = {}'.format(average_precision, precision_score, recall_score))
    print ("AUC Negative Class = {}, AUC Positive Class ={} ".format(auc[0], auc[1]))


    print('Starting Active Learning in Experiment ', experiment_index)

    for i in range(acquisition_iterations):
        print('POOLING ITERATION', i)
        test_dropout_slices = unlabeled_slices[:pool_batch_samples]  #sample a subset of unlabeled set
        slice_probabilities = np.zeros(len(test_dropout_slices))
        # print (test_dropout_slices)
        for i, aSlice in enumerate(test_dropout_slices):
            testing_gen_dropout = DataGenerator([aSlice], oversampler=None, **test_params)
            prob_supixels = model.predict_stochastic(get_Xy_generator_data(testing_gen_dropout)[0])
            prob_slice = np.prod(prob_supixels[:,0])
            slice_probabilities[i] = prob_slice
        #normalize slice_probabilities
        slice_probabilities = np.divide(slice_probabilities, np.sum(slice_probabilities))
        Log_Pi = np.log2(slice_probabilities)
        Entropy_Pi = -1 * np.multiply(slice_probabilities, Log_Pi)
        acquisition_index = Entropy_Pi.argsort()[-active_query_batch:][::-1]
        acquired_list = test_dropout_slices[acquisition_index]

        # add to the list of labeled
        initial_labeled_slices =np.concatenate([initial_labeled_slices, acquired_list])
        # print (initial_labeled_slices)

        unlabeled_slices = np.delete(unlabeled_slices, (test_dropout_slices))

        # print ('length of unlabed ',len(unlabeled_slices), len(test_dropout_slices))
        #
        # print (test_dropout_slices)
        test_dropout_slices = np.delete(test_dropout_slices, (acquired_list))


        unlabeled_slices =np.concatenate([unlabeled_slices, test_dropout_slices])
        print (initial_labeled_slices)
        training_Generator = DataGenerator(initial_labeled_slices, oversampler=SMOTE(random_state=0), **train_params)
        model = net(input_shape, filters=nb_filters
                    , kernel_size=kernel_size, maxpool=pool_size)

        history = model.fit_myGenerator(training_Generator, nb_epochs=nb_epoch)
        #performance evaluation metric for each experiment
        All_auc = list()  #Receiver Operator Characteristic data
        All_fpr = list()  # all false positive rates
        All_tpr = list()  # all true positive rates
        All_pre = list()
        All_rec = list()
        All_ap = list()
        All_recall_score = list()
        All_precision_score = list()

        testing_Generator = DataGenerator(test_slices, oversampler=None, **test_params)
        X_Test, Y_Test = get_Xy_generator_data(testing_Generator)
        Y_Predicted = model.pred(X_Test, batch_size=128)

        #collect statistics of performance
        y_reversed = np.argmax(Y_Test, axis=1)
        y_score = np.argmax(Y_Predicted, axis =1)

        fpr = dict()
        tpr = dict()
        auc = dict()
        #collect statistics for the two classes
        for ci in range(nb_classes):
            fpr[ci], tpr[ci], _ =  metrics.roc_curve(Y_Test[:, ci], Y_Predicted[:, ci])
            auc[ci] = metrics.auc(fpr[ci], tpr[ci])


        precision_score = metrics.precision_score(y_reversed, y_score)
        recall_score = metrics.recall_score(y_reversed, y_score)
        precision, recall, _ = metrics.precision_recall_curve(y_reversed, y_score, pos_label = 1)
        average_precision = metrics.average_precision_score(y_reversed, y_score)
        print ("Experiment ", experiment_index, "acquisition ", 0)
        print('Average Precision = {} \
              Precision score = {} Recall Score = {}'.format(average_precision, precision_score, recall_score))
        print ("AUC Negative Class = {}, AUC Positive Class ={} ".format(auc[0], auc[1]))


        # print (test_dropout_slices)


        # print (len(test_dropout_slices))
        # # # what whats left to the unlabeled_slices
        # # unlabeled_slices = np.concatenate([unlabeled_slices, test_dropout_slices])
        # # print (len(unlabeled_slices))



    #     Avg_Pi = model.predict_stochastic(
    #         X_Pool_Dropout, batch_size=batch_size, verbose=1)
    #
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
    # #
    # #     model = build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, X_Train.shape[0], c_param = 3.5)
    # #     model.compile(loss='categorical_crossentropy', optimizer='adam')
    # #     model.fit(
    # #         X_Train,
    # #         Y_Train,
    # #         batch_size=batch_size,
    # #         nb_epoch=nb_epoch,
    # #         show_accuracy=True,
    # #         verbose=1,
    # #         validation_data=(X_Valid, Y_Valid))
    # #
    # #     #collect statistics of performance
    # #     Y_Predicted = model.predict(X_Test, batch_size=batch_size)
    # #     y_reversed = np.argmax(Y_Test, axis=1)
    # #     y_score = np.argmax(Y_Predicted, axis =1)
    # #
    # #     fpr = dict()
    # #     tpr = dict()
    # #     auc = dict()
    # #     #collect statistics for the two classes
    # #     for ci in range(nb_classes):
    # #         fpr[ci], tpr[ci], _ =  metrics.roc_curve(Y_Test[:, ci], Y_Predicted[:, ci])
    # #         auc[ci] = metrics.auc(fpr[ci], tpr[ci])
    # #
    # #     precision_score = metrics.precision_score(y_reversed, y_score)
    # #     recall_score = metrics.recall_score(y_reversed, y_score)
    # #     precision, recall, _ = metrics.precision_recall_curve(y_reversed, y_score, pos_label = 1)
    # #     average_precision = metrics.average_precision_score(y_reversed, y_score)
    # #     print ("Experiment ", e, "acquisition ", i)
    # #     print('Average Precision', average_precision, "precision score", precision_score, "recall score ", recall_score)
    # #
    # #     All_auc.append(auc)
    # #     All_fpr.append(fpr)
    # #     All_tpr.append(tpr)
    # #     All_pre.append(precision)
    # #     All_rec.append(recall)
    # #     All_ap.append(average_precision)
    # #     All_recall_score.append(recall_score)
    # #     All_precision_score.append(precision_score)
    # #
    # # print('Saving Results Per Experiment')
    # #
    # # np.save('./Results/' + currentScript + '_AUC_Experiment_' + str(e) +
    # #         '.npy', All_auc)
    # # np.save('./Results/' + currentScript + '_FPR_Experiment_' + str(e) +
    # #         '.npy', All_fpr)
    # # np.save('./Results/' + currentScript + '_TPR_Experiment_' + str(e) +
    # #         '.npy', All_tpr)
    # # np.save('./Results/' + currentScript + '_PRE_Experiment_' + str(e) +
    # #         '.npy', All_pre)
    # # np.save('./Results/' + currentScript + '_REC_Experiment_' + str(e) +
    # #         '.npy', All_rec)
    # # np.save(
    # #     './Results/' + currentScript+'_AVG_pre_' + str(e) + '.npy',
    # #     All_ap)
    # # np.save(
    # #     './Results/' + currentScript+'_recall_score_' + str(e) + '.npy',
    # #     All_recall_score)
    # # np.save('./Results/' + currentScript+'_precision_score_' + str(e) + '.npy',
    # #     All_precision_score)
    # # print ("===================== Experiment number ",e+1, " completed======================== " )
    # # e += 1
    # # if (e >= n_experiments ):
    # #     break
