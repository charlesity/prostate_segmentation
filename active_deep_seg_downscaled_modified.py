'''Original Source from https://github.com/Riashat
'''

from __future__ import print_function
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
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




data_files = '.'
all_files = glob.glob(data_files + '/*.npy')
all_files = all_files  #load the number of folders indicated in the slice.... loading all will require more memory

n_experiments = 3  # number of experiments
batch_size = 128
nb_classes = 2

# input image dimensions
img_rows, img_cols = 28, 28

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 3
# convolution kernel size
nb_conv = 4

nb_epoch = 5

acquisition_iterations = 50 # number of aquisitions from unlabeled samples

dropout_iterations = 100  # number of dropout ROUNDS for uncertainty estimation

active_query_batch = 100  # number to consider from the unlabeled pool at a time
# All unlabeled samples could be considered

X_Train_percent = .2  # initial train percent from the entire training set
x_val_percent = .5  # of leftovers

num_labeled = 4

pool_batch_samples = 1000  #Number to sample from the Pool for dropout evaluation

img_dim = img_rows * img_cols  #flattened image dimension

XY_Data = fetch_data(all_files, 8)
X = XY_Data[:, :img_dim]
y = XY_Data[:, img_dim]
sss = StratifiedShuffleSplit(y, n_experiments, test_size=0.33, random_state=0)

# Number of times to perform experiments... Note this is different from the epoch
e = 0
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
    X_Train, Y_Train, X_Valid, Y_Valid, X_Pool, Y_Pool = split_train(X_Train_all, Y_Train_all, img_rows = img_rows, img_cols =img_cols, nb_classes= nb_classes,
     X_Train_percent = X_Train_percent, val_percent =x_val_percent)


    #performance evaluation metric for each experiment
    All_roc = np.zeros(shape=(
        acquisition_iterations + 1))  #Receiver Operator Characteristic data
    All_fpr = list()  # all false positive rates
    All_tpr = list()  # all true positive rates
    All_pre = list()
    All_rec = list()
    All_ap = list()
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
    
    #collect statistics of performance
    y_predicted = model.predict(X_Test, batch_size=batch_size)
    y_reversed = np.argmax(Y_Test, axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(
        y_reversed, y_predicted[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(
        y_reversed, y_predicted[:, 1])
    average_precision = metrics.average_precision_score(
        y_reversed, y_predicted[:, 1])
    print('Average Precision', average_precision)

#     All_roc[0] = roc_auc
#     All_fpr.append(fpr)
#     All_tpr.append(tpr)
#     All_pre.append(precision)
#     All_rec.append(recall)
#     All_ap.append(average_precision)
#     print('Starting Active Learning in Experiment ', e)

#     for i in range(acquisition_iterations):
#         print('POOLING ITERATION', i)

#         # pool_subset = len(X_Pool) # the remaining unlabeled set
#         pool_subset = pool_batch_samples  # sample just the given number.... usually we should sample from all the remaining unlabeled set
#         pool_subset_dropout = np.asarray(
#             random.sample(range(0, X_Pool.shape[0]),
#                           pool_subset))  # sample a subset of unlabeled set
#         X_Pool_Dropout = X_Pool[
#             pool_subset_dropout, :, :, :]  #sample a subset of unlabeled set
#         Y_Pool_Dropout = Y_Pool[
#             pool_subset_dropout]  #sample a subset of unlabeled set

#         score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))

#         for d in range(dropout_iterations):
#             print('Dropout Iteration', d)
#             dropout_score = model.predict_stochastic(
#                 X_Pool_Dropout, batch_size=batch_size, verbose=1)

#             #np.save(''+'Dropout_Score_'+str(d)+'Experiment_' + str(e)+'.npy',dropout_score)
#             score_All = score_All + dropout_score

#         # average out the probabilities .... ensembled based on dropout paper
#         Avg_Pi = np.divide(score_All, dropout_iterations)
#         # take log of the average
#         Log_Avg_Pi = np.log2(Avg_Pi)
#         #multply the average with their repective log probabilities -- to calculate entropy
#         Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
#         Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

#         U_X = Entropy_Average_Pi

#         # THIS FINDS THE MINIMUM INDEX
#         # a_1d = U_X.flatten()
#         # X_Pool_index = a_1d.argsort()[-active_query_batch:]

#         a_1d = U_X.flatten()
#         X_Pool_index = U_X.argsort()[-active_query_batch:][::-1]

#         # keep track of the pooled indices
#         X_Pool_All = np.append(X_Pool_All, X_Pool_index)
#         # print (X_Pool_index)

#         # saving pooled images
#         # for im in range(X_Pool_index[0:2].shape[0]):
#         #   Image = X_Pool[X_Pool_index[im], :, :, :]
#         #   img = Image.reshape((28,28))
#         #   sp.misc.imsave('./Pooled_Images/Max_Entropy'+ 'Exp_'+str(e) + 'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

#         Pooled_X = X_Pool_Dropout[X_Pool_index, :, :, ]
#         Pooled_Y = Y_Pool_Dropout[X_Pool_index]

#         print ("Pooled shape ", Pooled_X.shape, " from X Pool shape ", X_Pool.shape)
#         #first delete the random subset used for test time dropout from X_Pool
#         #Delete the pooled point from this pool set (this random subset)
#         #then add back the random pool subset with pooled points deleted back to the X_Pool set
#         # delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
#         # delete_Pool_Y = np.delete(Y_Pool, (pool_subset_dropout), axis=0)

#         X_Pool = np.delete(X_Pool, (pool_subset_dropout), axis=0)

		
#         Y_Pool = np.delete(Y_Pool, (pool_subset_dropout), axis=0) 

#         #delete from selected items from the dropout pool
#         X_Pool_Dropout = np.delete(
#             X_Pool_Dropout, (X_Pool_index), axis=0)
#         Y_Pool_Dropout = np.delete(
#             Y_Pool_Dropout, (X_Pool_index), axis=0)

# 		# delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (X_Pool_index), axis=0)
#   		# delete_Pool_Y_Dropout = np.delete(Y_Pool_Dropout, (X_Pool_index), axis=0)



#         #add whats left to the back to the pool
#         X_Pool = np.concatenate((X_Pool, X_Pool_Dropout), axis=0)
#         Y_Pool = np.concatenate((Y_Pool, Y_Pool_Dropout), axis=0)

#         print ("Shape of pool after acquisition ", X_Pool.shape, Y_Pool.shape)

#         print('Acquised Points added to training set')

#         X_Train = np.concatenate((X_Train, Pooled_X), axis=0)
#         Y_Train = np.concatenate((Y_Train, Pooled_Y), axis=0)

#         print('Train Model with pooled points')

#         # convert class vectors to binary class matrices
#         Y_Train = np_utils.to_categorical(Y_Train, nb_classes)

#         model = build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, X_Train.shape[0], c_param = 3.5)
#         model.compile(loss='categorical_crossentropy', optimizer='adam')
#         model.fit(
#             X_Train,
#             Y_Train,
#             batch_size=batch_size,
#             nb_epoch=nb_epoch,
#             show_accuracy=True,
#             verbose=1,
#             validation_data=(X_Valid, Y_Valid))

#         print('Evaluate Model Test Accuracy with pooled points')
#         y_predicted = model.predict_proba(X_Test, batch_size=batch_size)
#         y_reversed = np.argmax(Y_Test, axis=1)
#         fpr, tpr, thresholds = metrics.roc_curve(
#             y_reversed, y_predicted[:, 1], pos_label=1)
#         roc_auc = metrics.auc(fpr, tpr)
#         precision, recall, _ = metrics.precision_recall_curve(
#             y_reversed, y_predicted[:, 1])
#         average_precision = metrics.average_precision_score(
#             y_reversed, y_predicted[:, 1])
#         print('Average Precision ', average_precision)

#         #store successive results in numpy array
#         All_roc[i + 1] = roc_auc
#         All_fpr.append(fpr)
#         All_tpr.append(tpr)
#         All_pre.append(precision)
#         All_rec.append(recall)
#         All_ap.append(average_precision)

#     print('Saving Results Per Experiment')
#     np.save('./Results/' + 'Max_Entropy_Pool_ROC_' + 'Experiment_' + str(e) +
#             '.npy', All_roc)
#     np.save('./Results/' + 'Max_Entropy_Pool_FPR' + 'Experiment_' + str(e) +
#             '.npy', All_fpr)
#     np.save('./Results/' + 'Max_Entropy_Pool_TPR' + 'Experiment_' + str(e) +
#             '.npy', All_tpr)
#     np.save('./Results/' + 'Max_Entropy_Pool_PRE' + 'Experiment_' + str(e) +
#             '.npy', All_pre)
#     np.save('./Results/' + 'Max_Entropy_Pool_REC' + 'Experiment_' + str(e) +
#             '.npy', All_rec)
#     np.save(
#         './Results/' + 'Max_Entropy_Pool_Average_precision_' + str(e) + '.npy',
#         All_ap)
#     print ("===================== Experiment number ",e+1, " completed======================== " )
#     e += 1

# data_average_precious = np.load(
#     './Results/Max_Entropy_Pool_Average_precision_0.npy')

# plt.plot(data_average_precious)
# # plt.yscale('log', basey=2)
# # plt.plot([0, 1], [0, 1], color='navy', lw = 2,  linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# plt.xlabel('Iterations')
# plt.ylabel('Average Precision')
# plt.title('Average Precision curve')
# plt.show()
