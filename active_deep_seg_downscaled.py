'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
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


def fetch_data(files):
    #randomly pick test_percent of folders
    num = len(files)
    XY_Data = list()

    for f in files:
        Xy_tr = np.load(f)
        if len(XY_Data) == 0:
            XY_Data = Xy_tr
        else:
            XY_Data = np.append(XY_Data, Xy_tr, axis =0)
    return XY_Data

# def split_data(files, test_percent = .33):
#     #randomly pick test_percent of folders
#     num = len(files)
#     test_list = int(test_percent*num)
#     rand_index = np.arange(0, num)
#     np.random.shuffle(rand_index)
    
#     XY_Train = list()
#     XY_Test = list()    

#     # train set
#     for k in rand_index[test_list:]:
#       Xy_tr = np.load(files[k])              
#       if len(XY_Train) == 0:
#         XY_Train = Xy_tr
#       else:
#         XY_Train = np.append(XY_Train, Xy_tr, axis =0)
#     # test set
    
#     for  k in rand_index[:test_list]:
#       Xy_ts = np.load(files[k])        
#       if len(XY_Test) == 0:
#         XY_Test = Xy_ts
#       else:
#         XY_Test = np.append(XY_Test, Xy_ts, axis =0)    
#     return XY_Train, XY_Test


data_files = '.'
all_files = glob.glob(data_files+'/*.npy')
all_files = all_files[:8]  #load the number of folders indicated in the slice.... loading all will require more memory


Experiments = 1  #number of experiments
batch_size = 128  
nb_classes = 2


# input image dimensions
img_rows, img_cols = 28,28

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 5
# convolution kernel size
nb_conv = 3

nb_epoch = 50

acquisition_iterations = 50  #number of aquisitions for unlabeled samples to make

#use a large number of dropout iterations
dropout_iterations = 50   # number of dropout neurons to consider during uncertainty estimation

Queries = 100  #number of queries to consider from the unlabeled pool at a time. All unlabeled samples could be considered

X_Train_pos = .2  # of the entire training set
x_val_ratio = .5  # of leftovers
X_Pool_pos  = .5 # of leftovers

num_labeled = 4

pool_percent = .6   # percentage of the training dataset to consider as unlabeled pool

# class_weight = {0: 20, 1: 80}

# to detail with unbalanced dataset 

img_dim = img_rows*img_cols   #flattened image dimension

#Number of times to perform experiments... Note this is different from the epoch
for e in np.arange(Experiments):
    # the data, split between train and test sets
    XY_Train_All, XY_Test_ALL= split_data(all_files, test_percent=.33)
    (X_Train_all, Y_Train_all), (X_Test, Y_Test) =(XY_Train_All[:, :img_dim], XY_Train_All[:, img_dim]),  (XY_Test_ALL[:, :img_dim], XY_Test_ALL[:, img_dim])

    # if K.image_data_format() == 'channels_first':
    X_Train_all = X_Train_all.reshape(X_Train_all.shape[0], 1, img_rows, img_cols)
    X_Test = X_Test.reshape(X_Test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

    #shuffle the training set
    random_shuffle = np.asarray(random.sample(range(0,X_Train_all.shape[0]),X_Train_all.shape[0]))

    X_Train_all = X_Train_all[random_shuffle, :, :, :]
    Y_Train_all  = Y_Train_all[random_shuffle]

    # print ('Shape of all the X ', X_Train_all.shape)
    # print('Distribution of ALL Y_Train Classes:', np.bincount(Y_Train_all.reshape(-1).astype(np.int)))

    idx_negatives = np.array( np.where(Y_Train_all==0)).reshape(-1)
    idx_positives = np.array( np.where(Y_Train_all==1)).reshape(-1)


    train_num_half = int(X_Train_pos * idx_positives.shape[0])

    X_Train_pos = X_Train_all[idx_positives[:train_num_half], :, :, :]
    X_Train_neg = X_Train_all[idx_negatives[:train_num_half], :, :, :] # pick same size as that of positives 

    Y_Train_pos = Y_Train_all[idx_positives[:train_num_half]]
    Y_Train_neg = Y_Train_all[idx_negatives[:train_num_half]]



    X_Train = np.concatenate((X_Train_pos, X_Train_neg), axis = 0)
    Y_Train = np.concatenate((Y_Train_neg, Y_Train_pos), axis = 0)

    # print ('X_Train and Y_Train shapes ', X_Train.shape, Y_Train.shape)
    # print('Distribution of Y_Train Classes:', np.bincount(Y_Train.reshape(-1).astype(np.int)))

    left_over_after_xtrain_pos = idx_positives.shape[0] - train_num_half 
    left_over_after_xtrain_neg =  idx_negatives.shape[0] - train_num_half

    val_pos_start_index =  train_num_half
    val_pos_end_index = val_pos_start_index + int(left_over_after_xtrain_pos* x_val_ratio)

    X_Valid_pos = X_Train_all[idx_positives[val_pos_start_index:val_pos_end_index], :, :, :]
    Y_Valid_pos = Y_Train_all[idx_positives[val_pos_start_index:val_pos_end_index]]

    X_Valid_neg = X_Train_all[idx_negatives[val_pos_start_index:val_pos_end_index], :, :, :]
    Y_Valid_neg = Y_Train_all[idx_negatives[val_pos_start_index:val_pos_end_index]]

    X_Valid = np.concatenate((X_Valid_pos, X_Valid_neg), axis=0)
    Y_Valid = np.concatenate((Y_Valid_pos, Y_Valid_neg), axis=0)

    # print ('X_Valid and Y_Valid shapes ',X_Valid.shape, Y_Valid.shape)
    # print('Distribution of Y_Valid Classes:', np.bincount(Y_Valid.reshape(-1).astype(np.int)))

    X_Pool_neg = X_Train_all[idx_negatives[val_pos_end_index:], :, :, :]
    Y_Pool_neg = Y_Train_all[idx_negatives[val_pos_end_index:]]


    X_Pool_pos = X_Train_all[idx_positives[val_pos_end_index:], :, :, :]
    Y_Pool_pos =  Y_Train_all[idx_positives[val_pos_end_index:]]

    X_Pool = np.concatenate((X_Pool_neg, X_Pool_pos), axis=0)
    Y_Pool = np.concatenate((Y_Pool_neg, Y_Pool_pos), axis = 0)

    print ('X_Pool and Y_Pool shapes ', X_Pool.shape, Y_Pool.shape)
    print('Distribution of Y_Pool Classes:', np.bincount(Y_Pool.reshape(-1).astype(np.int)))



    #one -hot encode the vectors 
    Y_Test = np_utils.to_categorical(Y_Test, nb_classes)   
    Y_Valid = np_utils.to_categorical(Y_Valid, nb_classes)
    Y_Pool = np_utils.to_categorical(Y_Pool, nb_classes)
    Y_Train = np_utils.to_categorical(Y_Train, nb_classes)


    #performance evaluation for each experiment
    All_roc = np.zeros(shape=(acquisition_iterations+1))  #Receiver Operator Characteristic data
    All_fpr = list()  # all false positive rates
    All_tpr = list()  # all true positive rates
    X_Pool_All = np.zeros(shape=(1))  #store all the pooled indices



    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    c = 3.5
    Weight_Decay = c / float(X_Train.shape[0])
    model.add(Flatten())
    model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X_Train, Y_Train, batch_size=batch_size, nb_epoch=nb_epoch,  validation_data=(X_Valid, Y_Valid))
    y_predicted = model.predict_proba(X_Test, batch_size=batch_size)
    y_reversed = np.argmax(Y_Test, axis = 1)
    fpr, tpr, thresholds = metrics.roc_curve(y_reversed, y_predicted[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    All_roc[0] = roc_auc
    All_fpr.append(fpr)
    All_tpr.append(tpr)

    print ("Area under the curve ", roc_auc)
  
    print('Starting Active Learning in Experiment ', e)

    for i in range(acquisition_iterations):
        print('POOLING ITERATION', i)


        # pool_subset = len(X_Pool)  # sample all the remaining unlabeled set or use the
        pool_subset = 200  # sample just the remaining unlabeled set or use the
        pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset)) #sample a subset of unlabeled set
        X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]  #sample a subset of unlabeled set
        Y_Pool_Dropout = Y_Pool[pool_subset_dropout]  #sample a subset of unlabeled set

        score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))

        for d in range(dropout_iterations):
          print ('Dropout Iteration', d)
          dropout_score = model.predict_stochastic(X_Pool_Dropout,batch_size=batch_size, verbose=1)
          
          #np.save(''+'Dropout_Score_'+str(d)+'Experiment_' + str(e)+'.npy',dropout_score)
          score_All = score_All + dropout_score

        Avg_Pi = np.divide(score_All, dropout_iterations)
        Log_Avg_Pi = np.log2(Avg_Pi)
        Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

        U_X = Entropy_Average_Pi

        # THIS FINDS THE MINIMUM INDEX 
        # a_1d = U_X.flatten()
        # X_Pool_index = a_1d.argsort()[-Queries:]

        a_1d = U_X.flatten()
        X_Pool_index = U_X.argsort()[-Queries:][::-1]

        X_Pool_All = np.append(X_Pool_All, X_Pool_index)
        # print (X_Pool_index)

        # saving pooled images
        for im in range(X_Pool_index[0:2].shape[0]):
          Image = X_Pool[X_Pool_index[im], :, :, :]
          img = Image.reshape((28,28))
          sp.misc.imsave('./Pooled_Images/Max_Entropy'+ 'Exp_'+str(e) + 'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

        Pooled_X = X_Pool_Dropout[X_Pool_index, :,:,]
        Pooled_Y = Y_Pool_Dropout[X_Pool_index] 

        #first delete the random subset used for test time dropout from X_Pool
        #Delete the pooled point from this pool set (this random subset)
        #then add back the random pool subset with pooled points deleted back to the X_Pool set
        delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
        delete_Pool_Y = np.delete(Y_Pool, (pool_subset_dropout), axis=0)

        delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (X_Pool_index), axis=0)
        delete_Pool_Y_Dropout = np.delete(Y_Pool_Dropout, (X_Pool_index), axis=0)

        X_Pool = np.concatenate((X_Pool, X_Pool_Dropout), axis=0)
        Y_Pool = np.concatenate((Y_Pool, Y_Pool_Dropout), axis=0)

        print('Acquised Points added to training set')

        X_Train = np.concatenate((X_Train, Pooled_X), axis=0)
        Y_Train = np.concatenate((Y_Train, Pooled_Y), axis=0)

        print('Train Model with pooled points')

        # convert class vectors to binary class matrices
        Y_Train = np_utils.to_categorical(Y_Train, nb_classes)


        model = Sequential()
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))


        c = 3.5
        Weight_Decay = c / float(X_Train.shape[0])
        model.add(Flatten())
        model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))


        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(X_Train, Y_Train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_Valid, Y_Valid))
        
        print('Evaluate Model Test Accuracy with pooled points')
        y_predicted = model.predict_proba(X_Test, batch_size=batch_size)
        y_reversed = np.argmax(Y_Test, axis = 1)
        fpr, tpr, thresholds = metrics.roc_curve(y_reversed, y_predicted[:, 1], pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)


        #store successive results in numpy array
        All_roc[i+1] = roc_auc
        All_fpr.append(fpr)
        All_tpr.append(tpr)

        print ("Area under the curve ", roc_auc)

    print('Saving Results Per Experiment')
    np.save('./Results/'+'Max_Entropy_Pool_ROC_'+ 'Experiment_' + str(e) + '.npy', All_roc)
    np.save('./Results/'+'Max_Entropy_Pool_FPR'+ 'Experiment_' + str(e) + '.npy', All_fpr)
    np.save('./Results/'+'Max_Entropy_Pool_TPR'+ 'Experiment_' + str(e) + '.npy', All_tpr)


# #view output for just one experiment

data_roc = np.load('./Results/Max_Entropy_Pool_ROC_'+ 'Experiment_' + str(0) + '.npy')
data_fpr = np.load('./Results/Max_Entropy_Pool_FPR'+ 'Experiment_' + str(0) + '.npy')
data_tpr = np.load('./Results/Max_Entropy_Pool_TPR'+ 'Experiment_' + str(0) + '.npy')


for i in range(len(data_fpr)):
    plt.plot(data_fpr[i], data_tpr[i], label='ROC curve (area = %0.2f at interaction = %1.0f)' % (data_roc[i], i))
plt.plot([0, 1], [0, 1], color='navy', lw = 2,  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - ROC Curve')
plt.legend(loc="lower right")
plt.show()



