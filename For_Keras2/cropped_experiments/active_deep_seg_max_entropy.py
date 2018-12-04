'''Original Source from https://github.com/Riashat
'''
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '../../')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import keras.preprocessing as pp
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K
K.clear_session()
import random
random.seed(2001)
import scipy.io
import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE, RandomOverSampler

import glob as glob
from active_deep_utilities import *
from model_utilities_keras2 import *
import os
import argparse

def run():
    currentScript = os.path.basename(__file__)
    if dataset_type == '0':
        data_files = '../../dataset/cropped/zero/*.npz'
        temp = currentScript
        currentScript = "zeroData_"+temp
        if oversample:
            save_location = '../Results/Cropped_Results/oversampled/Zeros/' + currentScript
        else:
            save_location = '../Results/Cropped_Results/Zeros/' + currentScript
    elif dataset_type == '-1':
        data_files = '../../dataset/cropped/negative_backgrounds/*.npz'
        temp = currentScript
        currentScript = "negatives_"+temp
        if oversample:
            save_location = '../Results/Cropped_Results/oversampled/negatives/' + currentScript
        else:
            save_location = '../Results/Cropped_Results/negatives/' + currentScript

        save_location = '../Results/Cropped_Results/negatives/' + currentScript
    elif dataset_type == 'scaled_negative':
        data_files = '../../dataset/cropped/scaled_negative/*.npz'
        temp = currentScript
        currentScript = "scaled_negatives_"+temp
        if oversample:
            save_location = '../Results/Cropped_Results/oversampled/scaled_negatives/' + currentScript
        else:
            save_location = '../Results/Cropped_Results/scaled_negatives/' + currentScript
    else:
        print ("Pass the appropriate argument for the type of dataset")
        quit()
    # print (currentScript)
    # print (save_location)

    all_files = glob.glob(data_files)
    all_files = all_files  #load the number of folders indicated in the slice.... loading all will require more memory

    batch_size = 128
    nb_classes = 2

    # input image dimensions
    img_rows, img_cols = 40, 40

    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 3
    # convolution kernel size
    nb_conv = 4

    nb_epoch = 50

    acquisition_iterations = 30 # number of aquisitions from unlabeled samples

    dropout_iterations = 5  # number of dropout ROUNDS for uncertainty estimation

    active_query_batch = 60  # number to added to the training data after active score evaluation
    # All unlabeled samples could be considered

    X_Train_percent = .2  # initial train percent from the entire training set
    x_val_percent = .5  # of leftovers

    pool_batch_samples = 600  #Number to sample from the Pool for dropout evaluation

    img_dim = img_rows * img_cols  #flattened image dimension
    # all_files = all_files[:3]
    XY_Data = fetch_data(all_files, slice_range)


    X = XY_Data[:, :img_dim]
    y = XY_Data[:, img_dim]



    sss = StratifiedShuffleSplit(n_splits=n_experiments, test_size=0.33, random_state=0)

    smote_balancer = SMOTE(random_state=0)
    random_balancer = RandomOverSampler(random_state=0)
    # Number of times to perform experiments... Note this is different from the epoch
    e = 0 #starting experiment number
    for train_index, test_index in sss.split(X,y):
        # the data, split between train and test sets
        X_Train_all, X_Test = X[train_index], X[test_index]
        Y_Train_all, Y_Test = y[train_index], y[test_index]

        if K.image_data_format() == 'channels_first':
            X_Train_all = X_Train_all.reshape(X_Train_all.shape[0], 1, img_rows,img_cols)
            X_Test = X_Test.reshape(X_Test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X_Train_all = X_Train_all.reshape(X_Train_all.shape[0], img_rows,img_cols, 1)
            X_Test = X_Test.reshape(X_Test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

    	#reshape to appropriate backend format
        Y_Test = np_utils.to_categorical(Y_Test, nb_classes)	#one hot encode Y_Test


        #split train set into train, val, and unlabeled pool
        X_Train, Y_Train, X_Pool, Y_Pool = split_train_ratio_based(X_Train_all, Y_Train_all, img_rows = img_rows, img_cols =img_cols, nb_classes= nb_classes,
         X_Train_percent = X_Train_percent, val_percent =x_val_percent)

        #performance evaluation metric for each experiment
        All_auc = list()  #Receiver Operator Characteristic data
        All_pre = list()
        All_rec = list()
        All_ap = list()
        All_recall_score = list()
        All_precision_score = list()
        All_confusion_matrix = list()
        X_Pool_All = np.zeros(shape=(1))  #store all the pooled indices
    #
        model = build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, X_Train.shape[0], c_param = 3.5)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae', 'acc'])


        hist = model.fit(
            X_Train,
            Y_Train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,validation_split=0.2,
            verbose=1)
        # # print(hist.history.keys())
        # plt.plot(hist.history['loss'], label='traing_loss')
        # plt.plot(hist.history['mean_absolute_error'], label='mean_absolute_error')
        # plt.plot(hist.history['acc'],label='acc')
        # plt.plot(hist.history['val_loss'], label='val_loss')
        # plt.legend()
        # plt.show()

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
        confusion_matrix = metrics.confusion_matrix(y_reversed, y_score)
        print ("Script :"+currentScript)
        print ("Experiment ", e, "acquisition ", 0)
        print('Average Precision', average_precision, "precision score", precision_score, "recall score ", recall_score)
        print ('AUC', auc)

        All_auc.append(auc)
        All_pre.append(precision)
        All_rec.append(recall)
        All_ap.append(average_precision)
        All_recall_score.append(recall_score)
        All_precision_score.append(precision_score)
        All_confusion_matrix.append(confusion_matrix)
        print('Starting Active Learning in Experiment ', e)

        for i in range(acquisition_iterations):
            print('POOLING ITERATION', i)

            # pool_subset = len(X_Pool) # the remaining unlabeled set
            pool_subset = pool_batch_samples  # sample just the given number.... usually we should sample from all the remaining unlabeled set
            if X_Pool.shape[0] <= pool_subset:
                pool_subset_dropout = np.asarray(random.sample(range(0, X_Pool.shape[0]),
                              X_Pool.shape[0]))  # sample a subset of unlabeled set
            else:
                pool_subset_dropout = np.asarray(random.sample(range(0, X_Pool.shape[0]),
                                                               pool_subset))  # sample a subset of unlabeled set


            X_Pool_Dropout = X_Pool[
                pool_subset_dropout, :, :, :]  #sample a subset of unlabeled set
            Y_Pool_Dropout = Y_Pool[
                pool_subset_dropout]  #sample a subset of unlabeled set

            score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))

            f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])

            Avg_Pi, std =  predict_with_uncertainty(f, X_Pool_Dropout, nb_classes, dropout_iterations)

            Log_Avg_Pi = np.log2(Avg_Pi)
            #multply the average with their repective log Predictions -- to calculate entropy
            Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
            Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

            U_X = Entropy_Average_Pi

            # THIS FINDS THE MINIMUM INDEX
            # a_1d = U_X.flatten()
            # X_Pool_index = a_1d.argsort()[-active_query_batch:]

            a_1d = U_X.flatten()
            X_Pool_index = U_X.argsort()[-active_query_batch:][::-1]

            # keep track of the pooled indices
            X_Pool_All = np.append(X_Pool_All, X_Pool_index)
            # print (X_Pool_index)

            # saving pooled images
            # for im in range(X_Pool_index[0:2].shape[0]):
            #   Image = X_Pool[X_Pool_index[im], :, :, :]
            #   img = Image.reshape((28,28))
            #   sp.misc.imsave('./Pooled_Images/Max_Entropy'+ 'Exp_'+str(e) + 'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

            Pooled_X = X_Pool_Dropout[X_Pool_index, :, :, ]
            Pooled_Y = Y_Pool_Dropout[X_Pool_index]


            #first delete the random subset used for test time dropout from X_Pool
            #Delete the pooled point from this pool set (this random subset)
            #then add back the random pool subset with pooled points deleted back to the X_Pool set
            # delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
            # delete_Pool_Y = np.delete(Y_Pool, (pool_subset_dropout), axis=0)
            print (pool_subset_dropout.shape, "size of subset dropout")
            print (X_Pool.shape, "X_Pool before delete")
            X_Pool = np.delete(X_Pool, (pool_subset_dropout), axis=0)
            Y_Pool = np.delete(Y_Pool, (pool_subset_dropout), axis=0)
            print (X_Pool.shape, "X_Pool After delete")
            print (X_Pool_Dropout.shape, "X_Pool_Droput  Before delete")
            #delete from selected items from the dropout pool
            X_Pool_Dropout = np.delete(
                X_Pool_Dropout, (X_Pool_index), axis=0)
            Y_Pool_Dropout = np.delete(
                Y_Pool_Dropout, (X_Pool_index), axis=0)

            print (X_Pool_Dropout.shape, "X_Pool_Droput  Before delete")

    		# delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (X_Pool_index), axis=0)
      		# delete_Pool_Y_Dropout = np.delete(Y_Pool_Dropout, (X_Pool_index), axis=0)



            #add whats left to the back to the pool
            X_Pool = np.concatenate((X_Pool, X_Pool_Dropout), axis=0)
            Y_Pool = np.concatenate((Y_Pool, Y_Pool_Dropout), axis=0)

            print (X_Pool.shape, "X_Pool left")

            X_Train = np.concatenate((X_Train, Pooled_X), axis=0)
            Y_Train = np.concatenate((Y_Train, Pooled_Y), axis=0)

            if oversample:
                print ("Oversamplying")
                X_Train = X_Train.reshape((X_Train.shape[0], img_rows**2))
                # print (X_Train.shape)

                Y_Train = np.argmax(Y_Train, axis=1)
                # print (Y_Train)
                min_class_num =np.min(np.bincount(Y_Train.reshape(-1).astype(np.int)))
                if min_class_num < 4:
                    # print ("Random balancer")
                    X_Train, Y_Train =random_balancer.fit_sample(X_Train, Y_Train)
                else:
                    X_Train, Y_Train = smote_balancer.fit_sample(X_Train, Y_Train)
                    # print ("Smote balancer")
                # print (Y_Train)

                # print (X_Train.shape)
                # print (Y_Train)
                #reshape it back and continue
                X_Train= X_Train.reshape((X_Train.shape[0],)+ input_shape)
                Y_Train = np_utils.to_categorical(Y_Train, nb_classes)

            # model = build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, X_Train.shape[0], c_param = 3.5)
            # model.compile(loss='categorical_crossentropy', optimizer='adam')
            #fine tune the model
            init_epoch = int(nb_epoch/2)
            hist = model.fit(
                X_Train,
                Y_Train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,validation_split=0.2,
                verbose=1, initial_epoch = init_epoch+1)

            # # print(hist.history.keys())
            # plt.plot(hist.history['loss'], label='traing_loss')
            # plt.plot(hist.history['mean_absolute_error'], label='mean_absolute_error')
            # plt.plot(hist.history['acc'],label='acc')
            # plt.plot(hist.history['val_loss'], label='val_loss')
            # plt.legend()
            # plt.show()


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
            confusion_matrix = metrics.confusion_matrix(y_reversed, y_score)
            print ("Script :"+currentScript)
            print ("Experiment ", e, "acquisition ", i)
            print('Average Precision', average_precision, "precision score", precision_score, "recall score ", recall_score)
            print ('AUC', auc)

            All_auc.append(auc)
            All_pre.append(precision)
            All_rec.append(recall)
            All_ap.append(average_precision)
            All_recall_score.append(recall_score)
            All_precision_score.append(precision_score)
            All_confusion_matrix.append(confusion_matrix)

        print('Saving Results Per Experiment')

        np.save(save_location + '_AUC_Experiment_' + str(e) + '.npy', All_auc)
        np.save(save_location + '_PRE_Experiment_' + str(e) + '.npy', All_pre)
        np.save(save_location + '_REC_Experiment_' + str(e) + '.npy', All_rec)
        np.save(save_location+'_AVG_pre_' + str(e) + '.npy', All_ap)
        np.save(save_location+'_recall_score_' + str(e) + '.npy', All_recall_score)
        np.save(save_location+'_precision_score_' + str(e) + '.npy', All_precision_score)
        np.save(save_location+'_confusion_matrix' + str(e) + '.npy', All_confusion_matrix)
        print ("===================== Experiment number ",e+1, " completed======================== " )
        e += 1
        if (e >= n_experiments ):
            break
    K.clear_session()

def predict_with_uncertainty(f, x, no_classes, n_iter=100):
    result = np.zeros((n_iter,) + (x.shape[0], no_classes))
    for i in range(n_iter):
        result[i,:, :] = f((x, 1))[0]
    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("-ds_type", "--dataset_type", help=" 0 => '0 Background',  -1 -> '=1 Background', scaled_negative => 'Scaled negative background'")
        parser.add_argument("-ovs", "--oversampled", help ="Oversampled training",  action="store_true")
        parser.add_argument("-sr", "--slice_range", help="Number of subset to consider", default=6, type = int)
        parser.add_argument("-nexp", "--num_exp", help="Number of experiments", default=3, type = int)
        args = parser.parse_args()
        dataset_type = args.dataset_type
        oversample =args.oversampled
        slice_range = args.slice_range
        n_experiments = args.num_exp
        run()
