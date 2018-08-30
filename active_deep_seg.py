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
from sklearn.metrics import classification_report, confusion_matrix

import glob as glob


# class Metrics(Callback):

#   def on_train_begin(self, logs={}):
#    self.val_f1s = []
#    self.val_recalls = []
#    self.val_precisions = []
#   def on_epoch_end(self, epoch, logs={}):
#     print (dir(self))
#     # val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
#     # val_targ = self.model.validation_data[1]
#     # _val_f1 = f1_score(val_targ, val_predict)
#     # _val_recall = recall_score(val_targ, val_predict)
#     # _val_precision = precision_score(val_targ, val_predict)
#     # self.val_f1s.append(_val_f1)
#     # self.val_recalls.append(_val_recall)
#     # self.val_precisions.append(_val_precision)
#     # print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
#     return



def split_data(files, test_percent = .33):
    #randomly pick test_percent of folders
    num = len(files)
    test_list = int(test_percent*num)
    rand_index = np.arange(0, num)
    np.random.shuffle(rand_index)
    
    XY_Train = list()
    XY_Test = list()
    

    # train set
    for k in rand_index[test_list:]:
        data = np.load(files[k])        
        Xy_tr = [np.concatenate((d[0][0].toarray().ravel(), [d[0][1]], [d[1]])) for d in data]       
        if len(XY_Train) == 0:
            XY_Train = np.array(Xy_tr)
        else:
            XY_Train = np.vstack((XY_Train, Xy_tr))
    # test set
    
    for  k in rand_index[:test_list]:       
        data = np.load(files[k])        
        Xy_ts = [np.concatenate((d[0][0].toarray().ravel(), [d[0][1]], [d[1]])) for d in data]       
        if len(XY_Test) == 0:
            XY_Test = np.array(Xy_ts)
        else:
            XY_Test = np.vstack((XY_Test, Xy_ts))    
    return XY_Train, XY_Test


# metrics = Metrics()

data_files = '.'
all_files = glob.glob(data_files+'/*.npy')
all_files = all_files[:4]

Experiments = 1
batch_size = 10
num_classes = 2
epochs = 1

# input image dimensions
img_rows, img_cols = 160, 160

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 5
# convolution kernel size
nb_conv = 3

nb_epoch = 2



score=0
all_accuracy = 0
acquisition_iterations = 5

#use a large number of dropout iterations
dropout_iterations = 10

Queries = 10

X_Train_pos = .2  # of the entire training set
x_val_ratio = .4  # of leftovers
X_Pool_pos  = .6 # of leftovers

num_labeled = 4

pool_percent = .6
nb_classes = 2

Experiments_All_Accuracy = np.zeros(shape=(acquisition_iterations+1))

#Number of times to perform experiments... Note this is different from the epoch
for e in np.arange(Experiments):
  # the data, split between train and test sets
  
  XY_Train_All, XY_Test_ALL= split_data(all_files, test_percent=.33)
  (X_Train_all, Y_Train_all), (X_Test, Y_Test) =(XY_Train_All[:, :25600], XY_Train_All[:, 25600]),  (XY_Test_ALL[:, :25600], XY_Test_ALL[:, 25600])

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

  # print ('X_Pool and Y_Pool shapes ', X_Pool.shape, Y_Pool.shape)
  # print('Distribution of Y_Pool Classes:', np.bincount(Y_Pool.reshape(-1).astype(np.int)))


  Y_Test = np_utils.to_categorical(Y_Test, nb_classes)
  Y_Valid = np_utils.to_categorical(Y_Valid, nb_classes)
  Y_Pool = np_utils.to_categorical(Y_Pool, nb_classes)
  Y_Train = np_utils.to_categorical(Y_Train, nb_classes)


#loss values in each experiment
  Pool_Valid_Loss = np.zeros(shape=(nb_epoch, 1))   
  Pool_Train_Loss = np.zeros(shape=(nb_epoch, 1)) 
  Pool_Valid_Acc = np.zeros(shape=(nb_epoch, 1))  
  Pool_Train_Acc = np.zeros(shape=(nb_epoch, 1)) 
  X_Pool_All = np.zeros(shape=(1))

  

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
  hist = model.fit(X_Train, Y_Train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_Valid, Y_Valid))


  y_redict = model.predict_prob(X_Test)  
  

  target_names = ['Background', 'Foreground']
  m = confusion_matrix(known_1d, predicted_1d, labels = target_names)
  print (m)

  # plt.imshow(anImage.reshape(160,160))
  # plt.show()



  # Train_Result_Optimizer = hist.history
#   Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
#   Train_Loss = np.array([Train_Loss]).T
#   Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
#   Valid_Loss = np.asarray([Valid_Loss]).T
#   Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
#   Train_Acc = np.array([Train_Acc]).T
#   Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
#   Valid_Acc = np.asarray([Valid_Acc]).T


#   Pool_Train_Loss = Train_Loss
#   Pool_Valid_Loss = Valid_Loss
#   Pool_Train_Acc = Train_Acc
#   Pool_Valid_Acc = Valid_Acc

#   print('Evaluating Test Accuracy Without Acquisition')
#   score, acc = model.evaluate(X_Test, Y_Test, show_accuracy=True, verbose=0)

#   all_accuracy = acc

#   print('Starting Active Learning in Experiment ', e)

#   for i in range(acquisition_iterations):
#     print('POOLING ITERATION', i)


#     pool_subset = 20  # sample all the remaining unlabeled set
#     pool_subset_dropout = np.asarray(random.sample(range(0,X_Pool.shape[0]), pool_subset)) #sample a subset of unlabeled set
#     X_Pool_Dropout = X_Pool[pool_subset_dropout, :, :, :]  #sample a subset of unlabeled set
#     Y_Pool_Dropout = Y_Pool[pool_subset_dropout]  #sample a subset of unlabeled set

#     score_All = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))

#     for d in range(dropout_iterations):
#       print ('Dropout Iteration', d)
#       dropout_score = model.predict_stochastic(X_Pool_Dropout,batch_size=batch_size, verbose=1)
      
#       #np.save(''+'Dropout_Score_'+str(d)+'Experiment_' + str(e)+'.npy',dropout_score)
#       score_All = score_All + dropout_score

#     Avg_Pi = np.divide(score_All, dropout_iterations)
#     Log_Avg_Pi = np.log2(Avg_Pi)
#     Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
#     Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

#     U_X = Entropy_Average_Pi

#     # THIS FINDS THE MINIMUM INDEX 
#     # a_1d = U_X.flatten()
#     # X_Pool_index = a_1d.argsort()[-Queries:]

#     a_1d = U_X.flatten()
#     X_Pool_index = U_X.argsort()[-Queries:][::-1]

#     X_Pool_All = np.append(X_Pool_All, X_Pool_index)
#     print (X_Pool_index)

#       #saving pooled images
#     # for im in range(X_Pool_index[0:2].shape[0]):
#     #   Image = X_Pool[X_Pool_index[im], :, :, :]
#     #   img = Image.reshape((28,28))
#       #sp.misc.imsave('/home/ri258/Documents/Project/Active-Learning-Deep-Convolutional-Neural-Networks/ConvNets/Cluster_Experiments/Dropout_Max_Entropy/Pooled_Images/'+ 'Exp_'+str(e) + 'Pool_Iter'+str(i)+'_Image_'+str(im)+'.jpg', img)

#     Pooled_X = X_Pool_Dropout[X_Pool_index, :,:,]
#     Pooled_Y = Y_Pool_Dropout[X_Pool_index] 

#     #first delete the random subset used for test time dropout from X_Pool
#     #Delete the pooled point from this pool set (this random subset)
#     #then add back the random pool subset with pooled points deleted back to the X_Pool set
#     delete_Pool_X = np.delete(X_Pool, (pool_subset_dropout), axis=0)
#     delete_Pool_Y = np.delete(Y_Pool, (pool_subset_dropout), axis=0)

#     delete_Pool_X_Dropout = np.delete(X_Pool_Dropout, (X_Pool_index), axis=0)
#     delete_Pool_Y_Dropout = np.delete(Y_Pool_Dropout, (X_Pool_index), axis=0)

#     X_Pool = np.concatenate((X_Pool, X_Pool_Dropout), axis=0)
#     Y_Pool = np.concatenate((Y_Pool, Y_Pool_Dropout), axis=0)

#     print('Acquised Points added to training set')

#     X_Train = np.concatenate((X_Train, Pooled_X), axis=0)
#     Y_Train = np.concatenate((Y_Train, Pooled_Y), axis=0)

#     print('Train Model with pooled points')

#     # convert class vectors to binary class matrices
#     Y_Train = np_utils.to_categorical(Y_Train, nb_classes)


#     model = Sequential()
#     model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#     model.add(Dropout(0.25))


#     c = 3.5
#     Weight_Decay = c / float(X_Train.shape[0])
#     model.add(Flatten())
#     model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(nb_classes))
#     model.add(Activation('softmax'))


#     model.compile(loss='categorical_crossentropy', optimizer='adam')
#     hist = model.fit(X_Train, Y_Train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_Valid, Y_Valid))
#     Train_Result_Optimizer = hist.history
#     Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
#     Train_Loss = np.array([Train_Loss]).T
#     Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
#     Valid_Loss = np.asarray([Valid_Loss]).T
#     Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
#     Train_Acc = np.array([Train_Acc]).T
#     Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
#     Valid_Acc = np.asarray([Valid_Acc]).T

#     #Accumulate the training and validation/test loss after every pooling iteration - for plotting
#     Pool_Valid_Loss = np.append(Pool_Valid_Loss, Valid_Loss, axis=1)
#     Pool_Train_Loss = np.append(Pool_Train_Loss, Train_Loss, axis=1)
#     Pool_Valid_Acc = np.append(Pool_Valid_Acc, Valid_Acc, axis=1)
#     Pool_Train_Acc = np.append(Pool_Train_Acc, Train_Acc, axis=1) 

#     print('Evaluate Model Test Accuracy with pooled points')

#     score, acc = model.evaluate(X_Test, Y_Test, show_accuracy=True, verbose=0)
#     print('Test score:', score)
#     print('Test accuracy:', acc)
#     all_accuracy = np.append(all_accuracy, acc)

#     print('Use this trained model with pooled points for Dropout again')

#   print('Storing Accuracy Values over experiments')
#   Experiments_All_Accuracy = Experiments_All_Accuracy + all_accuracy


#   print('Saving Results Per Experiment')
#   np.save('./Results/'+'Max_Entropy_Pool_Train_Loss'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Loss)
#   np.save('./Results/'+'Max_Entropy_Pool_Valid_Loss'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Loss)
#   np.save('./Results/'+'Max_Entropy_Pool_Train_Acc'+ 'Experiment_' + str(e) + '.npy', Pool_Train_Acc)
#   np.save('./Results/'+'Max_Entropy_Pool_Valid_Acc'+ 'Experiment_' + str(e) + '.npy', Pool_Valid_Acc)
#   np.save('./Results/'+'Max_Entropy_Pool_Image_index'+ 'Experiment_' + str(e) + '.npy', X_Pool_All)
#   np.save('./Results/'+'Max_Entropy_Accuracy_Results'+ 'Experiment_' + str(e) + '.npy', all_accuracy)


# print('Saving Average Accuracy Over Experiments')
# Average_Accuracy = np.divide(Experiments_All_Accuracy, Experiments)
# np.save('./Results/'+'Max_Entropy_Average_Accuracy'+'.npy', Average_Accuracy)


