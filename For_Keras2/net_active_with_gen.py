'''
    Modified for active learning
'''
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.models import Sequential
from keras import backend as K
from sklearn.metrics import roc_auc_score, precision_score
import keras_metrics as k_metrics

import time


class net:

    def __init__(self, input_shape, filters, kernel_size,
        maxpool,  loss_function='binary_crossentropy', nb_classes= 2, droput_iteration=20, dropout = 0.5):

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        # if normalize:
        #     self.std_X_train = np.std(X_train, 0)
        #     self.std_X_train[ self.std_X_train == 0 ] = 1
        #     self.mean_X_train = np.mean(X_train, 0)
        # else:
        #     self.std_X_train = np.ones(X_train.shape[ 1 ])
        #     self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        self.droput_iteration = droput_iteration
        self.nb_classes = nb_classes

        model = Sequential()
        model.add(Conv2D(filters, (kernel_size, kernel_size), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(maxpool, maxpool)))
        model.add(Conv2D(filters, (kernel_size, kernel_size)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(maxpool, maxpool)))
        model.add(Dropout(dropout))
        # c = 3.5
        # Weight_Decay = c / float(X_train.shape[0])
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout))
        # model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss=loss_function, optimizer='adam')
        self.model = model

    def auroc(self, y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    def __prepare_stochastic_function(self):
        self.f = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])

    def fit_myGenerator(self, dataGen, nb_epochs):
        print ("Training with Generator...")
        self.history = self.model.fit_generator(dataGen, epochs=nb_epochs, workers=6, use_multiprocessing=True)
        self.__prepare_stochastic_function()
        print ("Done Training with Generator...")
        return self.history

    def predict_stochastic(self, X_test):        
        results = np.zeros((X_test.shape[0], self.nb_classes))
        for i in range(self.droput_iteration):
            results = results + self.f((X_test, 1))[0]
        prediction_hat = np.divide(results, self.droput_iteration)
        return prediction_hat

    def predict_gen(self, predGen):
        print ("Predicting with generator...")
        score = self.model.predict_generator(predGen, workers=6, use_multiprocessing = True)
        print ("Done Predicting with generator...")
        return score
    def evaluate_myGenerator(self, generator):
        print ("Evaluating model with generator")
        self.score= self.model.evaluate_generator(generator,  workers=6, use_multiprocessing=True)
        print ("Done evaluating model with generator")
        return self.score
    def pred(self, X_test, batch_size):
        print ("Predicting")
        return self.model.predict(X_test, batch_size=batch_size)
        print ("Done Predicting")
