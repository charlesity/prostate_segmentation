'''
    Modified for active learning
'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.models import Sequential
from keras import backend as K

import time


class net:

    def __init__(self, X_train, y_train, input_shape, loss_function='categorical_crossentropy', nb_classes= 2, droput_iteration=20, dropout = 0.05):

        self.nb_classes = nb_classes
        self.droput_iteration = droput_iteration
        c = 3.5
        Weight_Decay = c / float(X_train.shape[0])

        model = Sequential()        
        model.add(Dense(256, input_shape =input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(256, W_regularizer=l2(Weight_Decay)))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss=loss_function, optimizer='adam')
        self.model = model

    def fit_gen(self, dataGen, X_Train, y_train, batch_size, nb_epochs):
        self.model.fit_generator(dataGen.flow(X_Train, y_train, batch_size = batch_size)
            , steps_per_epoch = len(X_Train)/ batch_size, epochs = nb_epochs, verbose = 1)

        #function for bayesian inference with dropouts
        self.f = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])

    def fit(self, X_Train, Y_Train, batches, nb_epochs, validation_split):
        self.model.fit(X_Train, Y_Train, batch_size=batches, verbose =1, validation_split=validation_split)

    def predict_stochastic(self, X_test):
        print ("Performing Bayesian Inference using Dropout")
        results = np.zeros((X_test.shape[0], self.nb_classes))
        for i in range(self.droput_iteration):
            results = results + self.f((X_test, 1))[0]
        prediction_hat = np.divide(results, self.droput_iteration)
        print ("Done Performing Bayesian Inference using Dropout")        
        return prediction_hat

    def predict_gen(self, predGen, X_Test, batch_size):
        return self.model.predict_generator(predGen.flow(X_Test, batch_size = batch_size), use_multiprocessing = True)


    def pred(self, X_test, batch_size):
        return self.model.predict(X_test, batch_size=batch_size, verbose=1)
