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

    def __init__(self, X_train, y_train, input_shape, filters, kernel_size, 
        maxpool,  loss_function='categorical_crossentropy', nb_classes= 2, droput_iteration=20, dropout = 0.05):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

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
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        
        

        # model = Sequential()
        # model.add(Conv2D(filters, (kernel_size, kernel_size), padding='same',
        #                  input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(Conv2D(filters, (kernel_size, kernel_size)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(maxpool, maxpool)))
        # model.add(Dropout(dropout))        
        # c = 3.5
        # Weight_Decay = c / float(X_train.shape[0])
        # model.add(Flatten())
        # model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
        # model.add(Activation('relu'))
        # model.add(Dropout(dropout))
        # model.add(Dense(nb_classes))
        # model.add(Activation('softmax'))

        # model.compile(loss=loss_function, optimizer='adam')

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
        # # We iterate the learning process
        # model.fit(X_train, y_train, batch_size=self.batch_size, nb_epoch=n_epochs, verbose=0)
        
        # #function for bayesian inference using dropouts
        # self.f = K.function([model.layers[0].input, K.learning_phase()],
        #        [model.layers[-1].output])

    def fit_gen(self, dataGen, X_Train, y_train, batch_size, nb_epochs):
        self.model.fit_generator(dataGen.flow(X_Train, y_train, batch_size = batch_size)
            , steps_per_epoch = len(X_Train)/ batch_size, epochs = nb_epochs, verbose = 1)

        self.f = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        # for e in range(nb_epochs):
        #     print ('Epoch', e)
        #     batches = 0
        #     for v in dataGen.flow(X_Train, y_train, batch_size = batch_size):
        #         batches +=1
        #         print (v)
        #         if batches >= len(X_Train)/batch_size:
        #             break
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
