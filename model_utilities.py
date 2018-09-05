from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2, activity_l2


def build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, train_num, c_param = 3.5):
    model = Sequential()
    model.add(
        Convolution2D(
            nb_filters,
            nb_conv,
            nb_conv,
            border_mode='valid',
            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    c = 3.5
    Weight_Decay = c / float(train_num)
    model.add(Flatten())
    model.add(Dense(128, W_regularizer=l2(Weight_Decay)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model
