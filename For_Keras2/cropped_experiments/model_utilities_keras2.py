from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization


def build_model(nb_filters, nb_conv, nb_pool, input_shape, nb_classes, train_num, c_param = 3.5):
    model = Sequential()
    model.add(Conv2D(nb_filters,kernel_size=(nb_conv, nb_conv),input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(nb_filters*2,kernel_size=(nb_conv, nb_conv),input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model
