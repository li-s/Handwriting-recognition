from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

K.set_image_dim_ordering('th')

# define baseline model
def baseline_model(num_classes):
    model = Sequential()
    model.add(Reshape((128,), input_shape = (16,8)))
    model.add(Dense(128, input_dim=128, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    return model

def simple_CNN_model(num_classes):
    model = Sequential()
    model.add(Reshape((1, 16, 8), input_shape = (16,8)))
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 16, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def larger_CNN_model(num_classes): #need to reshape
    model = Sequential()
    model.add(Reshape((1, 16, 8), input_shape = (16,8)))
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 16, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
