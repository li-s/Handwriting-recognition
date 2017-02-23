from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.noise import GaussianDropout
from keras.regularizers import l2
from keras import backend as K
import os

K.set_image_dim_ordering('th')

def get_model_config(model_name):
    """
    Handle all configuaration in one place, so that no need to
    change filepaths elsewhere.
    """
    model_config = {}
    if model_name == 'baseline':
        model_config['model_builder'] = baseline_model
    elif model_name == 'simple_CNN':
        model_config['model_builder'] = simple_CNN_model
    elif model_name == 'larger_CNN':
        model_config['model_builder'] = larger_CNN_model
    else:
        raise Exception('Unknown model name.')
    model_config['filepath_weight'] = os.path.join('data', '{}_weight'.format(model_name))
    model_config['filepath_architechture'] = os.path.join('data', '{}_model'.format(model_name))
    return model_config

def baseline_model(num_classes, image_shape):
    model = Sequential()
    model.add(Reshape(int(image_shape[0] * image_shape[1]), input_shape = image_shape))
    model.add(Dense(128, input_dim=128, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    return model

def simple_CNN_model(num_classes, image_shape):
    model = Sequential()
    model.add(Reshape((1, image_shape[0], image_shape[1]), input_shape = image_shape))
    model.add(Convolution2D(32, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def larger_CNN_model(num_classes, image_shape): #need to reshape
    model = Sequential()
    model.add(Reshape((1, image_shape[0], image_shape[1]), input_shape = image_shape))
    # model.add(GaussianDropout(p = 0.5))
    model.add(Convolution2D(30, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(50, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.2))
		# model.add(Dense(256, activation='relu', W_regularizer=l2(0.1)))
    model.add(Dense(num_classes, activation='softmax'))
    return model
