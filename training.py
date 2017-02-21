import random
random.seed(1)

import numpy
numpy.random.seed(7)

import json
import pickle
from datasets import get_train_data
from keras.utils import np_utils
from models import get_model_config
from utils import profile

#(x_train, y_train), (x_val, y_val) = mnist.load_data()

@profile
def training(model_type, x_train, y_train, x_val, y_val):
    '''
    format of mnist:
    ((([x_train], dtype = unit8), ([y_train])), (([x_val]), dtype = unit8), ([y_val])))
    '''
    # Print out size of training sample
    size_of_batch = x_train.shape[0]
    print(size_of_batch)

    # One hot encoding -> converts the 26 alphabets(represented as integers) to a categorical system where the machine understands
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    num_classes = y_val.shape[1]

    # Build the model
    model = model_type(num_classes)
    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=100, batch_size=64, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(x_val, y_val, verbose=0)

    return scores, model

if __name__ == '__main__':
    model_name = input('Select model:(baseline/simple_CNN/larger_CNN)\n')
    del_val_from_train = input('Do you want to remove validation images([y]/n)?\n')

    # get model configuration
    model_config = get_model_config(model_name)

    # del_val_from_train is set to True by default
    if del_val_from_train == 'n':
        del_val_from_train = False
    else:
        del_val_from_train = True
    (x_train, y_train), (x_val, y_val) = get_train_data(del_val_from_train)

    #Runs model
    scores, model = training(model_config['model_builder'], x_train, y_train, x_val, y_val)
    print(type(model))

    #Saves model weights
    model.save_weights(model_config['filepath_weight'])
    print('Model weights saved in {}.'.format(model_config['filepath_weight']))

    #saves model architechture
    with open(model_config['filepath_architechture'], 'w') as outfile:
        outfile.write(model.to_json())
    print('Model architechture saved in {}.'.format(model_config['filepath_architechture']))

    print('{} validation error: {}%'.format(model_name, 100 - scores[1] * 100))
