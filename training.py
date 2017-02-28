import numpy
numpy.random.seed(1701)

from keras.utils import np_utils
from keras.optimizers import SGD

from datasets import get_train_data
from models import get_model_config, get_model_name, save_model
from utils import profile

#(x_train, y_train), (x_val, y_val) = mnist.load_data()

@profile
def training(model_type, x_train, y_train, x_val, y_val, image_shape):
    '''
    format of mnist:
    ((([x_train], dtype = unit8), ([y_train])), (([x_val]), dtype = unit8), ([y_val])))
    '''
    # One hot encoding -> converts the 26 alphabets(represented as integers) to a categorical system where the machine understands
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    num_classes = y_val.shape[1]

    # Build the model
    model = model_type(num_classes, image_shape)
    model.summary()

    # Compile model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=50, batch_size=60, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(x_val, y_val, verbose=0)
    print('Validation error: {}%'.format(100 - scores[1] * 100))

    return scores, model

if __name__ == '__main__':
    model_name = get_model_name()
    del_val_from_train = input('Do you want to remove validation images([y]/n)?\n')

    # Del_val_from_train is set to True by default
    if del_val_from_train == 'n':
        del_val_from_train = False
    else:
        del_val_from_train = True
    (x_train, y_train), (x_val, y_val) = get_train_data(del_val_from_train)

    # Finds the image shape, and passes it to the model for reshape, initial image shape is unimportant
    image_shape = [1, 1]
    image_shape[0] = x_train.shape[1]
    image_shape[1] = x_train.shape[2]
    image_shape = tuple(image_shape)

    # Get model configuration
    model_config = get_model_config(model_name)

    # Runs model
    scores, model = training(model_config['model_builder'], x_train, y_train, x_val, y_val, image_shape)

    # Saves model
    save_model(model, model_config)
