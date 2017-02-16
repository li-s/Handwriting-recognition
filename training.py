import numpy
#from keras.datasets import mnist	#this is a set of images
from keras.utils import np_utils
import pickle
from model import baseline_model
import json

seed = 7
numpy.random.seed(seed)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()


def training(x_train, y_train, x_test, y_test):
	'''
	format of mnist:
	((([x_train], dtype = unit8), ([y_train])), (([x_test]), dtype = unit8), ([y_test])))
	'''
	# flatten 28*28 images to a 784 vector for each image
	num_pixels = x_train.shape[1] * x_train.shape[2]
	x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

	#one hot encoding -> converts the 26 alphabets(represented as integers) to a categorical system where the machine understands
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]

	# build the model
	model = baseline_model(num_pixels, num_classes)

	# Fit the model
	model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

	# Final evaluation of the model
	scores = model.evaluate(x_test, y_test, verbose=0)

#	model.summary()

	return scores, model

if __name__ == '__main__':
	with open('./data/train_image.pkl', 'rb') as a:
		mydataset = pickle.load(a)
		(x_train, y_train) = mydataset

	with open('./data/test_image.pkl', 'rb') as a:
		mydataset = pickle.load(a)
		(x_test, y_test) = mydataset

	scores, model = training(x_train, y_train, x_test, y_test)

	#saves model weights
	a = input('Model name: ')
	filepath = './data/' + str(a) + '.m'
	model.save_weights(filepath)

	#saves model architechture
	filepath = './data/' + str(a) + '.json'
	with open(filepath, 'w') as outfile:
		json.dump(model.to_json(), outfile)
	print('Baseline Error: {}%'.format(100-scores[1]*100))
