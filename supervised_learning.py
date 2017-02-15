import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
from keras.utils import np_utils
import pickle

seed = 7
numpy.random.seed(seed)

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
'''
format of mnist:
((([x_train], dtype = unit8), ([y_train])), (([x_test]), dtype = unit8), ([y_test])))
'''
with open('./data/training_image.pkl', 'rb') as a:
	mydataset = pickle.load(a)
	(x_train, y_train) = mydataset

'''
Reminder: initialize y_test and x_test
'''

# flatten 28*28 images to a 784 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
'''
reshape(x_train.shape[0], num_pixels) = what does 0 and num_pixels input do?
'''

'''
Not implemented:
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
'''
# # normalize inputs from 0-255 to 0-1
# x_train = x_train / 255
# x_test = x_test / 255
#
# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
#
# def baseline_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
# 	model.add(Dense(num_classes, init='normal', activation='softmax'))
#
# 	# Compile model
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	return model
#
# # build the model
# model = baseline_model()
#
# # Fit the model
# model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
#
# # Final evaluation of the model
# scores = model.evaluate(x_test, y_test, verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
