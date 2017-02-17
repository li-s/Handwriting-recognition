import numpy as np
import pickle
from pprint import pprint
from time import time
import random

random.seed(1)

def convert(filename, position = None):
	filepath = 'data/' + filename + '.csv'
	with open(filepath, 'r') as read:
		#Converts other file types to appropriate arrays
		array = []
		str_label = []
		for line in read:
			doc = line.split(',')
			if doc[0] == 'Id':
				continue
			c = [int(ele) for ele in doc[4:]]
			x = np.asarray(c, dtype = np.int)
			x = x.reshape((16, 8))
			x = x.tolist()
			y = doc[1]
			array.append(x)
			str_label.append(y)

	#convert a-b to their int counterparts
	label = []
	for i in str_label:
		a = ord(i) - ord('a')
		label.append(int(a))

	return array, label

def get_train_data(del_val_from_train = False):
	array, label = convert('train')
	#selects 1000 images for testing out of the 40,000++ train images
	a = random.sample(range(len(array)), 1000)

	val_array = []
	val_label = []
	for i in a:
		val_array.append(array[i])
		val_label.append(label[i])

	if del_val_from_train == False:
		train_array = array
		train_label = label
	#Removes validation images from train
	elif del_val_from_train == True:
		train_array = []
		train_label = []
		for i in range(len(array)):
			if not i in a:
				train_array.append(array[i])
				train_label.append(label[i])

	#format for train
	train = []
	train_array = np.asarray(train_array, dtype = np.int)
	train_label = np.asarray(train_label, dtype = np.int)
	train.append(train_array)
	train.append(train_label)

    #format for val
	val = []
	val_array = np.asarray(val_array, dtype = np.int)
	val_label = np.asarray(val_label, dtype = np.int)
	val.append(val_array)
	val.append(val_label)

	return train, val

def get_test_data():
	array, label = convert('test')
	test_array = np.asarray(array, dtype = np.int)
	test_label = np.asarray(label, dtype = np.int)
	return [test_array, test_label]

if __name__ == '__main__':
	#Convert datasets for training the model
	del_val_from_train = input('Do you want to remove validation images(y/n)?\n')
	if del_val_from_train == 'y':
		del_val_from_train = True
	elif del_val_from_train == 'n':
		del_val_from_train = False

	train, val = get_train_data(del_val_from_train)
	with open('data/val.pkl', 'wb') as w:
		pickle.dump(val, w)
	with open('data/train_removed_{}.pkl'.format(del_val_from_train), 'wb') as w:
		pickle.dump(train, w)

	test = get_test_data()
	with open('data/test.pkl', 'wb') as w:
		pickle.dump(test, w)
