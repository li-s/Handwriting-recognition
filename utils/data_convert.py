import numpy as np
import pickle
from pprint import pprint
from time import time
from random import sample

def convert_train(option):
	with open('/home/li/Downloads/train.csv', 'r') as read:
		#makes the training sample for keras
		array = []
		str_answer = []
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
			str_answer.append(y)

		#convert a-b to their int counterparts
		answer = []
		for i in str_answer:
			a = ord(i) - ord('a')
			answer.append(int(a))

        if option == y:
    		#selects 1000 images for testing out of the 40,000++ train images and removes it from the array
    		train_array = []
    		train_answer = []
    		val_array = []
    		val_answer = []
    		a = sample(range(len(array)), 1000)
    		for i in range(len(array)):
    			if i in a:
    				val_array.append(array[i])
    				val_answer.append(answer[i])

    			else:
    				train_array.append(array[i])
    				train_answer.append(answer[i])
        	#format for val
            val = []
            val_array = np.asarray(val_array, dtype = np.int)
            val_answer = np.asarray(val_answer, dtype = np.int)
            val.append(val_array)
            val.append(val_answer)

        elif int(option) == n:
            train_array = array
    		train_answer = answer
            val = None

		#format for train
		train = []
		train_array = np.asarray(train_array, dtype = np.int)
		train_answer = np.asarray(train_answer, dtype = np.int)
		train.append(train_array)
		train.append(train_answer)

		return train, val

if __name__ == '__main__':
	start = time()
    option = input('Do you want to initiate testing images(y/n)?\n')
	train, val = convert_train(option)
	with open('../data/train_image.pkl', 'wb') as w:
		pickle.dump(train, w)
	with open('../data/val_image.pkl', 'wb') as w:
		pickle.dump(val, w)
	print('The program ran for: {}'.format(time() - start))
