import numpy as np
import pickle
from pprint import pprint
from time import time
from random import sample
import skimage.io as skio
import matplotlib.pyplot as plt

def convert_train(num):
	with open('/home/li/Downloads/train.csv', 'r') as read:
		#makes the training sample for keras
		train_array = []
		str_answer = []
		train = []
		for line in read:
			doc = line.split(',')
			if doc[0] == 'Id':
				continue
			c = [int(ele) for ele in doc[4:]]
			x = np.asarray(c, dtype = np.int)
			x = x.reshape((16, 8))
			x = x.tolist()
			y = doc[1]
			train_array.append(x)
			str_answer.append(y)

		#convert a-b to their int counterparts
		int_answer = []
		for i in str_answer:
			a = ord(i) - ord('a')
			int_answer.append(int(a))

		#select 1000 images for testing out of the 40,000++ train images
		a = sample(range(len(train_array)), 1000)
		test_array = []
		test_answer = []
		test = []
		for i in a:
			test_array.append(train_array[i])
			test_answer.append(int_answer[i])

		#removal of the 1000 images from train\
		print(len(train_array))
		train_array = [a_list for a_list in train_array if a_list not in test_array]
		print(len(test_array))
		print(len(train_array))

		#format for train
		train_array = np.asarray(train_array)
		int_answer = np.asarray(int_answer, dtype = np.int)
		train.append(train_array)
		train.append(int_answer)

		#format for test
		test_array = np.asarray(test_array, dtype = np.int)
		test_answer = np.asarray(test_answer, dtype = np.int)
		test.append(test_array)
		test.append(test_answer)

	#dont use this -- code changed, no longer functions
	if int(num) == 1:
		for i in range(15):
			skio.imshow(train[i][0])
			print(train[i][1])
			plt.show()

	#save train, test into file
	else:
		with open('../data/train_image.pkl', 'wb') as w:
			pickle.dump(train, w)
		with open('../data/test_image.pkl', 'wb') as w:
			pickle.dump(test, w)

if __name__ == '__main__':
	num = input('Enter a number: 1(unprogrammed) == show 15 images, 2 == save to file \nInput: ')
	start = time()
	convert_train(num)
print('The program ran for: {}'.format(time() - start))
