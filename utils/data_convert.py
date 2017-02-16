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

		#selects 1000 images for testing out of the 40,000++ train images and removes it from the array
		train_array = []
		train_answer = []
		test_array = []
		test_answer = []
		a = sample(range(len(array)), 1000)
		for i in range(len(array)):
			if i in a:
				test_array.append(array[i])
				test_answer.append(answer[i])

			else:
				train_array.append(array[i])
				train_answer.append(answer[i])


		#format for train
		train = []
		train_array = np.asarray(train_array, dtype = np.int)
		train_answer = np.asarray(train_answer, dtype = np.int)
		train.append(train_array)
		train.append(train_answer)

		#format for test
		test = []
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
