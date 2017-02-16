import numpy as np
import pickle
from pprint import pprint
from time import time

import skimage.io as skio
import matplotlib.pyplot as plt

def convert_train(num):
	with open('/home/li/Downloads/train.csv', 'r') as read:
		arrays = []
		str_answer = []
		data = []
		for line in read:
			doc = line.split(',')
			if doc[0] == 'Id':
				continue
			c = [int(ele) for ele in doc[4:]]
			x = np.asarray(c, dtype = np.int)
			x = x.reshape((16, 8))
			x = x.tolist()
			y = doc[1]
			arrays.append(x)
			str_answer.append(y)
		#makes the training sample for keras

		int_answer = []
		for i in str_answer:
			int_answer.append(ord(i) - ord('a'))
		#convert a-b to their int counterparts

		arrays = np.asarray(arrays)
		int_answer = np.asarray(int_answer, dtype = np.int)
		data.append(arrays)
		data.append(int_answer)
		#format

	#dont use this -- code changed, no longer functions
	if int(num) == 1:
		for i in range(15):
			skio.imshow(data[i][0])
			print(data[i][1])
			plt.show()

	else:
		with open('../data/train_image.pkl', 'wb') as w:
			pickle.dump(data, w)

def convert_test():
	with open('/home/li/Downloads/test.csv', 'r') as read:
		arrays = []
		str_answer = []
		data = []
		for line in read:
			doc = line.split(',')
			if doc[0] == 'Id':
				continue
			c = [int(ele) for ele in doc[4:]]
			x = np.asarray(c, dtype = np.int)
			x = x.reshape((16, 8))
			x = x.tolist()
			y = doc[1]
			arrays.append(x)
			str_answer.append(y)
		#makes the training sample for keras

		int_answer = []
		for i in str_answer:
			int_answer.append(ord(i) - ord('a'))
		#convert a-b to their int counterparts

		arrays = np.asarray(arrays)
		int_answer = np.asarray(int_answer, dtype = np.int)
		data.append(arrays)
		data.append(int_answer)
		#format

	with open('../data/test_image.pkl', 'wb') as w:
		pickle.dump(data, w)

if __name__ == '__main__':
	num = input('Enter a number: 1(unprogrammed) == show 15 images, 2 == save to file \nInput: ')
	start = time()
	convert_train(num)
	convert_test()
print('The program ran for: {}'.format(time() - start))
