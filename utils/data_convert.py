import numpy as np
import pickle
from pprint import pprint
from time import time

import skimage.io as skio
import matplotlib.pyplot as plt

def convert(num):
	with open('/home/li/Downloads/train.csv', 'r') as read:
		arrays = []
		answer = []
		data = []
		for line in read:
			doc = line.split(',')
			if doc[0] == 'Id':
				continue
			c = [int(ele) for ele in doc[4:]]
			x = np.asarray(c, dtype = np.int)
			x = x.reshape((16, 8))
			y = doc[1]
			arrays.append(x)
			answer.append(y)

		data.append(arrays)
		data.append(answer)

	if int(num) == 1:
		for i in range(15):
			skio.imshow(data[i][0])
			print(data[i][1])
			plt.show()

	else:
		with open('../data/training_image.pkl', 'wb') as w:
			pickle.dump(data, w)

if __name__ == '__main__':
	num = input('Enter a number: 1 == show 15 images, 2 == save to file \nInput: ')
	start = time()
	convert(num)
	print('The program ran for: {}'.format(time() - start))
