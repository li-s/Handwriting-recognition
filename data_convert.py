import numpy as np
from pprint import pprint

def convert():
	with open('./data/training_image(binary).txt', 'w') as w:
		with open('/home/li/Downloads/train.csv', 'r') as read:
			doc = []
			a = ''.join(read)
			b = a.split('\n')
			for line in b:
				c = line.split(',')
				doc.append(c)	#doc represents the entire train.csv
			doc2 = []
			for i in range(1, len(doc)):
				doc2.append(doc[i])	#doc2 represents the entire train.csv with each line as a seperate list
			
			matrix = np.zeros( (16, 8) )
			x = 0
			x_max = 7
			for i in range(0, len(doc)):	#literates over the number of training images
				y = 0
				for j in range(4, len(doc[0])):	#only literates over the binary part
					if x < x_max:	#
						matrix[y][x] = doc2[i][j]	# substitude the matrix for the binary
						x += 1
					else:
						print(x, y)
						y += 1
						x = 0
					
					
'''
	line 20: need to find way to initialise multiple numpy arrays at the same time(number of arrays = len(doc2)-1, as the first line is not important to us right now)
'''
					
			print(len(doc))
			pprint(matrix, stream = w)

convert()
