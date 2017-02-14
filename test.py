import numpy
from pprint import pprint

def convert():
	with open('./data/training_image(binary).txt', 'w') as w:
		with open('/home/li/Downloads/train.csv', 'r') as read:
			doc = []
			a = ''.join(read)
			b = a.split('\n')
			for line in b:
				c = line.split(',')
				doc.append(c)
#			
#			for i in range(1, len(b)):
#				doc.append([b[i]])
#			
			pprint(doc, stream = w)

convert()
