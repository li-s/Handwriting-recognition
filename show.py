import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
from datasets import convert
from random import randint

def show():
	array, label, index = convert('test')
	images_3 = []
	for i in range(0, 3):
		position = randint(0, len(array))
		images_3.append(array[position])
	return images_3

if __name__ == '__main__':
	images_3 = show()
	for image in images_3:
		img = np.asarray(image)
		skio.imshow(img)
		plt.show()
