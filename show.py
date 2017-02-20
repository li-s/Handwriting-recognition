import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.transform import rescale
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
    for i in range(0, len(images_3)):
        img = np.asarray(images_3[i], dtype = np.uint8)
        img = img * 255
        skio.imshow(img)
        img = rescale(img, 4)
        filepath = './data/show_image' + str(i) + '.jpg'
        skio.imsave(filepath, img)
        # skio.imsave('./data/show_image.jpg', img)
        plt.show()
