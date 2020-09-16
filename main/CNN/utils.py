import time
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.transform import rescale
import numpy as np
from datasets import convert
import random as rn
import argparse

def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        print('\t%s comsumed %.1fs' % (func.__name__, time.time() - started_at))
        return result
    return wrap

def answer_convert(result_alphabet, index_result):
    """
    Input the test data with the trained model and tags the predicted label with the id
    """
    result_list = [(index_result[i], result_alphabet[i]) for i in range(len(index_result))]

    # Tags the train label with thier index
    array, label, index_train = convert('train')
    label_alphabet = [chr(int(i) + ord('a')) for i in label]
    train_list = [(index_train[i], label_alphabet[i]) for i in range(len(index_train))]

    # Final sorting
    final_answer_for_submission = result_list + train_list
    final_answer_for_submission = sorted(final_answer_for_submission)

    return final_answer_for_submission

def get_image_samples(n, show = False, save = True):
    array, label, index = convert('test')
    images = []

    positions = rn.sample(range(len(array)), n)
    for position in positions:
        images.append(array[position])

    for i, img in enumerate(images):
        img = np.asarray(img, dtype = np.float64)
        if show:
            skio.imshow(img)
            plt.show()
        if save:
            filepath = './data/show_image{}.jpg'.format(i)
            img = rescale(img, 4)
            skio.imsave(filepath, img)

def ask_for_file_particulars():
    parser = argparse.ArgumentParser(description="Input file location of image to test with trained model.")
    parser.add_argument("location", help="Location of the image")
    # parser.add_argument('length', help ='Length of the image', type = int)
    # parser.add_argument('width', help ='Width of the image', type = int)
    args = parser.parse_args()
    # size = (args.length, args.width)
    return args.location

if __name__ == '__main__':
    get_image_samples(3)
