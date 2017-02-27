import numpy as np
from skimage.transform import resize, rotate
import skimage.io as skio
import matplotlib.pyplot as plt
import inspect

import pickle
from pprint import pprint
import random


random.seed(1701)

def convert(split_name):
    filepath = 'data/' + split_name + '.csv'
    with open(filepath, 'r') as read:
        #Converts other file types to appropriate arrays
        a_list = []
        str_label = []
        index = []
        for line in read:
            doc = line.split(',')
            if doc[0] == 'Id':
                continue
            c = doc[1]
            a = [int(ele) for ele in doc[4:]]
            b = np.asarray(a, dtype = np.float32)
            b = b.reshape((16, 8))
            # Resize images
            a = b[~(b==0).all(1)]
            b = resize(a, (8,8))
            angle = random.randint(-15, 15)
            z = rotate(b, angle)
            skio.imshow_collection([a, b, z])
            plt.show()
            b = b.tolist()

            d = int(doc[0])
            a_list.append(b)
            str_label.append(c)
            index.append(d)

    #convert a-b to their int counterparts
    label = []
    for i in str_label:
        a = ord(i) - ord('a')
        label.append(int(a))

    return a_list, label, index

def get_train_data(del_val_from_train = False, num_val_sample = 4000):
    a_list, label, index = convert('train')
    #selects images for validation out of the 40,000++ train images
    a = random.sample(range(len(a_list)), num_val_sample)

    val_array = []
    val_label = []
    for i in a:
        val_array.append(a_list[i])
        val_label.append(label[i])

    if del_val_from_train == False:
        train_array = a_list
        train_label = label
    #Removes validation images from train
    elif del_val_from_train == True:
        train_array = []
        train_label = []
        for i in range(len(a_list)):
            if not i in a:
                train_array.append(a_list[i])
                train_label.append(label[i])

    #format for train
    train = []
    train_array = np.asarray(train_array, dtype = np.float32)
    train_label = np.asarray(train_label, dtype = np.int)
    train.append(train_array)
    train.append(train_label)

    #format for val
    val = []
    val_array = np.asarray(val_array, dtype = np.float32)
    val_label = np.asarray(val_label, dtype = np.int)
    val.append(val_array)
    val.append(val_label)

    return train, val

def get_test_data():
    a_list, label, index = convert('test')
    # test_list_of_arrays = []
    # for i in a_list:
    # 	a_array = np.asarray(i, dtype = np.int)
    # 	test_list_of_arrays.append(a_array)
    test_array = np.asarray(a_list, dtype = np.float32)
    test_label = np.asarray(label, dtype = np.int)

    return [test_array, test_label], index

if __name__ == '__main__':
    #Convert datasets for training the model
    del_val_from_train = input('Do you want to remove validation images(y/[n])?\n')
    if del_val_from_train == 'y':
        del_val_from_train = True
    else:
        del_val_from_train = False

    train, val = get_train_data(del_val_from_train)
    with open('data/val.pkl', 'wb') as w:
        pickle.dump(val, w)
    with open('data/train_removed_{}.pkl'.format(del_val_from_train), 'wb') as w:
        pickle.dump(train, w)

    (test_array, test_label), index = get_test_data()
    with open('data/test.pkl', 'wb') as w:
        pickle.dump(test_array, w)
