import numpy as np
from pprint import pprint
from datasets import convert
from skimage.transform import resize

def a_resize():
    a_list, label, index = convert('test')

    array = []
    for i in a_list:
        i = np.asarray(i, dtype = np.float64)
        array.append('1:')
        array.append(i)
        array.append('2:')
        i = i[~(i==0).all(1)]
        print(i)
        i = resize(i, (8, 8))
        print(i)
        array.append(i)
    return array
    #not sure what to do with array type float 64 with range 0 to 10^(-20)

if __name__ == '__main__':
    with open('./data/resized.txt', 'w') as w:
        a = a_resize()
        pprint(a, stream = w)
