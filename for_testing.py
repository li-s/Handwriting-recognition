import numpy as np
from skimage.transform import resize

i = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 0]]

def a_resize():
    global i
    i = np.asarray(i, dtype = np.float64)
    print(i)
    i = i[~(i==0).all(1)]

    print(i)
    i = resize(i, (8, 8))
    print(i)

if __name__ == '__main__':
    a = a_resize()
