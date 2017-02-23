from skimage.transform  import resize, rotate

def resizing(image, image_size_tuple):
    image = resize(image, (16,16))
    return image

def rotating(image, angle_int):
    image = rotate(image, angle_int)
    return image

if __name__ == '__main__':
    print('something')
