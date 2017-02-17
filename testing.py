import json
from keras.models import model_from_json
import numpy
from datasets import get_test_data

def load_model():
    with open('./data/baseline.json', 'r') as read:
        a = read.readlines()
        model = model_from_json(a[0])

    model.load_weights('./data/baseline.m', by_name=False)

    return model

def prediction():
    model = load_model()

    (x_test, test_label), index = get_test_data()
    num_pixels = x_test.shape[1] * x_test.shape[2]
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

    prediction = model.predict(x_test, batch_size=32, verbose=0)

    result = numpy.argmax(prediction, axis=1)

    #result = numpy.amax(prediction, axis=1)
    result = result.tolist()
    result_alphabet = [chr(int(i) + ord('a')) for i in result]
    return result_alphabet, index

if __name__ == '__main__':
    a, b = prediction()
    print(a)
