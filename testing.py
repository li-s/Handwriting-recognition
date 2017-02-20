import json
from keras.models import model_from_json
import numpy
from datasets import get_test_data

def load_model(model_name):
    filepath_architechture = 'data/' + model_name + '.json'
    with open(filepath_architechture, 'r') as read:
        a = read.readlines()
        model = model_from_json(a[0])

    filepath_weights = 'data/' + model_name + '.m'
    model.load_weights(filepath_weights, by_name=False)

    return model

def prediction(model_name):
    model = load_model(model_name)

    (x_test, test_label), index = get_test_data()

    prediction = model.predict(x_test, batch_size=32, verbose=0)

    result = numpy.argmax(prediction, axis=1)

    #result = numpy.amax(prediction, axis=1)
    result = result.tolist()
    result_alphabet = [chr(int(i) + ord('a')) for i in result]
    return result_alphabet, index

if __name__ == '__main__':
    select = input('Enter the model name you are using(1 = baseline model, 2 = simple_CNN_model, 3 = larger_CNN_model)\n')
    if int(select) == 1:
        model_name = 'baseline model'
        a, b = prediction(model_name)

    elif int(select) == 2:
        model_name = 'simple_CNN_model'
        a, b = prediction(model_name)

    elif int(select) == 3:
        model_name = 'larger_CNN_model'
        a, b = prediction(model_name)

    print(a)
