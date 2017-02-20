import json
import numpy
from keras.models import model_from_json
from datasets import get_test_data
from models import get_model_config
from utils import answer_convert

def load_model(filepath_weights, filepath_architechture):
    with open(filepath_architechture, 'r') as read:
        a = read.readlines()
        model = model_from_json(a[0])

    model.load_weights(filepath_weights, by_name=False)

    return model

def prediction(model_config):
    model = load_model(model_config['filepath_weight'], model_config['filepath_architechture'])

    (x_test, test_label), index = get_test_data()

    prediction = model.predict(x_test, batch_size=32, verbose=0)

    result = numpy.argmax(prediction, axis=1)

    #result = numpy.amax(prediction, axis=1)
    result = result.tolist()
    result_alphabet = [chr(int(i) + ord('a')) for i in result]
    return result_alphabet, index

if __name__ == '__main__':
    model_name = input('Select model:(baseline/simple_CNN/larger_CNN)\n')
    model_config = get_model_config(model_name)

    result_alphabet, index_result = prediction(model_config)
    
    final_answer = answer_convert(result_alphabet, index_result)
    with open('./data/final_answer.csv', 'w') as w:
        w.write('Id,Prediction\n')
        for i, j in final_answer:
            w.write('{},{}\n'.format(i, j))
