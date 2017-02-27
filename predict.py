import numpy

from models import get_model_config, get_model_name, load_model

def predict(model_config):
    model = load_model(model_config['filepath_weight'], model_config['filepath_architechture'])

    # import image here

    prediction = model.predict(x_test, batch_size=32, verbose=0)

    result = numpy.argmax(prediction, axis=1)

    #result = numpy.amax(prediction, axis=1)
    result = result.tolist()
    result_alphabet = [chr(int(i) + ord('a')) for i in result]
    return result_alphabet, index

if __name__ == '__main__':
    # model_name = get_model_name()
    # model_config = get_model_config(model_name)

    # result_alphabet, index_result = predict(model_config)

    print('1')
