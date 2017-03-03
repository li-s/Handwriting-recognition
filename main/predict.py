import numpy as np
from skimage.transform import resize
import skimage.io as skio
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

from models import get_model_config, get_model_name, load_model
from utils import ask_for_file_particulars

def predict(model_config, file_location):
    model = load_model(model_config['filepath_weight'], model_config['filepath_architechture'])

    # Convert images to correct format
    a = []
    img = skio.imread(file_location)
    img = resize(img, (16, 16))
    img = img.tolist()
    a.append(img)
    img = np.asarray(a)
    x_test = img

    # Finds confidence of all 26 alphabets
    prediction = model.predict(x_test, batch_size=32, verbose=0)
    result = np.argmax(prediction, axis=1)
    result = result.tolist()
    for i in prediction:
        confidence = prediction[0][result]

    result_alphabet = [chr(int(i) + ord('a')) for i in result]
    confidence= Decimal(confidence[0]*100)

    confidence = Decimal(confidence.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
    return result_alphabet[0], confidence

if __name__ == '__main__':
    file_location = ask_for_file_particulars()
    model_config = get_model_config('larger_CNN')
    result_alphabet, confidence = predict(model_config, file_location)

    print('The model predicts alphabet: {} with {}% confidence'.format(result_alphabet, confidence))
