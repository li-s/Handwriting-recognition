# Handwritten character recognition

Handwritten character recognition is a typical task of image classification, which is to assigning an input image one label from a fixed set of categories. It enables the ability of a computer to interpret handwritings from texts, such as photographs, paper texts, or pdfs. So the computer is able to find the most plausible words for a given text with unclear words. This project developed a toolbox to recognize an English character from an image.

## Getting Started

#### Prerequisites
+ Python3
+ [numpy](http://www.numpy.org/)
+ [matplotlib](http://matplotlib.org/)
+ [keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
+ [tensorflow](https://www.tensorflow.org/) (for keras machine learning)

#### Usage
1. ~~Run `python3 predict.py [image_path]`, to get the prediction result from a given image path.~~
2. Run `python3 training.py`, to train a new model. Training data specification is mentioned API reference.

## Testing results

The provided model is trained and tested using the data from a [Kaggle competition](https://inclass.kaggle.com/c/cs5339-prediction-competition). The accuracy of the model according to different parameter settings is showed in the table below:

| # epoch | batch size | testing accuray |
| ------- | ---------- | --------------- |
| 100 | 20 | 86.46% |
| 200 | 64 | 85.03% |

Some real prediction examples is showed below:

## API reference

+ ~~Scripts and important functions that are not mentioned in Usage~~
+ `datasets.py`: Data should be in seperate folder 'data', in csv format. (look in sample.csv)
+ `training.py`: Trains the model (which you select), and saves it in seperate folder 'data'. There is a choice to remove validation images from training images
+ `testing.py`: Loads and trains the model from file 'test.csv' in folder entitled 'data'
+ `answer_convert.py`: Converts the answers

## Miscellaneous

#### Supervised vs unsupervised machine learning
A machine undergoing supervised learning has 'correct' outputs for each input, and is trained to generate the specific output to a minimal degree of error.
On the other hand, a machine undergoing unsupervised learning does not have correct outputs for the inputs. The machine is thus trained to generate plausible outputs for each individual output.

#### Readings
+ [One hot encoding](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)
+ [What is an intuitive explanation for neural networks?](https://www.quora.com/What-is-an-intuitive-explanation-for-neural-networks)
+ [What is an intuitive explanation of overfitting?](https://www.quora.com/What-is-an-intuitive-explanation-of-overfitting)
