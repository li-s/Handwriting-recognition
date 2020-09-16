# Handwritten character recognition

## Introduction

Handwritten character recognition is a typical task of image classification, which is to assigning an input image one label from a fixed set of categories. It enables the ability of a computer to interpret handwritings from images, photographs, paper texts, or pdfs. The computer finds the most plausible words for a given text with unclear words. This project developed a toolbox to recognize an English character from an image.

In this repository, I am using Keras to train a model in handritting recognition from a training sample of around 40,000 characters of the english alphabet, located in `train.csv` within seperate folder `data`. After which, the model is saved and loaded to test its accuracy against 10,000 characters from `test.csv`, located within the same folder `data`. Although I tried different approaches to increase the training and validation accuracy in the training stage, final accuracy of the testing data plateus at around 6 to 7 percentage points, seen in the table below. The main functions can be found in folder `main`.

## Getting started

### Prerequisites
+ [Python3](https://www.python.org/download/releases/3.0/)
+ [numpy](http://www.numpy.org/)
+ [matplotlib](http://matplotlib.org/)
+ [scikit-image](http://scikit-image.org/)
+ [keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
+ [tensorflow](https://www.tensorflow.org/) (for keras machine learning)
+ [skLearn](https://scikit-learn.org/stable/index.html) (For datasets)

### Usage
1. Run `python3 predict.py [image_path]`, to get the prediction result from a given image path.
2. Run `python3 training.py`, to train a new model. Training data specification is mentioned API reference.
3. Run `python3 testing.py`, to test the current model. Test images should be in csv format in seperate folder `data`.

## Testing results

The provided model is trained and tested using the data from a [Kaggle competition](https://inclass.kaggle.com/c/cs5339-prediction-competition). The accuracy of the model according to different parameter settings is shown in the table below:

| # Model | Model fit type | Epoch | Batch size | Testing accuray |
| ------- | -------------- | ----- | ---------- | --------------- |
| larger_CNN (convolution2D:30x3x3, 50x3x3) | Normal | 100 | 60 | 93.33% |
| larger_CNN (convolution2D:30x3x3, 50x3x3) | Generator | 50 | 60 | 92.84% |
| larger_CNN (convolution2D:30x3x3, 50x3x3) | Generator | 50 | 60 | 92.66% |
| larger CNN (convolution2D:30x5x5, 30X3X3) | Normal | 100 | 64 | 91.98% |
| simple CNN (convolution2D:32X5X5) | Normal |100 | 64 | 90.74% |
| Baseline (2 layers) | Normal | 100 | 20 | 86.46% |
| Baseline (2 layers) | Normal | 200 | 64 | 85.03% |

Some real prediction examples is shown below:  
<img src="https://github.com/li-s/Handwriting-recognition/blob/master/data/show_image0.jpg" height="30">: "s" (conf. = 99.83%),
<img src="https://github.com/li-s/Handwriting-recognition/blob/master/data/show_image1.jpg" height="30">: "f" (conf. = 52.98%),
<img src="https://github.com/li-s/Handwriting-recognition/blob/master/data/show_image2.jpg" height="30">: "i" (conf. = 100.00%).

## API reference
+ `datasets.py`: Data should be in seperate folder 'data', in csv format. (look in sample.csv)
+ `models.py`: Different models to train with, along with method to save, load, and call models.
+ `training.py`: Trains the model (which you select), and saves it in seperate folder 'data'. There is a choice to remove validation images from training images. There is choice to use normal fit or fit generator.
+ `testing.py`: Trains the model from file 'test.csv' in folder entitled 'data'.
+ `predict.py`: Predicts the alphabet in the image the user passes to it.
+ `utils.py`: Hosts miscellaneous programs, for keeping time, converting answers, or getting image samples etc.

## Miscellaneous

### Supervised vs unsupervised machine learning
A machine undergoing supervised learning has 'correct' outputs for each input, and is trained to generate the specific output to a minimal degree of error.
On the other hand, a machine undergoing unsupervised learning does not have correct outputs for the inputs. The machine is thus trained to generate plausible outputs for each individual output.

### Readings
+ [What is the difference between supervised and unsupervised learning algorithms?](https://www.quora.com/What-is-the-difference-between-supervised-and-unsupervised-learning-algorithms)
+ [One hot encoding](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)
+ [What is an intuitive explanation for neural networks?](https://www.quora.com/What-is-an-intuitive-explanation-for-neural-networks)
+ [What is an intuitive explanation of overfitting?](https://www.quora.com/What-is-an-intuitive-explanation-of-overfitting)
+ [Neural networks](http://cs231n.github.io/)
