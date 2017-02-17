# Handwriting recognition

### Supervised vs unsupervised machine learning

A machine undergoing supervised learning has 'correct' outputs for each input, and is trained to generate the specific output to a minimal degree of error.
On the other hand, a machine undergoing unsupervised learning does not have correct outputs for the inputs. The machine is thus trained to generate plausible outputs for each individual output.

### Image classification

Image classification is the task of assigning an input image one label from a fixed set of categories.

### Handwriting recognition

The ability of a computer to interpret handwritings from texts, such as photographs, paper texts, or pdfs. The computer is able to find the most plausible words for a given text with unclear words. (write more about the specific task of the Kaggle competition, like what is the input/output, the specific problem formulation and evaluation metric)

#### Usage
1. `datasets.py`: Data should be in seperate folder 'data', in csv format. (look in sample.csv)
2. `training.py`: Trains the model (which you select), and saves it in seperate folder 'data'. There is a choice to remove validation images from training images
3. `testing.py`: Loads and trains the model from file 'test.csv' in folder entitled 'data'
4. `answer_convert.py`: Converts the answers

#### Dependencies
1. [numpy](http://www.numpy.org/)
2. [scikit-image](http://scikit-image.org/)
3. [matplotlib](http://matplotlib.org/)
4. [keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
5. [tensorflow](https://www.tensorflow.org/) (for keras machine learning)

## Miscellaneous
+ [one hot encoding](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science)
+ [Task](https://inclass.kaggle.com/c/cs5339-prediction-competition)
+ [What is an intuitive explanation for neural networks?](https://www.quora.com/What-is-an-intuitive-explanation-for-neural-networks)
+ [What is an intuitive explanation of overfitting?](https://www.quora.com/What-is-an-intuitive-explanation-of-overfitting)
