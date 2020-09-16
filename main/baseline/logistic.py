import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_weights(dim):
    w = np.random.randn(dim, 1) * 0.1
    b = 0
    
    return w, b

def propogate(X, Y, w, b):
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - 1 / m * (np.sum((np.dot(Y, np.log(A).T), np.dot((1 - Y), np.log(1 - A).T))))
    
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propogate(X, Y, w, b)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if(A[0, i] > 0.5):
            Y_prediction[0, i] = 1

    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_weights(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b}

    return d

def clean(Y, number):
    clean_Y = []

    for i, num in enumerate(Y):
        if num == number:
            clean_Y.append(1)
        else:
            clean_Y.append(0)

    return np.asarray(clean_Y)

if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    hand_writting = datasets.load_digits()
    X = hand_writting.data
    Y = hand_writting.target
    
    m = X.shape[0]
    m_train = int(m * 0.8)
    m_test = m - m_train
    
    number = input("Enter number to predict: ")
    number = int(number)
    Y = clean(Y, number)
    
    X_train = X[:m_train]
    Y_train = Y[:m_train]
    X_test = X[m_train:]
    Y_test = Y[m_train:]
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)
    
    X_train /= 16
    X_test /= 16
    
    model(X_train.T, Y_train.T, X_test.T, Y_test.T, 2000, 0.005, True)
