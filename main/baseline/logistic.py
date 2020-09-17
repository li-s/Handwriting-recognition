import numpy as np
import csv

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

def clean(Y, alphabet):
    clean_Y = []

    for i, al in enumerate(Y):
        if al == alphabet.lower():
            clean_Y.append(1)
        else:
            clean_Y.append(0)
    
    clean_Y = np.asarray(clean_Y)
    return clean_Y.reshape(clean_Y.shape[0], 1)

def get_data():
    with open('../../data/train.csv') as train_csv:
        csv_reader = csv.reader(train_csv, delimiter = ',')
        line_count = 0
        X_train = []
        Y_train = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                X_train.append(row[4:])
                Y_train.append(row[1])
        
        X_train = np.asarray(X_train, dtype = int)
        Y_train = np.asarray(Y_train).reshape(X_train.shape[0], 1)
        X_train.reshape(X_train.shape[1], X_train.shape[0])
        Y_train.reshape(Y_train.shape[1], Y_train.shape[0])

    with open('../../data/test.csv') as test_csv:
        csv_reader = csv.reader(test_csv, delimiter = ',')
        line_count = 0
        X_test = []
        Y_test = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                X_test.append(row[4:])
                Y_test.append(row[1])

        X_test = np.asarray(X_test, dtype = int)
        Y_test = np.asarray(Y_test).reshape(X_test.shape[0], 1)
        X_test.reshape(X_test.shape[1], X_test.shape[0])
        Y_test.reshape(Y_test.shape[1], Y_test.shape[0])
    
    return X_train, Y_train, X_test, Y_test

def prune(X, Y, alphabet):
    m = 0

    for al in Y:
        if al[0] == alphabet:
            m += 1
    
    m_true = m
    m_false = m
    Y_pruned = []
    X_pruned = []
    for i in range(len(Y)):
        if Y[i][0] == alphabet:
            if m_true > 0:
                X_pruned.append(X[i])
                Y_pruned.append(Y[i][0])
                m_true -= 1
        else:
            if m_false > 0:
                X_pruned.append(X[i])
                Y_pruned.append(Y[i][0])
                m_false -= 1
    
    return np.asarray(X_pruned), np.asarray(Y_pruned).reshape(len(Y_pruned), 1)

def train_val_split(X, Y, alphabet, percentage = 0.8):
    m = X.shape[0]
    m_train = int(m * percentage)

    Y_clean = clean(Y, alphabet)
    X_train = X[:m_train]
    Y_train = Y_clean[:m_train]
    X_test = X[m_train:]
    Y_test = Y_clean[m_train:]
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    X, Y, _, _ = get_data()   

    alphabet = input("Enter alphabet to predict: ")
    
    X_pruned, Y_pruned = prune(X, Y, alphabet)
    X_train, Y_train, X_test, Y_test = train_val_split(X_pruned, Y_pruned, alphabet, 0.8)

    d = model(X_train.T, Y_train.T, X_test.T, Y_test.T, 1000, 0.005, True)

    w = d["w"]
    b = d["b"]
    costs = d["costs"]
    
    count = 0
    for i in range(len(Y_test)):
        if Y_test[i] == 1:
            count += 1
    
    s = 0
    for i in range(len(X_test)):
        pred = predict(w, b, X_test[i].reshape(X_test[i].shape[0], 1))
        if pred != Y_test[i]:
            s += 1
    
    print(f"Total number of values = {count}")
    print(f"Number of values predicted wrong = {s}, ie {(count - s) / count * 100}%")
