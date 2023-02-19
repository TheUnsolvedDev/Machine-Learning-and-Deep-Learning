
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def standardize(x):
    x = np.array(x)
    for i in range(np.shape(x)[1]):
        x[:, i] = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_parameters(dimensions):
    weights = np.zeros((dimensions, 1))
    bias = 0
    return (weights, bias)


def propagate(params, x, y):
    (w, b) = params

    # forward propagate

    m = x.shape[1]
    out = sigmoid(np.dot(w.T, x) + b)
    cost = np.sum(-np.log(out) * y + (1 - y) * np.log(1 - out)) / m

    # back propagate

    dw = np.dot(x, (out - y).T) / m
    db = np.sum(out - y) / m
    grads = {'dw': dw, 'db': db}

    return (grads, cost)


def predict(params, x):
    (w, b) = params
    return sigmoid(np.dot(w.T, x) + b)


def train(x,y,num_iterations=100000,learning_rate=0.0001):
    costs = []
    params = initialize_parameters(dimensions=8)
    for _ in range(num_iterations):
        (w, b) = params
        (grads, cost) = propagate(params, x, y)
        dw = grads['dw']
        db = grads['db']

        w -= learning_rate * dw
        b -= learning_rate * db
        params = (w, b)

        if _ % 100 == 0:
            costs.append(cost)
            output = np.array(predict(params,x) > 0.5, dtype=np.int32)
            print('Accuracy @', _, '=', (np.sum(output == y)/output.shape[1])*100)

    return (params, costs)


if __name__ == '__main__':
    n_samples = 300
    n_features = 8
    size = 0.2
    (X, y) = make_classification(n_samples=n_samples,
                                 n_features=n_features)
    X = standardize(X).reshape(n_features, n_samples)
    y = y.reshape(1, n_samples)

    (x_train, x_test, y_train, y_text) = (X[:int(n_samples * (1
            - size))], X[int(n_samples * (1 - size)):],
            y[:int(n_samples * (1 - size))], y[int(n_samples * (1
            - size)):])

    train(x_train, y_train)
