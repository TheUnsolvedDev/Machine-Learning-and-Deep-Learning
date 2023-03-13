import numpy as np


class SGD:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model

    def step(self):
        for layer in self.model.layers:
            layer.W = layer.W - self.lr * layer.dW
            layer.b = layer.b - self.lr * layer.db


class MomentumSGD:
    def __init__(self, model, lr, momentum=0.3):
        self.lr = lr
        self.model = model
        self.momentum = momentum

        self.change_w = [np.zeros_like(layer.W) for layer in self.model.layers]
        self.change_b = [np.zeros_like(layer.b) for layer in self.model.layers]

    def step(self):
        for ind, layer in enumerate(self.model.layers):
            self.change_w[ind] = self.lr * layer.dW + \
                self.momentum * self.change_w[ind]
            layer.W = layer.W - self.change_w[ind]
            self.change_b[ind] = self.lr * layer.db + \
                self.momentum * self.change_b[ind]
            layer.b = layer.b - self.change_b[ind]


class AdaGrad:
    def __init__(self, model, lr, epsilon=1e-8):
        self.lr = lr
        self.model = model
        self.epsilon = epsilon

        self.change_w = [np.zeros_like(layer.W) for layer in self.model.layers]
        self.change_b = [np.zeros_like(layer.b) for layer in self.model.layers]

    def step(self):
        for ind, layer in enumerate(self.model.layers):
            self.change_w[ind] += layer.dW ** 2
            self.change_b[ind] += layer.db ** 2
            layer.W = layer.W - (self.lr * layer.dW) / \
                np.sqrt(self.change_w[ind] + self.epsilon)
            layer.b = layer.b - (self.lr * layer.db) / \
                np.sqrt(self.change_b[ind] + self.epsilon)


class RMSprop:
    def __init__(self, model, lr, epsilon=1e-8, beta=0.9):
        self.lr = lr
        self.model = model
        self.epsilon = epsilon
        self.beta = beta

        self.change_w = [np.zeros_like(layer.W) for layer in self.model.layers]
        self.change_b = [np.zeros_like(layer.b) for layer in self.model.layers]

    def step(self):
        for ind, layer in enumerate(self.model.layers):
            self.change_w[ind] = self.beta*self.change_w[ind] + \
                (1 - self.beta)*layer.dW ** 2
            self.change_b[ind] = self.beta*self.change_b[ind] + \
                (1 - self.beta)*layer.db ** 2
            layer.W = layer.W - (self.lr * layer.dW) / \
                np.sqrt(self.change_w[ind] + self.epsilon)
            layer.b = layer.b - (self.lr * layer.db) / \
                np.sqrt(self.change_b[ind] + self.epsilon)


class Adam:
    def __init__(self, model, lr, epsilon=1e-8, beta1=0.9, beta2=0.99):
        self.lr = lr
        self.model = model
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

        self.change_w = [np.zeros_like(layer.W) for layer in self.model.layers]
        self.change_b = [np.zeros_like(layer.b) for layer in self.model.layers]

        self.change_mw = [np.zeros_like(layer.W)
                          for layer in self.model.layers]
        self.change_mb = [np.zeros_like(layer.b)
                          for layer in self.model.layers]

    def step(self, epoch):
        for ind, layer in enumerate(self.model.layers):
            self.change_mw[ind] = self.beta1*self.change_mw[ind] + \
                (1 - self.beta1)*layer.dW
            self.change_mb[ind] = self.beta1*self.change_mb[ind] + \
                (1 - self.beta1)*layer.db
            self.change_w[ind] = self.beta2*self.change_w[ind] + \
                (1 - self.beta2)*layer.dW ** 2
            self.change_b[ind] = self.beta2*self.change_b[ind] + \
                (1 - self.beta2)*layer.db ** 2

            change_mw = self.change_mw[ind] / \
                (1. - np.power(self.beta1, epoch + 1.))
            change_mb = self.change_mb[ind] / \
                (1. - np.power(self.beta1, epoch + 1.))
            change_w = self.change_w[ind] / \
                (1. - np.power(self.beta2, epoch + 1.))
            change_b = self.change_b[ind] / \
                (1. - np.power(self.beta2, epoch + 1.))

            layer.W = layer.W - (self.lr * change_mw) / \
                np.sqrt(change_w + self.epsilon)
            layer.b = layer.b - (self.lr * change_mb) / \
                np.sqrt(change_b + self.epsilon)
