import dataset
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Linear_Regression:

    def __init__(self, train_data, train_label, test_data, test_label, weights, biases):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.inputDimen = train_data.shape[1]

        self.input = tf.Variable(self.train_data, dtype=tf.float32)
        self.targets = tf.Variable(self.train_label, dtype=tf.float32)

        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.biases = tf.Variable(biases, dtype=tf.float32)

        self.perceptron = self.model(self.train_data)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.00014)

    @tf.function
    def model(self, input_data):
        val = tf.linalg.matmul(input_data, self.weights) + self.biases
        return val

    def train(self, epochs=100):
        self.plot()
        self.loss = lambda: tf.reduce_mean((self.model(data)-label)**2)

        count = 0
        while epochs:
            data = self.train_data
            label = self.train_label
            self.optimizer.minimize(self.loss, var_list=[
                                    self.weights, self.biases])
            if epochs % 50 == 0:
                count += 1
                print('\rloading please wait'+'.'*(count % 10), end='')
                sys.stdout.flush()
            epochs -= 1
        print()

    def test(self):
        # predict = self.model(self.test_data)
        # for i in range(200):
        #     print(predict[i], self.test_label[i])

        self.plot()

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('param1')
        ax.set_ylabel('param2')
        ax.set_zlabel('targets')
        ax.scatter(np.array(self.test_data).T[0], np.array(self.test_data).T[1],
                   self.test_label, c='g', label="ground truth")
        ax.scatter(np.array(self.test_data).T[0], np.array(self.test_data).T[1],
                   self.model(self.test_data), c='r', label="predicted")
        ax.legend()
        plt.show()


def initialize_parameter():
    weights = tf.ones(shape=[train_data.shape[1], 1], dtype=tf.float32)
    biases = tf.ones(shape=[train_data.shape[0], 1], dtype=tf.float32)
    return (weights, biases)


if __name__ == '__main__':
    epochs = 5000

    (train_data, train_label), (test_data, test_label) = dataset.load_data(
        'linear_regression', plot=False)

    train_data = tf.constant(train_data)  # float64
    train_label = tf.constant(train_label)
    test_data = tf.constant(test_data)
    test_label = tf.constant(test_label)

    (weights, biases) = initialize_parameter()  # weights=float32
    linear_regression = Linear_Regression(
        train_data, train_label, test_data, test_label, weights, biases)
    linear_regression.train(epochs)
    linear_regression.test()
