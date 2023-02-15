import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.ndimage import affine_transform
import math
import sys

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier:
    def __init__(self, train_data, train_labels, test_data, test_labels, epochs=500000, features=500):
        train_data = np.array([self.affine_transform(i) for i in train_data])
        self.train_data = tf.cast(tf.reshape(
            (train_data/255), shape=[-1, 28*28]), dtype=tf.float32)
        test_data = np.array([self.affine_transform(i) for i in test_data])
        self.test_data = tf.cast(tf.reshape(
            (test_data/255), shape=[-1, 28*28]), dtype=tf.float32)
        self.train_labels = tf.one_hot(
            train_labels, depth=10, dtype=tf.float32)
        self.test_labels = tf.one_hot(test_labels, depth=10, dtype=tf.float32)

        self.weights = tf.Variable(tf.random.normal(
            shape=[784, 10], mean=0, stddev=1, seed=12, dtype=tf.float32))
        self.bias = tf.Variable(tf.random.normal(
            shape=[10], mean=0, stddev=1, seed=12, dtype=tf.float32))
        self.activation = tfp.bijectors.NormalCDF()

        self.best_weights = tf.Variable(self.weights, dtype=tf.float32)
        self.best_bias = tf.Variable(self.bias, dtype=tf.float32)

        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.SGD(0.01)
        self.losses = tf.keras.losses.CategoricalCrossentropy()
        self.perceptron = self.pca(self.train_data, components=features)

    def pca(self, data, components=500):
        covar = tf.linalg.matmul(tf.transpose(data), data)
        e, v = np.linalg.eig(covar)
        data = tf.linalg.matmul(data, tf.transpose(v[:components]))
        return data

    @tf.function
    def update_parameters(self, data, label):
        def loss(): return self.losses(label, self.model(data))
        self.optimizer.minimize(
            loss, [self.weights, self.bias])

    @tf.function
    def model(self, data):
        linear = tf.linalg.matmul(data, self.weights) + self.bias
        return self.activation(linear)
        # mean = tf.math.reduce_mean(linear)
        # std = tf.math.reduce_std(linear)
        # x = (linear - mean) / (std * math.sqrt(2))
        # return 0.5*(1 + tf.math.erf(x))

    def affine_transform(self, data):
        a = [a for a in range(28)]
        b = [a for a in range(28)]

        x, y = np.mgrid[:28, :28]

        # centroid coordinates
        x_mean = np.sum(x*data)/(np.sum(data))
        y_mean = np.sum(y*data)/(np.sum(data))

        # covariance matrix
        u_20 = np.sum(((x-x_mean)**2)*data)
        u_02 = np.sum(((y-y_mean)**2)*data)
        u_11 = np.sum(((x-x_mean)*(y-y_mean))*data)
        u_00 = np.sum(data)

        corr = u_11/u_20
        m = [[u_20/u_00, u_11/u_00], [u_11/u_00, u_02/u_00]]

        # angle of rotation
        angle = -1*0.5*np.arctan((2*m[0][1])/(m[0][0]-m[1][1]))

        # rotation matrix
        rot_mat = [[1, 0], [corr, 1]]

        # offset
        offset = [x_mean, y_mean]-np.dot(rot_mat, [14, 14])
        final_img = affine_transform(data, rot_mat, offset=offset)
        return final_img

    def train(self, plot=True):
        decline = 0
        prev_acc = 0

        while self.epochs != 0:
            # if decline > 6:
            #     break

            self.update_parameters(self.train_data, self.train_labels)
            if self.epochs % 100 == 0:
                (train_loss, train_accuracy), (test_loss,
                                               test_accuracy) = self.metrics()
                acc = test_accuracy

                if acc > prev_acc:
                    self.best_weights.assign(self.weights)
                    self.best_bias.assign(self.bias)
                else:
                    decline += 1

                print('\r Training Accuracy: {:.2f}% Testing Accuracy: {:.2f}% Training Loss: {:.2f} Testing Loss: {:.2f} Epoch: {}'.format(
                    train_accuracy*100, test_accuracy*100, train_loss, test_loss, self.epochs), end=' ')
                sys.stdout.flush()
                prev_acc = acc

            self.epochs -= 1

        print()

    @tf.function
    def metrics(self):
        train_predictions = self.model(self.train_data)
        test_predictions = self.model(self.test_data)

        train_loss = self.losses(self.train_labels, train_predictions)
        test_loss = self.losses(self.test_labels, test_predictions)

        train_accuracy = tf.cast(tf.equal(tf.argmax(train_predictions, axis=1), tf.argmax(
            self.train_labels, axis=1)), dtype=tf.float32)
        test_accuracy = tf.cast(tf.equal(tf.argmax(test_predictions, axis=1), tf.argmax(
            self.test_labels, axis=1)), dtype=tf.float32)
        return (train_loss, tf.reduce_mean(train_accuracy)), (test_loss, tf.reduce_mean(test_accuracy))


if __name__ == "__main__":
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.mnist.load_data()

    classify = Classifier(train_data, train_labels, test_data, test_labels)
    classify.train()
    classify.metrics()
