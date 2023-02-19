import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import outer
from scipy.sparse import data
import sklearn
import tensorflow as tf
import seaborn as sns
from sklearn import datasets
import sys


class dataset:
    def __init__(self, n_samples) -> None:
        self.samples = n_samples
        self.data, self.label = datasets.make_circles(
            n_samples=self.samples, noise=0.1, factor=0.5)
        self.label[self.label == 0] = -1

    def get_data(self, if_plot=False):
        if if_plot:
            self.plot_data()
        return self.data, self.label

    def plot_data(self):
        plt.scatter(self.data[:, 0], self.data[:, 1],
                    c=self.label, cmap=plt.cm.Spectral)
        plt.show()


class non_linear_svm:
    def __init__(self, data, label, epochs=1200) -> None:
        self.data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        self.label = np.array(label,dtype = np.float32)
        self.alpha = tf.Variable(
            tf.random.normal(shape=[1, self.data.shape[0]]))
        self.gamma = tf.constant(-50, dtype=tf.float32)
        self.optimzer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.epochs = epochs

    @tf.function
    def rbf_kernel(self, euclidean_distance):
        return tf.exp(tf.multiply(self.gamma, euclidean_distance))

    @tf.function
    def loss(self):
        dist = tf.reduce_sum(tf.square(self.data), axis=1)
        dist = tf.reshape(dist, [-1, 1])

        sq_dist = dist - 2 * \
            tf.matmul(self.data, tf.transpose(self.data)) + tf.transpose(dist)
        flt_sq_dist = tf.cast(sq_dist, tf.float32)

        term1 = tf.reduce_sum(self.alpha)

        alpha_sq = tf.matmul(tf.transpose(self.alpha), self.alpha)
        y_sqr = tf.matmul(self.label.reshape(-1, 1),
                          tf.transpose(self.label.reshape(-1, 1)))
        flt_y_sqr = tf.cast(y_sqr, tf.float32)
        term2 = tf.reduce_sum(tf.multiply(self.rbf_kernel(
            flt_sq_dist), tf.multiply(alpha_sq, flt_y_sqr)))

        return tf.negative(tf.subtract(term1, term2))

    def train(self):
        for epoch in range(self.epochs):
            self.optimzer.minimize(self.loss, var_list=[self.alpha])
            if epoch % 10 == 0:
                tf.print("\r", "Epoch:", epoch, "Loss:", self.loss(), end="")
                sys.stdout.flush()

        print("")

    def prediction(self, points):
        x_sq = tf.reshape(tf.reduce_sum(self.data**2,1),[-1,1])
        x_prime_sq = tf.reshape(tf.reduce_sum(points**2,1),[-1,1])
        sq_dist=tf.add(tf.subtract(x_sq, tf.multiply(2,tf.matmul(self.data,tf.transpose(points)))),tf.transpose(x_prime_sq))
        sq_dist=tf.cast(sq_dist,tf.float32)
        kernel = tf.exp(tf.multiply(self.gamma,tf.abs(sq_dist)))
        output = tf.matmul(tf.multiply(tf.transpose(self.label),self.alpha),kernel)
        return tf.sign(output - tf.reduce_mean(output))

    def plot_decision_boundary(self):
        x1_min, x1_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        x2_min, x2_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1

        x1,x2=np.meshgrid(np.arange(x1_min,x1_max,0.02), np.arange(x2_min,x2_max,0.02))
        points = np.c_[x1.ravel(), x2.ravel()]
        predictions = self.prediction(points)
        predictions = np.reshape(predictions, x1.shape)

        fig = plt.figure(figsize=(12, 8))
        plt.contourf(x1, x2, predictions, alpha=0.7,cmap=plt.cm.Paired)

        pos = self.data[self.label == 1]
        neg = self.data[self.label == -1]
        sns.scatterplot(x=pos[:, 0], y=pos[:, 1],
                        label="Positive", color="red")
        sns.scatterplot(x=neg[:, 0], y=neg[:, 1],
                        label="Negative", color="blue")
        plt.show()


if __name__ == '__main__':
    obj = dataset(n_samples=250)
    obj.get_data(if_plot=True)

    clf = non_linear_svm(obj.data, obj.label)
    clf.train()
    clf.plot_decision_boundary()
