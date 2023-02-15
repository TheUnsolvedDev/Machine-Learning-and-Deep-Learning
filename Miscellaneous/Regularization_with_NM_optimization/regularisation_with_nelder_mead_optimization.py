import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
import matplotlib.pyplot as plt


class linear_classifier:
    def __init__(self, trainable_weights, trainable_biases):
        self.trainable_weights = tf.Variable(
            trainable_weights, dtype=tf.float32)
        self.trainable_biases = tf.Variable(trainable_biases, dtype=tf.float32)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function
    def predict(self, input_data):
        input_data = tf.reshape(input_data/255, shape=[-1, 28*28])
        x = tf.linalg.matmul(
            input_data, self.trainable_weights)+self.trainable_biases
        return tf.nn.softmax(x)

    def update_param(self, data, label):
        self.data = data
        self.label = label

        def losses(params):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.predict(self.data), labels=self.label)) + tf.reduce_sum(tf.abs(self.trainable_weights)) + tf.reduce_sum(tf.abs(self.trainable_biases))
        self.start_weights = np.random.randn(
            self.trainable_weights.shape[0]*self.trainable_weights.shape[1] + 1)
        self.start_bias = np.random.randn(self.trainable_biases.shape[0]+1)

        self.initial_vertex_weights = tf.expand_dims(
            tf.constant(self.start_weights, dtype=tf.float32), axis=-1)
        self.initial_vertex_bias = tf.expand_dims(
            tf.constant(self.start_bias, dtype=tf.float32), axis=-1)

        results = tfp.optimizer.nelder_mead_minimize(
            losses,
            initial_vertex=self.initial_vertex_weights,
            func_tolerance=1e-10,
            position_tolerance=1e-10)
        return results

    def train(self, epochs, train_data, train_label, test_data, test_label):
        for ep in range(epochs):
            self.update_param(train_data, train_label)
            if ep % 10 == 0:
                loss, acc = self.test(test_data, test_label)
                tf.print("\rloss={} acc = {}".format(loss, acc), end='')
        print()

    @tf.function
    def test(self, test_data, test_labels):
        predictions = self.predict(test_data)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=predictions, labels=test_labels))
        acc = tf.cast(tf.equal(tf.argmax(predictions, axis=1),
                               tf.argmax(test_labels, axis=1)), dtype=tf.float32)
        return loss, tf.reduce_mean(acc)


def initialize_parameters(data):
    weights = tf.ones(
        shape=[data.shape[1]*data.shape[2], 10], dtype=tf.float32)
    bias = tf.ones(shape=[10, ], dtype=tf.float32)
    return weights, bias


if __name__ == '__main__':
    (train_data, train_label), (test_data,
                                test_label) = tf.keras.datasets.mnist.load_data()

    train_data = tf.constant(train_data, dtype=tf.float32)
    test_data = tf.constant(test_data, dtype=tf.float32)

    train_label = tf.constant(tf.one_hot(
        train_label, depth=10), dtype=tf.float32)
    test_label = tf.constant(tf.one_hot(
        test_label, depth=10), dtype=tf.float32)

    weights, bias = initialize_parameters(train_data)
    classifier = linear_classifier(weights, bias)
    classifier.train(5000, train_data, train_label, test_data, test_label)
