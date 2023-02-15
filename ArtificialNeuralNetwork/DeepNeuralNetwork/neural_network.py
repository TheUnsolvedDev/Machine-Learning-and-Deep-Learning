import tensorflow as tf
import numpy as np


class NN:
    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = tf.constant(train_data, dtype=tf.float32)
        self.train_label = tf.one_hot(train_label, depth=10)
        self.test_data = tf.constant(test_data, dtype=tf.float32)
        self.test_label = tf.one_hot(test_label, depth=10)
        self.shape = (train_data.shape[1], train_data.shape[2])

    # @tf.function
    def model(self):
        input = tf.keras.layers.Input(shape=self.shape)
        x = tf.keras.layers.Flatten()(input)
        x = tf.keras.layers.Dense(
            256, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dense(
            256, activation='relu', kernel_initializer='he_uniform')(x)
        final = tf.keras.layers.Dense(10, activation='softmax')(x)

        model = tf.keras.Model(inputs=input, outputs=final)
        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        self.classifier = self.model()
        self.classifier.fit(self.train_data, self.train_label,
                            batch_size=200, epochs=100)

    def test(self):
        model = self.classifier
        print('Accuracy ', model.evaluate(
            self.test_data, self.test_label))


if __name__ == '__main__':
    (train_data, train_label), (test_data,
                                test_label) = tf.keras.datasets.mnist.load_data()
    classifier = NN(train_data, train_label, test_data, test_label)
    classifier.train()
    classifier.test()
