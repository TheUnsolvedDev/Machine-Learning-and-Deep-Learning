import tensorflow as tf
import numpy as np


class Dataset(object):
    def __init__(self):
        (self.train_data, self.train_labels), (self.test_data,
                                               self.test_labels) = tf.keras.datasets.mnist.load_data()

        self.dataA_train = []
        self.labelA_train = []
        self.dataA_test = []
        self.labelA_test = []

        self.dataB_train = []
        self.labelB_train = []
        self.dataB_test = []
        self.labelB_test = []

        hashes = {i: i % 2 for i in range(10)}

        for ind, number in enumerate(self.train_labels):
            if hashes[number]:
                self.dataA_train.append(self.train_data[ind])
                self.labelA_train.append(self.train_labels[ind])
            else:
                self.dataB_train.append(self.train_data[ind])
                self.labelB_train.append(self.train_labels[ind])

        for ind, number in enumerate(self.test_labels):
            if hashes[number]:
                self.dataA_test.append(self.test_data[ind])
                self.labelA_test.append(self.test_labels[ind])
            else:
                self.dataB_test.append(self.test_data[ind])
                self.labelB_test.append(self.test_labels[ind])

    def task_A(self):
        return np.array(self.dataA_train, dtype=np.float32)/255, np.array(self.labelA_train), np.array(self.dataA_test, dtype=np.float32)/255, np.array(self.labelA_test)

    def task_B(self):
        return np.array(self.dataB_train, dtype=np.float32)/255, np.array(self.labelB_train), np.array(self.dataB_test, dtype=np.float32)/255, np.array(self.labelB_test)


if __name__ == '__main__':
    obj = Dataset()
    a, b, c, d = obj.task_A()
    print(a.shape, b.shape, c.shape, d.shape)
    a, b, c, d = obj.task_B()
    print(a.shape, b.shape, c.shape, d.shape)
