import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')


class lg_dataset:
    def __init__(self):
        self.train_size = 2000
        self.train_param1 = tf.random.uniform([100], minval=2, maxval=10)
        self.train_param2 = tf.random.uniform([100], minval=50, maxval=110)
        self.train_data = tf.stack(
            [self.train_param1, self.train_param2], axis=1)
        self.test_size = 200
        self.train_param11 = tf.random.uniform([100], minval=12, maxval=20)
        self.train_param22 = tf.random.uniform([100], minval=60, maxval=120)
        self.train_data = tf.concat([self.train_data, tf.stack(
            [self.train_param11, self.train_param22], axis=1)], 0)
        self.train_label = tf.random.uniform([100], minval=1, maxval=5)
        self.train_label = tf.concat(
            [self.train_label, tf.random.uniform([100], minval=-1, maxval=-5)], 0)

        self.train_label = np.array([int(a > 0) for a in self.train_label])
        self.train_label[self.train_label == 0] = -1

    def load_data(self):
        return (self.train_data, self.train_label)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('param1')
        ax.set_ylabel('param2')
        ax.set_zlabel('targets')
        ax.scatter(self.train_data[:, 0], self.train_data[:, 1],
                   self.train_label, c='g', label="ground truth", cmap='coolwarm')
        ax.legend()
        plt.show()


def load_data(dataset_name, plot=False):
    if dataset_name == 'svm':
        __dataset = lg_dataset()
        if plot:
            __dataset.plot()
        return __dataset.load_data()
    else:
        raise ValueError('Dataset not found')


class svm_model:
    def __init__(self, train_data, train_label, epochs=250, visualization=True) -> None:
        self.train_data = (train_data - np.mean(train_data,
                                                axis=0)) / np.std(train_data, axis=0)
        self.train_label = train_label

        self.colors = {1: 'r', -1: 'b'}

        self.epochs = epochs
        self.epsilon = 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.loss = tf.keras.losses.Hinge()

        self.w = tf.Variable(tf.random.normal([2, 1]))
        self.b = tf.Variable(tf.random.normal([1]))

    # @tf.function
    # def model(self, x):
    #     return tf.linalg.matmul(x, self.w) - self.b

    def svc(self):
        inputs = tf.keras.layers.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train(self):

        self.svc_clf = self.svc()
        self.svc_clf.fit(self.train_data, self.train_label, epochs=self.epochs)
        self.w = self.svc_clf.get_weights()[0]
        self.b = self.svc_clf.get_weights()[1]

    def predict(self, x):
        return self.model(x)

    def visualize_svm(self):
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(
            self.train_data[:, 0], self.train_data[:, 1], marker="o", c=self.train_label)

        x0_1 = np.min(self.train_data[:, 0])
        x0_2 = np.max(self.train_data[:, 0])

        x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
        x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, self.w, self.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.w, self.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, self.w, self.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.w, self.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(self.train_data[:, 1])
        x1_max = np.amax(self.train_data[:, 1])
        ax.set_ylim([x1_min - 0.75, x1_max + 0.75])

        ax.set_xlabel('param1')
        ax.set_ylabel('param2')
        ax.legend()
        ax.set_title('SVM')
        plt.show()


if __name__ == '__main__':
    dataset_name = 'svm'
    dataset = load_data(dataset_name, plot=True)
    (train_data, train_label) = dataset
    print(train_data.shape, train_label.shape)

    clf = svm_model(train_data, train_label)
    clf.train()
    clf.visualize_svm()
