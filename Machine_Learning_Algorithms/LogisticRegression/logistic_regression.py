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

        self.optimizer = tf.keras.optimizers.Adam()

    def true_false_positive(self,threshold_vector, y_test):
        true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
        true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
        false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
        false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

        tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
        fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

        return tpr, fpr

    def roc_from_scratch(self,probabilities, y_test, partitions=100):
        roc = np.array([])
        for i in range(partitions + 1):
            
            threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
            tpr, fpr = self.true_false_positive(threshold_vector, y_test)
            roc = np.append(roc, [fpr, tpr])
            
        return roc.reshape(-1, 2),threshold_vector

    @tf.function
    def model(self, input_data):
        val = tf.linalg.matmul(input_data, self.weights) + self.biases
        val = tf.sigmoid(val)
        return val

    def train(self, epochs=100):
        self.loss_data = []
        self.loss = lambda:tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits = self.model(data), labels = label))
        count = 0
        while epochs:
            data = self.train_data
            label = self.train_label
            self.loss_data.append(int(self.loss()))

            self.optimizer.minimize(self.loss, var_list=[self.weights, self.biases])
            if epochs % 50 == 0:
                count += 1
                print('\rloading please wait'+'.'*(count % 10), end='')
                sys.stdout.flush()
            epochs -= 1
        print()
        self.gmeans,self.threshold= self.roc_plot(self.model(data),label)

    def test(self):
        count = 0
        predict = self.model(self.test_data)
        for i in range(len(predict)):
            if predict[i] <= self.threshold:
                temp = 1.0
            else:
                temp = 0
            if temp == self.test_label[i]:
                count += 1

        self.plot()
        print('Test accuracy is',count/len(predict)*100)

    def plot(self):
        plt.plot(range(len(self.loss_data)),self.loss_data)
        plt.show()

    def roc_plot(self,predictions,labels):
        ROC,thresholds = self.roc_from_scratch(predictions,labels,200)
        plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=100)
        plt.title('ROC Curve',fontsize=20)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        gmeans = np.sqrt(ROC[:,1]*(1-ROC[:,0]))
        ind = np.argmax(gmeans)

        return gmeans[ind],thresholds[ind]


def initialize_parameter():
    weights = tf.ones(shape=[train_data.shape[1], 1], dtype=tf.float32)
    biases = tf.ones(1, dtype=tf.float32)
    return (weights, biases)


if __name__ == '__main__':
    epochs = 5000

    (train_data, train_label), (test_data, test_label) = dataset.load_data(
        'logistic_regression', plot=False)

    train_data = tf.constant(train_data)  # float64
    train_label = tf.constant(train_label)
    test_data = tf.constant(test_data)
    test_label = tf.constant(test_label)

    (weights, biases) = initialize_parameter()  # weights=float32
    linear_regression = Linear_Regression(
        train_data, train_label, test_data, test_label, weights, biases)
    linear_regression.train(epochs)
    linear_regression.test()
