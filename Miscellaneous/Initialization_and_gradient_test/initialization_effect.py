import numpy as np
import sklearn
import tensorflow as tf
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt


class dataset:
    def __init__(self, n_samples) -> None:
        self.samples = n_samples
        self.data, self.label = datasets.make_circles(
            n_samples=self.samples, noise=0.1, factor=0.5)
        # self.label[self.label == 0] = -1

    def get_data(self, if_plot=False):
        if if_plot:
            self.plot_data()
        return self.data, self.label

    def plot_data(self):
        plt.scatter(self.data[:, 0], self.data[:, 1],
                    c=self.label, cmap=plt.cm.Spectral)
        plt.show()


def plot_history(history_array):
	datas = ['zeros','random','he_uniform']
	for ind,history in enumerate(history_array):
		plt.plot(history.history['accuracy'],label = datas[ind])

	plt.legend(loc='upper left')	
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.show()

def NN_model(input_shape=(2,), initialization='zeros'):
	if initialization == 'zeros':
		kernel_initializer = 'zeros'
		bias_initializer = 'zeros'
	elif initialization == 'random':
		kernel_initializer = 'random_normal'
		bias_initializer = 'random_normal'
	elif initialization == 'he':
		kernel_initializer = 'he_uniform'
		bias_initializer = 'he_uniform'

	inputs = tf.keras.layers.Input(input_shape)
	x = tf.keras.layers.Dense(
		10, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(inputs)
	x = tf.keras.layers.Dense(
		10, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
	outputs = tf.keras.layers.Dense(
		1, activation='sigmoid')(x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	return model


if __name__ == "__main__":
    datasets = dataset(300)
    data, labels = datasets.get_data(if_plot=True)

    model_zeros = NN_model(initialization = 'zeros')
    model_random = NN_model(initialization = 'random')
    model_he = NN_model(initialization = 'he')

    models = [model_zeros, model_random, model_he]
    histories = []
    for model in models:
    	model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    	history = model.fit(data,labels,epochs = 25*4)
    	histories.append(history)
    	print()
    plot_history(histories)

    string = ''' Note:
    	from this you can see he_uniform converges faster
    	than the other initializers like zeros or random.
    '''
    print(string)
