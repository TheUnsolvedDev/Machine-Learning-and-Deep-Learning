import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 128
LAMBDA = 0.1
MAX_RHO = 6


def lenet5_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(
        3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    return model


def augment_dataset(model, x_train, y_train):
    data, labels = [], []
    num_batches = len(x_train)//BATCH_SIZE
    print(num_batches)
    for batch in tqdm.tqdm(range(num_batches)):
        data.append(x_train.numpy()[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
        labels.append(y_train.numpy()[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])

    data = np.array(data).reshape(-1, 28, 28, 1)
    labels = np.array(labels).reshape(-1, 28, 28, 1)
    return data, labels


loss_object = tf.keras.losses.CategoricalCrossentropy()


def augment(model, data, labels):
    temp_data = data
    with tf.GradientTape() as tape:
        tape.watch(temp_data)
        pred = model(temp_data)

    gradient = tape.gradient(pred, temp_data)
    data = temp_data + LAMBDA * tf.sign(gradient)
    labels = tf.argmax(pred, axis=-1)
    labels = tf.one_hot(labels, depth=10)
    return data, labels


class Augmented_dataset(tf.keras.utils.Sequence):
    def __init__(self, x_train, y_train, model) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.model = model
        self.batch_size = BATCH_SIZE
        self.lambda_ = LAMBDA

    def __len__(self):
        return 2*(len(self.x_train)//self.batch_size)

    def __getitem__(self, index):
        if index % 2 == 0:
            index //= 2
            data = self.x_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            labels = self.y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        else:
            index //= 2
            temp_data = self.x_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            labels = self.y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            data, labels = augment(self.model, temp_data, labels)
        return data, labels


def train_substitute(oracle, datagen, num_samples=200):
    substitute = lenet5_model()
    substitute.compile(loss='categorical_crossentropy',
                       optimizer='adam', metrics=['accuracy'])
    sample_nums = [random.randint(0, len(datagen)) for _ in range(num_samples)]
    data, labels = [], []
    for index in sample_nums:
        images, label = datagen.__getitem__(index)
        data.append(images[0])
        labels.append(label[0])

    data = tf.Variable(np.array(data))
    labels = tf.Variable(np.array(labels))

    for rho in range(MAX_RHO):
        data, labels = augment(oracle, data, labels)
        for epoch in range(10):
            substitute.train_on_batch(data, labels)
        substitute.evaluate(data,labels)
    substitute.save_weights('SubstituteModel.h5')
    return substitute


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train/255, axis=-1)
    x_test = tf.expand_dims(x_test/255, axis=-1)
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    model = lenet5_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=15, batch_size=BATCH_SIZE,
    #           validation_data=(x_test, y_test))
    # model.save_weights('BlackBoxLenet.h5')

    oracle_model = lenet5_model()
    oracle_model.load_weights('BlackBoxLenet.h5')

    datagen = Augmented_dataset(
        x_train=x_train, y_train=y_train, model=model)

    model2 = lenet5_model()
    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
    # model2.fit(datagen, epochs=12)
    # model2.save_weights('temp_weights.h5')

    substitute = train_substitute(oracle_model, datagen)
    substitute.evaluate(x_test,y_test)
