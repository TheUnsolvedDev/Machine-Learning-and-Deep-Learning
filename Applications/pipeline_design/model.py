import tensorflow as tf
import numpy as np

def lenet5():
    input = tf.keras.Input(shape=(32, 32, 1))
    rescale = tf.keras.layers.Lambda(lambda x:x/255.0)(input)
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                                   activation='relu', padding='same')(rescale)
    maxpool2 = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid')(conv1)
    droupout1 = tf.keras.layers.Dropout(0.25)(maxpool2)


    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=5,
                                   activation='relu', padding='valid')(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid')(conv3)
    droupout2 = tf.keras.layers.Dropout(0.25)(maxpool3)

    flat = tf.keras.layers.Flatten()(droupout2)
    fc1 = tf.keras.layers.Dense(units=240, activation='relu')(flat)
    fc2 = tf.keras.layers.Dense(units=128, activation='relu')(fc1)
    final = tf.keras.layers.Dense(units=4, activation='softmax')(fc2)

    model = tf.keras.Model(inputs=input, outputs=final)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
	m = lenet5()
	m.summary()