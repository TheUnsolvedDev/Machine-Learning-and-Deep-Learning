import numpy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from param import *


def model_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=IMAGE_SIZE),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='relu'),
    ])
    return model


if __name__ == '__main__':
    model = model_cnn()
    a = np.expand_dims(np.ones(IMAGE_SIZE)*255, axis=0)
    model.summary()
    print(model(a))
