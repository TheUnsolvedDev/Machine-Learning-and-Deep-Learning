import numpy as np
import matplotlib.pyplot as plt
from pandas import test
import tensorflow as tf
import tensorflow_probability as tfp
import os

from param import *

image_path = './UTKFace/'


class Dataset:
    def __init__(self, train_size=0.9):
        image_paths = [image_path + i for i in os.listdir(image_path)]
        size = len(image_paths)
        np.random.shuffle(image_paths)

        train_size = int(size * train_size)
        train = image_paths[:train_size]
        test = image_paths[train_size:]

        self.train = tf.data.Dataset.from_tensor_slices(
            train).map(self.read_image_and_label)
        self.test = tf.data.Dataset.from_tensor_slices(
            test).map(self.read_image_and_label)

    def read_image_and_label(self, path):
        label = tf.strings.to_number(tf.strings.split(tf.strings.split(
            path, os.path.sep)[2], '_')[0])
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=IMAGE_SIZE[2])
        img = tf.image.resize(img, [IMAGE_SIZE[0], IMAGE_SIZE[1]])
        img = tf.cast(img,tf.uint8)
        return img, label

    def get_data(self):
        return self.train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), self.test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    obj = Dataset()
    train, test = obj.get_data()

    for elem in train.take(1):
        print(elem[0])
        print(elem[1])
