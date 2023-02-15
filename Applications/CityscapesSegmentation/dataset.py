import numpy as np
import tensorflow as tf
import tqdm
import os
import matplotlib.pyplot as plt

from params import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def read_files(type_of='train'):
    images = []
    for city in tqdm.tqdm(os.listdir(ROOT_DIR_IMAGES + type_of + '/')):
        for image in os.listdir(ROOT_DIR_IMAGES + type_of + '/'+city):
            if image.endswith('.png'):
                images.append(ROOT_DIR_IMAGES + type_of +
                              '/' + city + '/'+image)
    labels = []
    for city in tqdm.tqdm(os.listdir(ROOT_DIR_LABELS + type_of + '/')):
        for image in os.listdir(ROOT_DIR_LABELS + type_of + '/'+city):
            if image.endswith('gtFine_labelIds.png') and not image.startswith('.'):
                labels.append(ROOT_DIR_LABELS + type_of +
                              '/' + city + '/'+image)
    return sorted(images), sorted(labels)


class Dataset:
    def __init__(self):
        train_data, train_labels = read_files('train')
        validation_data, validation_labels = read_files('val')
        self.train = tf.data.Dataset.from_tensor_slices(
            (train_data, train_labels))
        self.validation = tf.data.Dataset.from_tensor_slices(
            (validation_data, validation_labels)).shuffle(BUFFER_SIZE)

    @tf.function
    def read_image(self, path1, path2):
        img = tf.io.read_file(path1)
        img = tf.io.decode_png(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)

        lbl = tf.io.read_file(path2)
        lbl = tf.io.decode_png(lbl, channels=1)
        lbl = tf.image.resize(lbl, IMAGE_SIZE)
        lbl = tf.cast(lbl, dtype=tf.uint8)
        lbl = tf.squeeze(lbl)
        lbl = tf.one_hot(lbl, depth=NUM_CLASSES, axis=-1)
        return tf.cast(img, dtype=tf.uint8), lbl

    def make_dataset(self):
        self.data_train = self.train.prefetch(tf.data.AUTOTUNE).map(
            self.read_image, num_parallel_calls=BATCH_SIZE).shuffle(BUFFER_SIZE)
        self.data_validation = self.validation.prefetch(tf.data.AUTOTUNE).map(
            self.read_image, num_parallel_calls=BATCH_SIZE).shuffle(BUFFER_SIZE)
        return self.data_train.batch(BATCH_SIZE), self.data_validation.batch(BATCH_SIZE)


if __name__ == '__main__':
    obj = Dataset()
    data, datum = obj.make_dataset()

    for i, j in data:
        print(i.shape, j.shape)
        input()
