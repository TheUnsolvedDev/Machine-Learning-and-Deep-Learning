import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import cv2
import tqdm

from params import *

ROOT_DIR = './train/'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def read_dataset():
    classes = os.listdir(ROOT_DIR)
    files = [ROOT_DIR+clas+'/' +
             file for clas in classes for file in os.listdir(ROOT_DIR+clas+'/')]
    random.shuffle(files)
    return files


class Dataset:
    def __init__(self, test_size=0.2, if_set_inception=False):
        self.files = read_dataset()
        self.test_size = test_size
        self.num_files = len(self.files)
        self.test_files = self.files[:int(self.num_files*self.test_size)]
        self.train_files = self.files[len(self.test_files):]
        self.if_inception = if_set_inception

        self.train_files = tf.data.Dataset.list_files(
            self.train_files, shuffle=True).prefetch(tf.data.AUTOTUNE)
        self.test_files = tf.data.Dataset.list_files(
            self.test_files, shuffle=True).prefetch(tf.data.AUTOTUNE)

    def get_label(self, path):
        return tf.where(tf.strings.split(path, os.path.sep)[2] == 'CE', 0, 1)

    def read_image_and_label(self, path):
        label = self.get_label(path)
        img = tf.io.read_file(path)
        img = tf.io.decode_png(img)
        img = img[:, :, :3]
        img = tf.image.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
        return img, label

    def generators(self):
        train_gen = self.train_files.map(self.read_image_and_label).batch(
            BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_gen = self.test_files.map(self.read_image_and_label).batch(
            BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return train_gen, val_gen


if __name__ == '__main__':
    obj = Dataset()

    files = read_dataset()
    for i in tqdm.tqdm(files):
        try:
            img = tf.io.read_file(i)
            img = tf.io.decode_png(img)
        except Exception as e:
            print(i, e)
