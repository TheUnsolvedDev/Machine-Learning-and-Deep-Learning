import tensorflow as tf
import numpy as np
import os
from silence_tensorflow import silence_tensorflow

from params import *

silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def normalize_img(x, dtype):
    x = tf.cast(x, dtype=dtype)
    return x / 127.5 - 1  # range: -1 ~ 1


def decode_img(img):
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = normalize_img(img, tf.float32)  # range: -1 ~ 1
    return img


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img


def augment_image(img, mask):
    rand = tf.random.uniform(())
    if rand <= 0.25:
        img = flip(img)
        mask = flip(mask)
    elif rand > 0.25 and rand <= 0.5:
        d = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = rotate(img, d)
        mask = rotate(mask, d)
    elif rand > 0.5 and rand <= 0.75:
        img = color(img)

    return img, mask


def flip(img):
    img = tf.image.flip_left_right(img)
    img = tf.image.flip_up_down(img)
    return img


def rotate(img, degree):
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(img, degree)


def color(img):
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    return img


def dataset():
    train_A = tf.data.Dataset.list_files(
        '/home/shuvrajeet/datasets/vangogh2photo/train/A/*')
    train_B = tf.data.Dataset.list_files(
        '/home/shuvrajeet/datasets/vangogh2photo/train/B/*')

    train_A = train_A.map(lambda x: process_path(
        x), num_parallel_calls=tf.data.AUTOTUNE).repeat(1)
    train_B = train_B.map(lambda x: process_path(
        x), num_parallel_calls=tf.data.AUTOTUNE).repeat(1)

    train_dataset = tf.data.Dataset.zip((train_A, train_B)).map(
        lambda x, y: augment_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.cache().batch(BATCH_SIZE).shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.zip((train_A, train_B))
    test_dataset = test_dataset.cache().batch(BATCH_SIZE).shuffle(BUFFER_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset


if __name__ == '__main__':
    data,dats2 = dataset()
    for dat in data:
        print(dat[0].numpy().shape)
        print(dat[1].numpy().shape)
        break