import tensorflow as tf
import numpy as np
import os
import pandas as pd
from silence_tensorflow import silence_tensorflow

from params import *

silence_tensorflow()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def normalize(image):
    image = tf.cast(image, dtype=tf.float16)
    image = (image / 127.5) - 1
    return image


def resize(image, size):
    h, w = size
    image = tf.image.resize(
        image, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    return img


def preprocess_data(file_path, orig_label, target_label):
    image = process_path(file_path)
    image = resize(image, (128, 128))
    image = normalize(image)

    return image, orig_label, target_label


def get_label(path, dict):
    path = path.split('/')[-1]
    return dict[path]


def create_dataset():
    ATTRIBUTES = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    data = pd.read_csv('/home/shuvrajeet/datasets/celeba/list_attr_celeba.csv')
    ATTRIBUTES = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    for attr in data.columns[1:]:
        if attr not in ATTRIBUTES:
            data = data.drop([attr], axis=1)
    data = data.replace(-1, 0)
    data = (np.array(data))

    img2attr = {}
    for i in range(len(data)):
        img2attr[data[i][0]] = data[i][1:]

    base_path = '/home/shuvrajeet/datasets/celeba/images/'
    file_paths = [base_path + file for file in os.listdir(base_path)]
    file_paths = file_paths[:len(file_paths)//2]
    image_ds = tf.data.Dataset.list_files(
        '/home/shuvrajeet/datasets/celeba/images/*', shuffle=False)
    image_label = np.array([get_label(path, img2attr)
                           for path in file_paths], dtype=np.uint8)
    ori_label_ds = tf.data.Dataset.from_tensor_slices(image_label)
    tar_label_ds = tf.data.Dataset.from_tensor_slices(
        image_label).shuffle(BUFFER_SIZE)

    train_dataset = tf.data.Dataset.zip((image_ds, ori_label_ds, tar_label_ds)).map(lambda x, y, z: preprocess_data(
        x, y, z), num_parallel_calls=tf.data.AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_dataset.shuffle(BUFFER_SIZE)


if __name__ == '__main__':
    data = create_dataset()
    for a, b, c in data:
        print(a.shape, b.shape, c.shape)
        break
