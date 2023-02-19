import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tqdm
from silence_tensorflow import silence_tensorflow

from params import *

silence_tensorflow()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    image = (image / 127.5) - 1
    return image


def resize(image, size):
    h, w = size
    image = tf.image.resize(image, [h, w])
    return image


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    return img


def preprocess_data(file_path):
    image = process_path(file_path)
    image = resize(image, (128, 128))
    image = normalize(image)

    return image


def get_label(path, dict):
    path = path.split('/')[-1]
    return dict[path]


class create_dataset(tf.keras.utils.Sequence):

    def __init__(self, batch_size=32, input_size=(128, 128, 3), shuffle=True):
        data = pd.read_csv(
            '/home/shuvrajeet/datasets/celeba/list_attr_celeba.csv')
        ATTRIBUTES = ['Black_Hair', 'Blond_Hair',
                      'Brown_Hair', 'Male', 'Young']
        for attr in data.columns[1:]:
            if attr not in ATTRIBUTES:
                data = data.drop([attr], axis=1)
        data = data.replace(-1, 0)
        data = (np.array(data))
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.img2attr = {}
        for i in range(len(data)):
            self.img2attr[data[i][0]] = data[i][1:]

        base_path = '/home/shuvrajeet/datasets/celeba/images/'
        self.file_paths = [base_path + file for file in os.listdir(base_path)]
        self.indices = [i for i in range(len(self.file_paths))]
        # file_paths = file_paths[:len(file_paths)//2]

        self.n = len(self.file_paths)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        choosen = self.indices[index *
                               self.batch_size:(index + 1) * self.batch_size]
        sub_files = [self.file_paths[i] for i in choosen]
        labels = np.array([get_label(path, self.img2attr)
                          for path in sub_files], dtype=np.uint8)

        images = []
        new_targets = labels
        for ind, file in enumerate(sub_files):
            image = preprocess_data(file)
            images.append(image)
        images = tf.cast(np.array(images, dtype=np.float32), tf.float32)
        labels = tf.cast(np.array(labels, dtype=np.uint8), tf.uint8)
        np.random.shuffle(new_targets)
        new_targets = tf.cast(new_targets, dtype=tf.uint8)
        return images, labels, new_targets

    def __len__(self):
        return self.n // self.batch_size


if __name__ == '__main__':
    data = create_dataset()
    for ind in tqdm.tqdm(range(len(data))):
        a, b, c = data.__getitem__(ind)
        print(ind,a.shape, b, c)
        break
