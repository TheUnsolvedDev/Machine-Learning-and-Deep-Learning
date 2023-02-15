import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import collections
import random
import pickle

from param import *

caption_folder = '/home/shuvrajeet/datasets/annotations_trainval2017/annotations'
image_folders = ['/home/shuvrajeet/datasets/train2017',
                 '/home/shuvrajeet/datasets/val2017']
caption_files = [caption_folder + '/' +
                 i for i in os.listdir(caption_folder) if 'captions' in i]
print(caption_files)


def mark_captions(captions_listlist):
    mark_start = 'ssss '
    mark_end = ' eeee'
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                       for captions_list in captions_listlist]

    return captions_marked


def replace_se(string):
    return string.replace('ssss ', '').replace(' eeee', '')


class Dataset:
    def __init__(self, train_size=0.8):
        self.train_size = train_size
        seed = np.random.randint(0, 10000)
        total_image_paths = []
        total_image_captions = []
        total_captions = []

        for i in range(len(caption_files)):
            with open(caption_files[i], 'r') as f:
                annotations = json.load(f)

            image_path_to_caption = collections.defaultdict(list)
            for val in annotations['annotations']:
                caption = f"ssss {val['caption']} eeee"
                image_path = image_folders[i]+'/' + \
                    '%012d.jpg' % (val['image_id'])
                image_path_to_caption[image_path].append(caption)
            image_paths = list(image_path_to_caption.keys())

            for image_path in image_paths:
                captions_list = image_path_to_caption[image_path]
                total_image_captions.extend(captions_list)
                np.random.shuffle(captions_list)
                total_captions.extend(list(map(replace_se, captions_list)))
                total_image_paths.extend(
                    [image_path for i in range(len(captions_list))])

        np.random.seed(seed)
        np.random.shuffle(total_image_paths)
        np.random.seed(seed)
        np.random.shuffle(total_captions)
        np.random.seed(seed)
        np.random.shuffle(total_image_captions)

        self.tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB,
            standardize=standardize,
            output_sequence_length=MAX_LENGTH)
        try:
            with open('tokenizer.pkl', 'rb') as f:
                dicts = pickle.load(f)
                self.tokenizer.from_config(dicts['config'])
                self.tokenizer.set_weights(dicts['weights'])
        except FileNotFoundError:
            print('Making tokenizer...')
            self.tokenizer.adapt(
                list(set('  '.join(total_image_captions).split(' '))))
            with open('tokenizer.pkl', 'wb') as f:
                dicts = {
                    'config': self.tokenizer.get_config(),
                    'weights': self.tokenizer.get_weights(),
                }
                pickle.dump(dicts, f)
        size = int(len(total_image_paths)*self.train_size)
        self.train_data = total_image_paths[:size]
        self.train_labels = total_captions[:size]
        self.train_data_text = total_image_captions[:size]

        self.test_data = total_image_paths[size:]
        self.test_labels = total_captions[size:]
        self.test_data_text = total_image_captions[size:]

        self.train_data = tf.data.Dataset.from_tensor_slices(
            self.train_data).map(self.load_image)
        self.train_data_text = tf.data.Dataset.from_tensor_slices(self.train_data_text).map(
            lambda x: self.tokenizer(x))
        self.train_data = tf.data.Dataset.zip(
            (self.train_data, self.train_data_text))
        self.train_labels = tf.data.Dataset.from_tensor_slices(self.train_labels).map(
            lambda x: self.tokenizer(x))

        self.test_data = tf.data.Dataset.from_tensor_slices(
            self.test_data).map(self.load_image)
        self.test_data_text = tf.data.Dataset.from_tensor_slices(self.test_data_text).map(
            lambda x: self.tokenizer(x))
        self.test_data = tf.data.Dataset.zip(
            (self.test_data, self.test_data_text))
        self.test_labels = tf.data.Dataset.from_tensor_slices(self.test_labels).map(
            lambda x: self.tokenizer(x))

    def get_data(self):
        self.train_dataset = tf.data.Dataset.zip((self.train_data, self.train_labels)).shuffle(
            1000, reshuffle_each_iteration=True)
        self.test_dataset = tf.data.Dataset.zip((self.test_data, self.test_labels)).shuffle(
            1000, reshuffle_each_iteration=True)
        return self.train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), self.test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def load_image(self, path, size=IMAGE_SIZE):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.int8)
        img = tf.image.resize(img, [IMAGE_SIZE[0], IMAGE_SIZE[1]])
        return img


if __name__ == "__main__":
    data = Dataset()
    train_data, test_data = data.get_data()
    for data in train_data.take(1):
        print(data)
