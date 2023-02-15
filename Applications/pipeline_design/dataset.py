import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
BATCH_SIZE = 16

BASE_DIR = 'Dataset'


class Dataset:
    def __init__(self):
        self.train_files = tf.data.Dataset.list_files(
            'Dataset/train/*/*', shuffle=True)
        self.test_files = tf.data.Dataset.list_files('Dataset/test/*',shuffle = False)

    def get_label(self, file_path):
        return tf.io.decode_raw(tf.strings.split(tf.strings.split(file_path, os.path.sep)[2])[1], tf.uint8)-49

    def func_read_image(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = tf.io.decode_png(img)
        img = img[:, :, :3]
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize(img, [32, 32])

        return img, label

    def data_return(self):
        return self.train_files.map(self.func_read_image)

    def generator(self):
        for data in self.data_return():
            yield data

    def create_tf_gen(self):
        return tf.data.Dataset.from_generator(
            generator=self.generator,
            output_types=(tf.float32, tf.uint8),
            output_shapes=((32, 32, 1), (1,))).prefetch(tf.data.AUTOTUNE)

    def test_func_read_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_png(img)
        img = img[:, :, :3]
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize(img, [32, 32])

        return img

    def test_data_return(self):
        return self.test_files.map(self.test_func_read_image)

    def test_generator(self):
        for data in self.test_data_return():
            yield data

    def test_create_tf_gen(self):
        return tf.data.Dataset.from_generator(
            generator=self.test_generator,
            output_types=(tf.float32),
            output_shapes=((32, 32, 1))).prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    obj = Dataset()

    for label in obj.test_create_tf_gen():
        plt.imshow(np.array(label))
        plt.show()
