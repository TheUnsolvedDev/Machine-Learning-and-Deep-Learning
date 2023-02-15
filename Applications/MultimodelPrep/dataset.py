import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tqdm

from params import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def parse_example(example_proto):
    # Parse a single example from the dataset
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        # 'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        # 'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        # 'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        # 'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))
    image = tf.image.convert_image_dtype(image, tf.uint8)
    # xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    # ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    # xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    # ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(example['image/object/class/label'])
    # (xmin, ymin, xmax, ymax, labels)
    return image, tf.one_hot(labels[0], depth=NUM_CLASSES)


def load_dataset(dataset_dir, batch_size=32, shuffle_buffer_size=1000, prefetch_buffer_size=1000, repeat=-1):
    dataset_dir = [dataset_dir+file for file in os.listdir(dataset_dir)]
    print(dataset_dir)
    dataset = tf.data.TFRecordDataset(dataset_dir)
    dataset = dataset.map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(
        batch_size).prefetch(prefetch_buffer_size)
    if repeat > 0:
        dataset = dataset.repeat(repeat)
    return dataset


if __name__ == '__main__':
    dataset = load_dataset('records/val/')
    for data in dataset:
        print(data)
        break
