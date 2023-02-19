import tensorflow as tf
import numpy as np
import os
import cv2
import tqdm
import json

from params import *


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def parse_example(example_proto):
    # Parse a single example from the dataset
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))
    image = tf.image.convert_image_dtype(image, tf.uint8)
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(example['image/object/class/label'])
    return image, (xmin, ymin, xmax, ymax, labels)


def load_dataset(dataset_dir, batch_size=32, shuffle_buffer_size=1000, prefetch_buffer_size=1000, repeat=-1):
    dataset_dir = [dataset_dir+file for file in os.listdir(dataset_dir)]
    print(dataset_dir[-1],100000)
    dataset = tf.data.TFRecordDataset(dataset_dir[0])
    dataset = dataset.map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(
        batch_size).prefetch(prefetch_buffer_size)
    if repeat > 0:
        dataset = dataset.repeat(repeat)
    return dataset

if __name__ == '__main__':
    dataset = load_dataset('records/')

    for data in dataset:
        print(data)
        break
