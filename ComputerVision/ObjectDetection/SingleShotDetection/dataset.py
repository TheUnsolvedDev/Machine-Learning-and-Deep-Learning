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
    print(dataset_dir)
    dataset = tf.data.TFRecordDataset(dataset_dir)
    dataset = dataset.map(
        parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(
        batch_size).prefetch(prefetch_buffer_size)
    if repeat > 0:
        dataset = dataset.repeat(repeat)
    return dataset


def create_tf_record(json_file, tfrecord_file='coco_2014.tfrecord', data_path='val2014/COCO_val2014_'):
    with open(json_file, 'r') as f:
        data = json.load(f)

    num = 1
    class_names = [obj['name'] for obj in data['categories']]
    for id, image in tqdm.tqdm(enumerate(data['images'])):
        if id % RECORD_SIZE == 0:
            name = tfrecord_file.split('.')
            writer = tf.io.TFRecordWriter(
                'records/'+name[0]+str(num)+'.'+name[-1])
            num += 1

        image_id = image['id']
        image_path = data_path + \
            '0'*(12-len(str(image_id)))+str(image_id) + '.jpg'
        image_data = open(image_path, 'rb').read()
        try:
            annotation = [obj for obj in data['annotations']
                          if obj['image_id'] == image_id]
        except IndexError:
            continue

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        labels = []
        for obj in annotation:
            xmin.append(obj['bbox'][0] / IMG_SIZE[0])
            ymin.append(obj['bbox'][1] / IMG_SIZE[0])
            xmax.append((obj['bbox'][0] +
                        obj['bbox'][2]) / IMG_SIZE[0])
            ymax.append((obj['bbox'][1] +
                        obj['bbox'][3]) / IMG_SIZE[0])
            label = obj['category_id']
            labels.append(label)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            'image/object/bbox/xmin': _float_feature(xmin),
            'image/object/bbox/ymin': _float_feature(ymin),
            'image/object/bbox/xmax': _float_feature(xmax),
            'image/object/bbox/ymax': _float_feature(ymax),
            'image/object/class/label': _int64_feature(labels),
        }))

        writer.write(example.SerializeToString())
        if id % RECORD_SIZE == (RECORD_SIZE-1):
            writer.close()


if __name__ == '__main__':
    # create_tf_record(
    #     '/home/shuvrajeet/datasets/annotations_trainval2014/annotations/instances_train2014.json', 'coco_train.tfrecord', '/home/shuvrajeet/datasets/train2014/COCO_train2014_')
    # create_tf_record(
    #     '/home/shuvrajeet/datasets/annotations_trainval2014/annotations/instances_val2014.json', 'coco_val.tfrecord', '/home/shuvrajeet/datasets/val2014/COCO_val2014_')

    dataset = load_dataset('records/val/')
    print(dataset.take(1))
