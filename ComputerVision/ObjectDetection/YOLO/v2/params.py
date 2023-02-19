import numpy as np
import tensorflow as tf

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable', 'dog',    'horse',  'motorbike', 'person',
          'pottedplant', 'sheep',  'sofa',   'train',   'tvmonitor']

train_image_folder = "/home/shuvrajeet/datasets/VOC2012/JPEGImages/"
train_annot_folder = "/home/shuvrajeet/datasets/VOC2012/Annotations/"

IMG_SIZE = (416, 416)
IMG_CHANNELS = 3

ANCHORS = np.array([0.08285376, 0.13705531,
                    0.20850361, 0.39420716,
                    0.80552421, 0.77665105,
                    0.42194719, 0.62385487])

GRID_W = 13
GRID_H = 13
BATCH_SIZE = 16
LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT = 5.0
LAMBDA_COORD = 1.0
LAMBDA_CLASS = 1.0
BOX = int(len(ANCHORS)/2)


def get_model_memory_usage(model, batch_size):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([tf.keras.backend.count_params(p)
                             for p in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(p)
                                 for p in model.non_trainable_weights])

    number_size = 4.0
    if tf.keras.backend.floatx() == 'float16':
        number_size = 2.0
    if tf.keras.backend.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * \
        (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + \
        internal_model_mem_count
    return gbytes
