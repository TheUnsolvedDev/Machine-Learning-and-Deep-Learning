import tensorflow as tf
import keras_cv
import numpy as np
import os

from params import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

random_flip = keras_cv.layers.RandomFlip(
    mode="horizontal", bounding_box_format="xywh")
rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=2,
    # we disable geometric augmentations for object detection tasks
    geometric=False,
)

metrics = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=range(20),
        bounding_box_format="xywh",
        name="Mean Average Precision",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall",
    ),
]


def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


def augment(inputs):
    # In future KerasCV releases, RandAugment will support
    # bounding box detection
    inputs['images'] = tf.image.resize(
        inputs["images"], [256, 256], method='nearest')
    inputs["images"] = rand_augment(inputs["images"])
    inputs = random_flip(inputs)
    return inputs


if __name__ == '__main__':
    train_ds, train_dataset_info = keras_cv.datasets.pascal_voc.load(
        bounding_box_format="xywh", split="train", batch_size=BATCH_SIZE
    )
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds, val_dataset_info = keras_cv.datasets.pascal_voc.load(
        bounding_box_format="xywh", split="validation", batch_size=BATCH_SIZE
    )

    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
    
