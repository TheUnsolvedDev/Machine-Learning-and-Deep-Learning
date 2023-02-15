import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from params import *
from dataset import *
from model import *


gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='weights_yolo_on_voc2012.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def custom_loss(y_true, y_pred):
    return (custom_loss_core(
        y_true,
        y_pred,
        true_boxes,
        GRID_W,
        GRID_H,
        BATCH_SIZE,
        ANCHORS,
        LAMBDA_COORD,
        LAMBDA_CLASS,
        LAMBDA_NO_OBJECT,
        LAMBDA_OBJECT))


if __name__ == '__main__':
    IMAGE_H, IMAGE_W = 416, 416
    GRID_H,  GRID_W = 13, 13
    TRUE_BOX_BUFFER = 50
    BOX = int(len(ANCHORS)/2)

    generator_config = {
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'GRID_H': GRID_H,
        'GRID_W': GRID_W,
        'LABELS': LABELS,
        'ANCHORS': ANCHORS,
        'BATCH_SIZE': BATCH_SIZE,
        'TRUE_BOX_BUFFER': TRUE_BOX_BUFFER,
    }
    train_image, seen_train_labels = parse_annotation(
        train_annot_folder, train_image_folder, labels=LABELS)
    train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                                 norm=normalize, shuffle=True)
    CLASS = len(LABELS)
    model, true_boxes = define_YOLOv2(IMAGE_H, IMAGE_W, GRID_H, GRID_W, TRUE_BOX_BUFFER, BOX, CLASS,
                                      trainable=False)
    

    model.summary()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=custom_loss)
    model.fit(train_batch_generator,
              steps_per_epoch=len(train_batch_generator),
              epochs=50,
              verbose=1,
              # validation_data  = valid_batch,
              # validation_steps = len(valid_batch),
              callbacks=callbacks)
