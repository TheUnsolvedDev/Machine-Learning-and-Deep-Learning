import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tqdm

from params import *
from model import *
from dataset import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_SSD_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def train():
    model = ssd_model()
    optimizer = tf.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer,
                  loss=total_loss)


if __name__ == '__main__':
    pass
