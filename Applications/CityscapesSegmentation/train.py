import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from dataset import *
from params import *
from model import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.keras.backend.sum(
        y_true, [1, 2, 3])+tf.keras.backend.sum(y_pred, [1, 2, 3])-intersection
    iou = tf.keras.backend.mean(
        (intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.keras.backend.sum(
        y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    dice = tf.keras.backend.mean(
        (2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = tf.keras.backend.sum(tf.keras.backend.round(
            tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        Positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))

        recall = TP / (Positives+tf.keras.backend.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = tf.keras.backend.sum(tf.keras.backend.round(
            tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+tf.keras.backend.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))


def total_loss(y_true, y_pred):
    loss1 = DiceLoss()
    loss2 = tf.keras.losses.CategoricalCrossentropy()
    return loss1(y_true, y_pred) + loss2(y_true, y_pred)


def plot_history(history, model_name):
    print(history)
    keys = list(history.keys())
    keys = [key for key in keys if 'val' not in key]

    for key in keys:
        if key == 'lr':
            continue
        plt.plot(history[key])
        plt.plot(history['val_'+key])
        plt.title(model_name + ' '+key)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    dataset = Dataset()
    train, validation = dataset.make_dataset()

    models = [JPU, Gated_SCNN]
    for m in models:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=m.__name__+'.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs', histogram_freq=1, write_graph=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
            tf.keras.callbacks.CSVLogger(
                filename='model_'+m.__name__+'_.csv', separator=",", append=True)
        ]
        model = m()
        model.compile(
            optimizer='adam', loss=total_loss, metrics=['accuracy', iou_coef, dice_coef, f1])
        tf.keras.utils.plot_model(
            model, to_file=m.__name__+'.png', show_shapes=True)
        history = model.fit(train, epochs=50, validation_data=validation,
                            callbacks=callbacks)
        
        # plot_history(history.history, m.__name__)
