
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


if __name__ == '__main__':
    dataset = Dataset()
    train, validation = dataset.make_dataset()

    models = [JPU, Gated_SCNN, ResUNet, DeepLabV3Plus, Unet]

    for data, label in validation:
        img = data[0]  # tf.reduce_mean(data[0], axis=-1)
        mask = tf.argmax(label[0], axis=-1)
        break

    for m in models:
        model = m()
        model.compile(
            optimizer='adam', loss=total_loss, metrics=['accuracy', iou_coef, dice_coef, f1])
        model.load_weights(m.__name__+'.h5')
        
        pred = tf.argmax(model.predict(data)[0], axis=-1)
        print(m.__name__)
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)

        plt.subplot(1, 3, 2)
        plt.imshow(mask)

        plt.subplot(1, 3, 3)
        plt.imshow(pred)

        plt.show()
