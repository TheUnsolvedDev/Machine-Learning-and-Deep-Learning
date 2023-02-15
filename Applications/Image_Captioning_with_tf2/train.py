import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from param import *
from model import model
from dataset import Dataset

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_talk.h5', save_weights_only=True, monitor='loss', save_best_only=True)
]



if __name__ == '__main__':
    data = Dataset()
    m = model()
    m.summary()
    try:
        m.load_weights('model_talk.h5')
        print("Loaded model weights")
    except FileNotFoundError:
        pass
    train,test = data.get_data()
    m.fit(train,epochs = EPOCHS,validation_data = test,callbacks = callbacks)
