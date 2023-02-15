from calendar import EPOCH
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from param import *
from dataset import Dataset
from model import model_cnn

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelA*.h5', save_weights_only=True, monitor='loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]

if __name__ == '__main__':
    model = model_cnn()
    data = Dataset()
    train,test = data.get_data()
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(loss = 'MAE',optimizer='adam')
    # model.fit(train, epochs = EPOCHS,validation_data = test,callbacks = callbacks)
    
    model.load_weights('modelA*.h5')
    
    for elem in train.take(1):
        out = model.predict(elem[0])
        labels = elem[1]
        
    for pred,real in zip(out,labels):
        print("Pred:",round(pred[0]),"Real:",round(real.numpy()))    
    