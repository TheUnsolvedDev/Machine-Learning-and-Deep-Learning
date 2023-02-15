from dataset import Dataset, BATCH_SIZE
from model import lenet5
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
    obj = Dataset()
    model = lenet5()

    model.fit(obj.create_tf_gen().batch(BATCH_SIZE),
              epochs=25, callbacks=callbacks)
    values = tf.cast(tf.argmax(model.predict(obj.test_create_tf_gen().batch(BATCH_SIZE)),axis = 1),dtype = tf.int32)
    print(values)

    np.savetxt('answers.txt', values.numpy())
