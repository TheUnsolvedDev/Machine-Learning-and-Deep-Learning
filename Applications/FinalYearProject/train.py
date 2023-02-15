import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from dataset import *
from model import *
from params import *

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

models = [MobileNet, DenseNet, InceptionNet, ResNext, XCeption, SE_ResNet]

if __name__ == '__main__':
    dataset = Dataset()
    train, val = dataset.generators()
    for ind, m in enumerate(models):
        print('***** Training model:', m.__name__, '*****')
        model = m()
        model.summary()
        tf.keras.utils.plot_model(model, to_file=m.__name__+'.png', show_shapes=True,
                                  show_layer_names=True, show_layer_activations=True)
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

        
        history = model.fit(train, epochs=EPOCHS,
                            validation_data=val, callbacks=callbacks + [
                                tf.keras.callbacks.ModelCheckpoint(filepath='model_'+m.__name__+'_.h5', save_weights_only=True, monitor='val_loss',
                                                                   save_best_only=True),
                                tf.keras.callbacks.CSVLogger(
                                    filename='Histories/CSV/model_'+m.__name__+'_.csv', separator=",", append=True)
                            ])
        # np.save('Histories/NPY/model_'+m.__name__+'_.npy', history.history)
