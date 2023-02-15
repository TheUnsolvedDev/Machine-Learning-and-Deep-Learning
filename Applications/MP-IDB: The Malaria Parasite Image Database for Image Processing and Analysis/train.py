from params import *
from dataset import *
from model import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

if __name__ == '__main__':
    models = [MobileNet, ResNext, SE_ResNet, XCeption]
    train_dataset = Dataset()
    validation_dataset = Dataset(shuffle=False)

    for ind, m in enumerate(models):
        print('***** Training model', ind, ':', m.__name__, '*****')
        model = m()
        model.summary()
        tf.keras.utils.plot_model(model, to_file='diagrams/architecture/'+m.__name__+'.png', show_shapes=True,
                                  show_layer_names=True, show_layer_activations=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy', 'categorical_accuracy'])

        history = model.fit(train_dataset, epochs=EPOCHS,
                            validation_data=validation_dataset, callbacks=callbacks + [
                                tf.keras.callbacks.ModelCheckpoint(filepath='models/model_'+m.__name__+'_.h5', save_weights_only=True, monitor='val_loss',
                                                                   save_best_only=True),
                                tf.keras.callbacks.CSVLogger(
                                    filename='Histories/CSV/model_'+m.__name__+'_.csv', separator=",", append=True)
                            ])
        
