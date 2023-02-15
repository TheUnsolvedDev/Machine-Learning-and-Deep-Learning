import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from dataset import *
from model import *
from params import *


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

callbacks = [
    # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

models = [MobileNet, DenseNet, InceptionNet,
          ResNext, XCeption, SE_ResNet, Alexnet, ZFnet,  VGG, lenet5_model, PyramidalNet]


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    label = tf.one_hot(label[0], depth=NUM_CLASSES)
    return image, label


if __name__ == '__main__':
    # dataset = Dataset()
    # train, val, test = dataset.generators()
    # train = load_dataset('records/train/')
    # val = load_dataset('records/val/')
    # test = load_dataset('records/val/')

    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (validation_images, validation_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(
        validation_ds).numpy()

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=BATCH_SIZE, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=BATCH_SIZE, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=BATCH_SIZE, drop_remainder=True))

    for ind, m in enumerate(models):
        print('***** Training model', ind, ':', m.__name__, '*****')
        model = m()
        model.summary()
        tf.keras.utils.plot_model(model, to_file='diagrams/architecture/'+m.__name__+'.png', show_shapes=True,
                                  show_layer_names=True, show_layer_activations=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(learning_rate=0.001), metrics=['accuracy', 'categorical_accuracy'])

        history = model.fit(train_ds, epochs=EPOCHS,
                            validation_data=validation_ds, callbacks=callbacks + [
                                tf.keras.callbacks.ModelCheckpoint(filepath='models/model_'+m.__name__+'_.h5', save_weights_only=True, monitor='val_loss',
                                                                   save_best_only=True),
                                tf.keras.callbacks.CSVLogger(
                                    filename='Histories/CSV/model_'+m.__name__+'_.csv', separator=",", append=True)
                            ])
        history_test = model.evaluate(test_ds, callbacks=callbacks + [
            tf.keras.callbacks.CSVLogger(
                filename='Histories/CSV/model_test_'+m.__name__+'_.csv', separator=",", append=True)
        ])
