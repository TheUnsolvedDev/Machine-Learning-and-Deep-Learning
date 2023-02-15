import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

IMG_SIZE = (64, 64, 3)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_SE_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def pyramidal_module(x, filters, strides, name=None):
    y = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides,
                               padding='same', name=name+'_conv_1x1')(x)
    y = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=name+'_bn_1x1')(y)
    y = tf.keras.layers.Activation('relu', name=name+'_relu_1x1')(y)

    y = tf.keras.layers.Conv2D(filters, (3, 3), strides=1,
                               padding='same', name=name+'_conv_3x3')(y)
    y = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=name+'_bn_3x3')(y)
    y = tf.keras.layers.Activation('relu', name=name+'_relu_3x3')(y)

    y = tf.keras.layers.Conv2D(filters * 4, (1, 1), strides=1,
                               padding='same', name=name+'_conv_1x1_bottleneck')(y)
    y = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=name+'_bn_1x1_bottleneck')(y)

    if strides != 1 or x.shape[3] != filters * 4:
        x = tf.keras.layers.Conv2D(filters * 4, (1, 1), strides=strides,
                                   padding='same', name=name+'_conv_1x1_shortcut')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=3, epsilon=1.001e-5, name=name+'_bn_1x1_shortcut')(x)

    y = tf.keras.layers.add([x, y], name=name+'_add')
    y = tf.keras.layers.Activation('relu', name=name+'_relu_add')(y)
    return y


def PyramidalNet(input_shape=IMG_SIZE):
    input_layer = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(
        2, 2), padding='same', name='conv_1_3x3/2')(input_layer)
    x = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name='bn_1_3x3/2')(x)
    x = tf.keras.layers.Activation('relu', name='relu_1_3x3/2')(x)

    x = pyramidal_module(x, filters=64, strides=2, name='pyramidal_3')
    x = pyramidal_module(x, filters=128, strides=2, name='pyramidal_4')
    x = pyramidal_module(x, filters=128, strides=2, name='pyramidal_5')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_6_3x3/1')(x)

    x = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)

    model = tf.keras.Model(input_layer, x, name='pyramidal_net')
    return model


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, label


if __name__ == '__main__':
    model = PyramidalNet()
    model.summary()
    tf.keras.utils.plot_model(model, to_file=PyramidalNet.__name__+'.png')

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
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=64, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=64, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=64, drop_remainder=True))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
