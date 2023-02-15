import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_inception_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, (label, label, label)


kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = tf.keras.layers.MaxPool2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = tf.keras.layers.Concatenate(axis=3, name=name)([conv_1x1, conv_3x3, conv_5x5,
                                                             pool_proj])

    return output


def InceptionNet():
    input_layer = tf.keras.layers.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu',
                               name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(
        1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')
    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(
        10, activation='softmax', name='auxilliary_output_1')(x1)
    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')
    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(
        10, activation='softmax', name='auxilliary_output_2')(x2)
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)
    model = tf.keras.Model(input_layer, [x, x1, x2], name='inception_v1')
    return model


if __name__ == '__main__':
    model = InceptionNet()
    model.summary()
    tf.keras.utils.plot_model(model,to_file = InceptionNet.__name__+'.png')

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
                .batch(batch_size=8, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=8, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=8, drop_remainder=True))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
