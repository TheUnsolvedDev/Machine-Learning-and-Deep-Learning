import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_inception_net_v2.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


def inception_module(x, f1, f3_reduce, f3, f5_reduce, f5, pool_proj):
    conv1 = tf.keras.layers.Conv2D(
        f1, (1, 1), padding='same', activation='relu')(x)
    conv3_reduce = tf.keras.layers.Conv2D(
        f3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv3 = tf.keras.layers.Conv2D(
        f3, (3, 3), padding='same', activation='relu')(conv3_reduce)
    conv5_reduce = tf.keras.layers.Conv2D(
        f5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv5 = tf.keras.layers.Conv2D(
        f5, (5, 5), padding='same', activation='relu')(conv5_reduce)
    pool = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(
        pool_proj, (1, 1), padding='same', activation='relu')(pool)
    output = tf.keras.layers.Concatenate(axis=-1)(
        [conv1, conv3, conv5, pool_proj])
    return output


def inception_B(x):
    # 1x1 convolution
    conv1 = tf.keras.layers.Conv2D(
        192, (1, 1), padding='same', activation='relu')(x)
    conv2 = tf.keras.layers.Conv2D(
        192, (1, 1), padding='same', activation='relu')(x)
    conv2 = tf.keras.layers.Conv2D(
        192, (1, 7), padding='same', activation='relu')(conv2)
    conv2 = tf.keras.layers.Conv2D(
        192, (7, 1), padding='same', activation='relu')(conv2)
    output = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2])
    return output


def inception_C(x):
    conv1 = tf.keras.layers.Conv2D(
        320, (1, 1), padding='same', activation='relu')(x)
    conv2_reduce = tf.keras.layers.Conv2D(
        384, (1, 1), padding='same', activation='relu')(x)
    conv2_1 = tf.keras.layers.Conv2D(
        384, (1, 3), padding='same', activation='relu')(conv2_reduce)
    conv2_2 = tf.keras.layers.Conv2D(
        384, (3, 1), padding='same', activation='relu')(conv2_reduce)
    conv2 = tf.keras.layers.Concatenate(axis=-1)([conv2_1, conv2_2])
    output = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2])
    return output


def linear_bottleneck(x, filters, strides):
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shortcut = tf.keras.layers.Conv2D(
        filters, (1, 1), strides=strides, padding='same', use_bias=False)(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def InceptionV2(input_shape, num_classes):
    # input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    # stem module
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Inception modules
    x = inception_module(x, f1=16, f3_reduce=16, f3=16,
                         f5_reduce=16, f5=8, pool_proj=8)
    x = inception_module(x, f1=32, f3_reduce=32, f3=32,
                         f5_reduce=8, f5=16, pool_proj=16)
    x = inception_B(x)
    x = inception_module(x, f1=128, f3_reduce=32, f3=128,
                         f5_reduce=24, f5=16, pool_proj=16)
    x = inception_module(x, f1=128, f3_reduce=32, f3=128,
                         f5_reduce=24, f5=16, pool_proj=16)
    x = inception_module(x, f1=128, f3_reduce=32, f3=128,
                         f5_reduce=24, f5=16, pool_proj=16)
    x = inception_module(x, f1=128, f3_reduce=128, f3=128,
                         f5_reduce=48, f5=32, pool_proj=32)
    x = inception_C(x)
    x = inception_module(x, f1=128, f3_reduce=128, f3=128,
                         f5_reduce=48, f5=32, pool_proj=32)
    x = inception_module(x, f1=128, f3_reduce=128, f3=128,
                         f5_reduce=48, f5=32, pool_proj=32)

    # top layer
    x = tf.keras.layers.AveragePooling2D((4, 4))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    num_classes = 10
    model = InceptionV2(input_shape, num_classes)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file=InceptionV2.__name__+'.png', show_shapes=True)

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
                .batch(batch_size=16, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=16, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=16, drop_remainder=True))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
