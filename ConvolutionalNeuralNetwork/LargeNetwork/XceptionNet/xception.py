import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_xception_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def conv_bn(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def sep_bn(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.SeparableConv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding='same',
                                        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def entry_flow(x):

    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = tf.keras.layers.ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3, strides=1)
    tensor = tf.keras.layers.ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])
    return x


def middle_flow(tensor):

    for _ in range(8):
        x = tf.keras.layers.ReLU()(tensor)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        tensor = tf.keras.layers.Add()([tensor, x])

    return tensor


def exit_flow(tensor):

    x = tf.keras.layers.ReLU()(tensor)
    x = sep_bn(x, filters=256,  kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=512, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Dense(units=10, activation='softmax')(x)

    return x


def XCeption():
    input = tf.keras.layers.Input(shape=(64, 64, 3))
    x = entry_flow(input)
    x = middle_flow(x)
    output = exit_flow(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, label


if __name__ == '__main__':
    model = XCeption()
    model.summary()
    tf.keras.utils.plot_model(model,to_file = XCeption.__name__+'.png')

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
