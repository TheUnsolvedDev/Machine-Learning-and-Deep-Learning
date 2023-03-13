import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_res_attn_net_v2.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


def attention_module(x, n_filters):
    skip = x

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        1, 1), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        1, 1), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.Multiply()([x, skip])
    x = tf.keras.layers.Add()([x, skip])

    return x


def residual_block(x, n_filters):
    skip = x

    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = attention_module(x, n_filters)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Concatenate(axis=-1)([x, skip])

    return x


def ResAttNet(input_shape=(32, 32, 3), n_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    input_shape = (32, 32, 3)
    num_classes = 10
    model = ResAttNet()
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file=ResAttNet.__name__+'.png', show_shapes=True)

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
