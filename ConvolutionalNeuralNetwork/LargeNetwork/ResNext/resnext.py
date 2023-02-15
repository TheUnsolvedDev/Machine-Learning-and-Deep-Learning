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
        filepath='model_ResNext_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def IdentityBlock(prev_Layer, filters):
    f1, f2, f3 = filters
    block = []

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(
            1, 1), strides=(1, 1), padding='valid')(prev_Layer)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(
            3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f3, kernel_size=(
            1, 1), strides=(1, 1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        block.append(x)

    block.append(prev_Layer)
    x = tf.keras.layers.Add()(block)
    x = tf.keras.layers.Activation(activation='relu')(x)

    return x


def ConvBlock(prev_Layer, filters, strides):
    f1, f2, f3 = filters

    block = []

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(
            1, 1), padding='valid', strides=strides)(prev_Layer)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(
            3, 3), padding='same', strides=(1, 1))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f3, kernel_size=(
            1, 1), padding='valid', strides=(1, 1))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        block.append(x)

    x2 = tf.keras.layers.Conv2D(filters=f3, kernel_size=(
        1, 1), padding='valid', strides=strides)(prev_Layer)
    x2 = tf.keras.layers.BatchNormalization(axis=3)(x2)

    block.append(x2)
    x = tf.keras.layers.Add()(block)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x


def ResNext():
    input_layer = tf.keras.layers.Input(shape=IMG_SIZE)
    # Stage 1
    x = tf.keras.layers.ZeroPadding2D((3, 3))(input_layer)
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Stage 2
    x = ConvBlock(prev_Layer=x, filters=[64, 64, 128], strides=1)
    x = IdentityBlock(prev_Layer=x, filters=[64, 64, 128])

    # Stage 3
    x = ConvBlock(prev_Layer=x, filters=[128, 128, 256], strides=2)
    x = IdentityBlock(prev_Layer=x, filters=[128, 128, 256])

    # Stage 6
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='ResNet50')
    return model


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, label


if __name__ == '__main__':
    model = ResNext()
    model.summary()
    tf.keras.utils.plot_model(model, to_file=ResNext.__name__+'.png')

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
