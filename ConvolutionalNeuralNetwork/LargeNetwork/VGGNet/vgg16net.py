import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_vgg_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, label


def VGG16():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(64, 64, 3), filters=64,
                                     kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                     padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), name='vgg16'))
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='output'))
    return model


if __name__ == '__main__':
    model = VGG16()
    model.summary()
    tf.keras.utils.plot_model(model,to_file = VGG16.__name__+'.png')

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
                  optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    model.summary()
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
