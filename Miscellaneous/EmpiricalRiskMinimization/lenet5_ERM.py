import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_lenet5_ERM.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def lenet5_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(
        3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    return model


def mixup(x, y, alpha=0.2):
    lam = tfp.distributions.Beta(alpha, alpha).sample((x.shape[0],))
    lam_vec = tf.reshape(lam, [x.shape[0], 1, 1, 1])

    index = tf.random.shuffle(tf.range(x.shape[0]))
    x = lam_vec * tf.cast(x, tf.float32) + (1 - lam_vec) * \
        tf.gather(tf.cast(x, tf.float32), index)
    return x, y


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train/255, axis=-1)
    x_test = np.expand_dims(x_test/255, axis=-1)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()

    train_ds = (train_ds
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=64, drop_remainder=True)
                .map(mixup))
    test_ds = (test_ds
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=64, drop_remainder=True))

    model = lenet5_model()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(train_ds, epochs=100,
              validation_data=test_ds, callbacks=callbacks)
