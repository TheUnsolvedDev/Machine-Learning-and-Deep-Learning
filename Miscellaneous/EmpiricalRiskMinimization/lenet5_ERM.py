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


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    y_train = tf.one_hot(y_train, 10)

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    y_test = tf.one_hot(y_test, 10)

    BATCH_SIZE = 64
    EPOCHS = 10

    val_samples = 2000
    x_val, y_val = x_train[:val_samples], y_train[:val_samples]
    new_x_train, new_y_train = x_train[val_samples:], y_train[val_samples:]

    train_ds_one = (
        tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
    )
    train_ds_two = (
        tf.data.Dataset.from_tensor_slices((new_x_train, new_y_train))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
    )
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_val, y_val)).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(BATCH_SIZE)
    
    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE
    )

    model = lenet5_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_ds_mu, validation_data=val_ds, epochs=EPOCHS)
    _, test_acc = model.evaluate(test_ds)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
