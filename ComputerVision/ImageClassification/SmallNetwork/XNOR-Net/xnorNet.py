import tensorflow as tf
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_XNOR_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

def binary(x):
    return tf.where(x >= 0, tf.ones_like(x), -tf.ones_like(x))

def XNOR_net():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation=None, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Lambda(binary)(x)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(64, 3, activation=None, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Lambda(binary)(x)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(128, 3, activation=None, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Lambda(binary)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    xnornet = tf.keras.Model(inputs=inputs, outputs=x)
    return xnornet



def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

if __name__ == '__main__':
    model = XNOR_net()
    tf.keras.utils.plot_model(
        model, to_file=XNOR_net.__name__+'.png', show_shapes=True)

    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((-1,28,28,1))
    test_images = test_images.reshape((-1,28,28,1))
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
                .batch(batch_size=32, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=8, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=32, drop_remainder=True))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(0.001), metrics=['accuracy'])
    model.summary()
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
