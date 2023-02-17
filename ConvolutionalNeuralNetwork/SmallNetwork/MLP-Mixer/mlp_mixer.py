import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_mlp_mixer.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(units, activation="gelu")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(units, activation="softmax")

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def MLPMixer():
    num_blocks = 4
    hidden_units = 64
    dropout_rate = 0.1
    patch_size = 5

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))

    # Stem
    x = tf.keras.layers.Conv2D(
        hidden_units, patch_size, strides=patch_size, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    # MLP blocks
    for i in range(num_blocks):
        x1 = tf.keras.layers.LayerNormalization()(x)
        x2 = tf.keras.layers.Permute((2, 1, 3))(x)
        x2 = tf.keras.layers.Reshape((-1, x2.shape[-1]))(x2)
        x2 = MLPBlock(hidden_units, dropout_rate)(x2)
        x2 = tf.keras.layers.Reshape((x.shape[1], x.shape[2], -1))(x2)
        x = tf.keras.layers.Add()([x, x2])
        x3 = tf.keras.layers.LayerNormalization()(x)
        x4 = tf.keras.layers.Reshape((-1, x3.shape[-1]))(x3)
        x4 = MLPBlock(hidden_units, dropout_rate)(x4)
        x4 = tf.keras.layers.Reshape((x.shape[1], x.shape[2], -1))(x4)
        x = tf.keras.layers.Add()([x, x4])

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(hidden_units*2, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    model = MLPMixer()
    model.summary()

    tf.keras.utils.plot_model(
        model, to_file='mlp_mixer_model.png', show_shapes=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # One-hot encode the target labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    model.fit(x_train, y_train, batch_size=64, epochs=50,
              validation_data=(x_test, y_test), callbacks=callbacks)
