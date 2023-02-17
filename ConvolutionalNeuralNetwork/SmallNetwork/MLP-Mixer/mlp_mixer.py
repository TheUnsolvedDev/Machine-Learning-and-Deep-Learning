import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MLP(tf.keras.layers.Layer):
    def __init__(self, hdim=512, out_dim=256):
        super().__init__()
        self.hdim = hdim
        self.out_dim = out_dim

    def call(self, x):
        x = tf.keras.layers.Dense(self.hdim, activation="linear")(x)
        x = tf.nn.gelu(x)
        x = tf.keras.layers.Dense(self.out_dim, activation="linear")(x)

        return x


class MixerLayer(tf.keras.layers.Layer):
    def __init__(self, hdim=512, image_size=256, n_channels=3):
        super().__init__()

        self.inp = tf.keras.layers.Input(
            shape=[n_channels, image_size, image_size])
        self.MLP1 = MLP(hdim, out_dim=image_size)
        self.MLP2 = MLP(hdim, out_dim=image_size)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        y = self.norm1(x)
        y = tf.transpose(y, [0, 2, 1])
        out_1 = self.MLP1(y)
        in_2 = tf.transpose(out_1, [0, 2, 1]) + x

        y = self.norm2(in_2)
        out_2 = self.MLP2(y) + in_2

        return out_2
    
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, num_patches, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.projection = tf.keras.layers.Conv2D(
            embed_dim, patch_size, strides=patch_size)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patches = self.projection(inputs)
        patches = tf.reshape(
            patches, (batch_size, self.num_patches, self.embed_dim))
        return patches


class MLPMixer(tf.keras.Model):
    def __init__(self, n_classes, depth, patch_size, image_size=28, n_channels=1, hdim=512):
        super().__init__()
        self.pp_fc = tf.keras.layers.Dense(image_size)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_patches = (image_size**2 / patch_size**2)

        self.mixer_layers = []
        for _ in range(depth):
            self.mixer_layers.append(
                MixerLayer(hdim)
            )

        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.head = tf.keras.layers.Dense(n_classes, activation="softmax")

    def call(self, x):
        x = self.pp_fc(x)
        for layer in self.mixer_layers:
            x = layer(x)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.expand_dims(x, axis=0)
        x = self.gap(x)
        out = self.head(x)

        return

    def summary(self):
        x = tf.keras.layers.Input(shape=(28, 28, 1))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_lenet5.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

if __name__ == '__main__':
    model = MLPMixer(
        n_classes=10,
        image_size=28,
        n_channels=1,
        patch_size=5,
        depth=3,
        hdim=64
    )
    model.summary()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshape the input images to 28x28x1 and normalize the pixel values to [0, 1]
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # One-hot encode the target labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    model.fit(x_train, y_train, batch_size=64, epochs=10,
              validation_data=(x_test, y_test))
