import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_ViT.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


image_size = 32
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
mlp_head_units = [128, 64]

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Normalization(),
        tf.keras.layers.Resizing(image_size, image_size),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def ViTModel():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    representation = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units,
                   dropout_rate=0.5)
    logits = tf.keras.layers.Dense(10)(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


if __name__ == '__main__':
    model = ViTModel()
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=ViTModel.__name__+'.png', show_shapes=True, expand_nested=True)

    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    
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
                .batch(batch_size=256, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=128, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=256, drop_remainder=True))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.optimizers.Adam(0.001), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                                                                tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),])
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
