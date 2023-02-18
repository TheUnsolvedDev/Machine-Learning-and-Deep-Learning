import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

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


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = tf.keras.layers.Dense(embed_dim)
        self.wk = tf.keras.layers.Dense(embed_dim)
        self.wv = tf.keras.layers.Dense(embed_dim)

        self.dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention = tf.matmul(q, k, transpose_b=True)
        scaled_attention = scaled_attention / \
            tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))

        output = self.dense(output)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"),
             tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


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


class ViTModel(tf.keras.Model):
    def __init__(self, patch_size, num_patches, num_classes, embed_dim, num_heads, ff_dim, num_layers, rate=0.1):
        super(ViTModel, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.rate = rate

        self.patch_embedding = PatchEmbedding(
            patch_size, num_patches, embed_dim)
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim)
        self.transformer_blocks = [TransformerBlock(
            embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.classifier = tf.keras.layers.Dense(
            num_classes, activation="softmax")

    def call(self, inputs, training=True):
        patch_embeddings = self.patch_embedding(inputs)
        pos_embeddings = self.pos_embedding(
            tf.range(start=0, limit=self.num_patches, delta=1))
        embeddings = patch_embeddings + pos_embeddings
        embeddings = self.layernorm(embeddings)

        for i in range(self.num_layers):
            embeddings = self.transformer_blocks[i](embeddings, training)

        embeddings = self.dropout(embeddings, training)
        output = self.classifier(embeddings[:, 0, :])
        return output

    def summary(self):
        x = tf.keras.layers.Input(shape=(32, 32, 3))
        model = tf.keras.Model(
            inputs=[x], outputs=self.call(x, training=False))
        return model.summary()

    def model(self):
        x = tf.keras.layers.Input(shape=(32, 32, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


if __name__ == '__main__':
    model = ViTModel(patch_size=2, num_patches=16*16, num_classes=10,
                     embed_dim=64, num_heads=4, ff_dim=128, num_layers=6, rate=0.1).model()
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=ViTModel.__name__+'.png', show_shapes=True,expand_nested=True)

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
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
