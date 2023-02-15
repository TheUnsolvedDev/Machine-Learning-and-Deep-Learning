import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_res_net_CBAM.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(
            self.channels // self.reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.channels, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.global_avg_pool(inputs)
        x = tf.reshape(x, (-1, 1, 1, self.channels))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(1, (7, 7), padding='same')

    def call(self, inputs, **kwargs):
        x = tf.nn.sigmoid(self.conv(inputs))
        return x


class ConvolutionBlockAttentionModule(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(ConvolutionBlockAttentionModule, self).__init__(**kwargs)
        self.channels = channels
        self.channel_attention = ChannelAttention(self.channels)
        self.spatial_attention = SpatialAttention()

    def call(self, inputs, **kwargs):
        channel_attention = self.channel_attention(inputs)
        spatial_attention = self.spatial_attention(inputs)
        x = inputs * channel_attention * spatial_attention
        return x


def resnet_block(inputs, filters, strides=1):
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if strides != 1:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), strides=strides)(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    x = ConvolutionBlockAttentionModule(filters)(x)

    return x


def ResNet18(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 512, strides=2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, x)
    return model


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


if __name__ == '__main__':
    model = ResNet18((32, 32, 3), 10)
    model.summary()
    tf.keras.utils.plot_model(model, to_file=ResNet18.__name__+'.png')

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
                  optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
