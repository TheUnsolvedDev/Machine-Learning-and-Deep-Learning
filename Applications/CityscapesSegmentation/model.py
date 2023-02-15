import tensorflow as tf
import numpy as np

from params import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation("relu")(x)
    return x


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tf.keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size,
                      padding=padding, strides=strides)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(
        1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tf.keras.layers.Add()([conv, shortcut])
    return output


def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size,
                     padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size,
                     padding=padding, strides=1)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(
        1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tf.keras.layers.Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c


def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = tf.keras.layers.Input((*IMAGE_SIZE, 3))
    x = tf.keras.layers.Lambda(lambda i: i/255)(inputs)
    # Encoder
    e0 = x
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    # Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    # Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = tf.keras.layers.Conv2D(
        NUM_CLASSES, (1, 1), padding="same", activation="softmax")(d4)
    model = tf.keras.models.Model(inputs, outputs)
    return model


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


def Unet(output_channels=NUM_CLASSES):
    inputs = tf.keras.layers.Input(shape=[*IMAGE_SIZE, 3])
    inputs = tf.keras.layers.Lambda(lambda i: i/255)(inputs)

    down_stack = [
        downsample(8, 4, apply_batchnorm=False),
        downsample(32, 4),
        downsample(64, 4),
        downsample(128, 4),
        downsample(128, 4),
        downsample(128, 4),
        downsample(128, 4),
    ]

    up_stack = [
        upsample(128, 4, apply_dropout=True),
        upsample(128, 4, apply_dropout=True),
        upsample(128, 4, apply_dropout=True),
        upsample(128, 4),
        upsample(64, 4),
        upsample(32, 4),
        upsample(8, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4, activation='softmax',
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           use_bias=False)
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


""" Atrous Spatial Pyramid Pooling """


def ASPP(inputs):
    shape = inputs.shape

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(
        shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = tf.keras.layers.Conv2D(filters=32, kernel_size=1,
                                    padding='same', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = tf.keras.layers.Activation('relu', name=f'relu_1')(y_pool)
    y_pool = tf.keras.layers.UpSampling2D((shape[1], shape[2]),
                                          interpolation="bilinear")(y_pool)

    y_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, dilation_rate=1,
                                 padding='same', use_bias=False)(inputs)
    y_1 = tf.keras.layers.BatchNormalization()(y_1)
    y_1 = tf.keras.layers.Activation('relu')(y_1)

    y_6 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=6,
                                 padding='same', use_bias=False)(inputs)
    y_6 = tf.keras.layers.BatchNormalization()(y_6)
    y_6 = tf.keras.layers.Activation('relu')(y_6)

    y_12 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=12,
                                  padding='same', use_bias=False)(inputs)
    y_12 = tf.keras.layers.BatchNormalization()(y_12)
    y_12 = tf.keras.layers.Activation('relu')(y_12)

    y = tf.keras.layers.Concatenate()([y_pool, y_1, y_6, y_12])

    y = tf.keras.layers.Conv2D(filters=32, kernel_size=1, dilation_rate=1,
                               padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    return y


def DeepLabV3Plus(shape=(*IMAGE_SIZE, 3)):
    """ Inputs """
    inputs = tf.keras.layers.Input(shape)

    """ Pre-trained ResNet50 """
    base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                include_top=False, input_tensor=inputs)

    """ Pre-trained ResNet50 Output """
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    """ Get low-level features """
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = tf.keras.layers.Conv2D(filters=32, kernel_size=1,
                                 padding='same', use_bias=False)(x_b)
    x_b = tf.keras.layers.BatchNormalization()(x_b)
    x_b = tf.keras.layers.Activation('relu')(x_b)

    x = tf.keras.layers.Concatenate()([x_a, x_b])
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same',
                               activation='relu', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(x)

    """ Outputs """
    x = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), name='output_layer')(x)
    x = tf.keras.layers.Activation('softmax')(x)

    """ Model """
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def JPU(shape=(*IMAGE_SIZE, 3)):
    inputs = tf.keras.layers.Input(shape)
    inputs = tf.keras.layers.Lambda(lambda i: i/255)(inputs)

    # Add a series of convolutional and max pooling layers
    x = inputs
    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    # Add a JPU module with three branches at different resolutions
    branch1 = tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, padding='same', activation='relu')(x)
    branch2 = tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, padding='same', activation='relu')(x)
    branch2 = tf.keras.layers.UpSampling2D(size=2)(branch2)
    branch3 = tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, padding='same', activation='relu')(x)
    branch3 = tf.keras.layers.UpSampling2D(size=4)(branch3)

    branch1 = tf.keras.layers.ZeroPadding2D(
        padding=((0, IMAGE_SIZE[0]-branch1.shape[1]), (0, IMAGE_SIZE[1]-branch1.shape[2])))(branch1)
    branch2 = tf.keras.layers.ZeroPadding2D(
        padding=((0, IMAGE_SIZE[0]-branch2.shape[1]), (0, IMAGE_SIZE[1]-branch2.shape[2])))(branch2)
    branch3 = tf.keras.layers.ZeroPadding2D(
        padding=((0, IMAGE_SIZE[0]-branch3.shape[1]), (0, IMAGE_SIZE[1]-branch3.shape[2])))(branch3)
    x = tf.keras.layers.Concatenate()([branch1, branch2, branch3])

    # Add a final convolutional layer with a single output channel for each pixel
    outputs = tf.keras.layers.Conv2D(
        filters=NUM_CLASSES, kernel_size=1, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class GatedSCNNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, activation, **kwargs):
        super(GatedSCNNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding=self.padding, activation=self.activation)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding=self.padding, activation=self.activation)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding=self.padding, activation=self.activation)
        super(GatedSCNNBlock, self).build(input_shape)

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x1 * tf.sigmoid(x2) + x3 * tf.sigmoid(x2)
        return x


def Gated_SCNN(shape=(*IMAGE_SIZE, 3)):
    inputs = tf.keras.layers.Input(shape)
    inputs = tf.keras.layers.Lambda(lambda i: i/255)(inputs)
    x = inputs
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    for filters in [128, 128, 128]:
        x = GatedSCNNBlock(filters=filters, kernel_size=3,
                           padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D(size=2)(x)
    outputs = tf.keras.layers.Conv2D(
        filters=NUM_CLASSES, kernel_size=1, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = JPU()
    model.summary()
