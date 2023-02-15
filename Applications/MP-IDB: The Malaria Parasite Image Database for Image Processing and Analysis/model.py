from params import *
import numpy as np
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def mobilnet_block(x, filters, strides):
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def MobileNet():
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )

    input = tf.keras.layers.Input(shape=IMG_SIZE)
    x = tf.keras.layers.Lambda(lambda i: i/255.0)(input)
    x = data_augmentation(x)
    x = tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = mobilnet_block(x, filters=64, strides=1)
    x = mobilnet_block(x, filters=128, strides=2)
    x = mobilnet_block(x, filters=256, strides=1)
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(
        units=NUM_CLASSES, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def IdentityBlock(prev_Layer, filters):
    f1, f2, f3 = filters
    block = []

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(
            1, 1), strides=(1, 1), padding='valid')(prev_Layer)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(
            3, 3), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f3, kernel_size=(
            1, 1), strides=(1, 1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        block.append(x)

    block.append(prev_Layer)
    x = tf.keras.layers.Add()(block)
    x = tf.keras.layers.Activation(activation='relu')(x)

    return x


def ConvBlock(prev_Layer, filters, strides):
    f1, f2, f3 = filters

    block = []

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(
            1, 1), padding='valid', strides=strides)(prev_Layer)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(
            3, 3), padding='same', strides=(1, 1))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        x = tf.keras.layers.Conv2D(filters=f3, kernel_size=(
            1, 1), padding='valid', strides=(1, 1))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        block.append(x)

    x2 = tf.keras.layers.Conv2D(filters=f3, kernel_size=(
        1, 1), padding='valid', strides=strides)(prev_Layer)
    x2 = tf.keras.layers.BatchNormalization(axis=3)(x2)

    block.append(x2)
    x = tf.keras.layers.Add()(block)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x


def ResNext():
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    input_layer = tf.keras.layers.Input(shape=IMG_SIZE)
    x = tf.keras.layers.Lambda(lambda i: i/255.0)(input_layer)
    x = data_augmentation(x)
    # Stage 1
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Stage 2
    x = ConvBlock(prev_Layer=x, filters=[64, 64, 128], strides=1)
    x = IdentityBlock(prev_Layer=x, filters=[64, 64, 128])

    # Stage 3
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='ResNet50')
    return model


def squeeze_excitation_layer(input_layer, out_dim, ratio, conv):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_layer)

    excitation = tf.keras.layers.Dense(
        units=out_dim / ratio, activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(
        out_dim, activation='sigmoid')(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

    scale = tf.keras.layers.multiply([input_layer, excitation])

    if conv:
        shortcut = tf.keras.layers.Conv2D(out_dim, kernel_size=1, strides=1,
                                          padding='same', kernel_initializer='he_normal')(input_layer)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_layer
    out = tf.keras.layers.Add()([shortcut, scale])
    return out


def conv_block(input_layer, filters):
    layer = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(input_layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(filters*4, kernel_size=1, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    return layer


def SE_ResNet(input_w=IMG_SIZE[0], input_h=IMG_SIZE[1], include_top=True):
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    model_input = tf.keras.layers.Input(shape=(input_w, input_h, 3))
    x = tf.keras.layers.Lambda(lambda i: i/255.0)(model_input)
    x = data_augmentation(x)
    identity_blocks = [3, 4, 6, 3]
    # Block 1
    layer = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(x)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    block_1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(layer)

    # Block 2
    block_2 = conv_block(block_1, 64)
    block_2 = squeeze_excitation_layer(
        block_2, out_dim=256, ratio=32.0, conv=True)
    for _ in range(identity_blocks[0]-1):
        block_2 = conv_block(block_1, 64)
        block_2 = squeeze_excitation_layer(
            block_2, out_dim=256, ratio=32.0, conv=False)

    # Block 3
    block_3 = conv_block(block_2, 128)
    block_3 = squeeze_excitation_layer(
        block_3, out_dim=512, ratio=32.0, conv=True)
    for _ in range(identity_blocks[1]-1):
        block_3 = conv_block(block_2, 128)
        block_3 = squeeze_excitation_layer(
            block_3, out_dim=512, ratio=32.0, conv=False)

    if include_top:
        pooling = tf.keras.layers.GlobalAveragePooling2D()(block_3)
        x = tf.keras.layers.Dense(128, activation='relu')(pooling)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        model_output = tf.keras.layers.Dense(NUM_CLASSES,
                                             activation='softmax')(x)

        model = tf.keras.models.Model(model_input, model_output)
    else:
        model = tf.keras.models.Model(model_input, block_3)
    return model


def conv_bn(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def sep_bn(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.SeparableConv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding='same',
                                        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def entry_flow(x):
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = tf.keras.layers.ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3, strides=1)
    tensor = tf.keras.layers.ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])
    return x


def middle_flow(tensor):
    for _ in range(8):
        x = tf.keras.layers.ReLU()(tensor)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        tensor = tf.keras.layers.Add()([tensor, x])

    return tensor


def exit_flow(tensor):
    x = tf.keras.layers.ReLU()(tensor)
    x = sep_bn(x, filters=256,  kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=512, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    x = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(x)
    return x


def XCeption():
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    inputs = tf.keras.layers.Input(shape=IMG_SIZE)
    x = tf.keras.layers.Lambda(lambda i: i/255.0)(inputs)
    x = data_augmentation(x)
    x = entry_flow(x)
    x = middle_flow(x)
    output = exit_flow(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


if __name__ == '__main__':
    models = [MobileNet, ResNext, SE_ResNet, XCeption]
    for model in models:
        model().summary()
