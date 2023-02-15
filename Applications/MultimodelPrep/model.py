import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from params import *

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
    x = mobilnet_block(x, filters=128, strides=1)
    x = mobilnet_block(x, filters=256, strides=2)
    x = mobilnet_block(x, filters=256, strides=1)
    x = tf.keras.layers.AvgPool2D(
        pool_size=7, strides=1, data_format='channels_first')(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(
        units=NUM_CLASSES, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model


num_blocks = 3
num_layers_per_block = 4
growth_rate = 16
dropout_rate = 0.4
compress_factor = 0.5
eps = 1.1e-5
num_filters = 16


def H(inputs, num_filters, dropout_rate):
    x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.SeparableConv2D(
        num_filters, kernel_size=3, use_bias=False, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def transition(inputs, num_filters, compression_factor, dropout_rate):
    # compression_factor is the 'θ'
    x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    num_feature_maps = inputs.shape[1]  # The value of 'm'

    x = tf.keras.layers.Conv2D(np.floor(compression_factor * num_feature_maps).astype(np.int8),
                               kernel_size=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x


def dense_block(inputs, num_layers, num_filters, growth_rate, dropout_rate):
    for i in range(num_layers):  # num_layers is the value of 'l'
        conv_outputs = H(inputs, num_filters, dropout_rate)
        inputs = tf.keras.layers.Concatenate()([conv_outputs, inputs])
        # To increase the number of filters for each layer.
        num_filters += growth_rate
    return inputs, num_filters


def DenseNet(input_shape=IMG_SIZE):
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

    global num_filters
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda i: i/255.0)(inputs)
    x = data_augmentation(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(
        3, 3), use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    for i in range(num_blocks):
        x, num_filters = dense_block(
            x, num_layers_per_block, num_filters, growth_rate, dropout_rate)
        x = transition(x, num_filters, compress_factor, dropout_rate)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='output')(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model


kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = tf.keras.layers.MaxPool2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = tf.keras.layers.Concatenate(axis=3, name=name)([conv_1x1, conv_3x3, conv_5x5,
                                                             pool_proj])

    return output


def InceptionNet():
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
    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu',
                               name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', strides=(
        1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')
    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='auxilliary_output_1')(x1)
    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')
    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='auxilliary_output_2')(x2)
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='output')(x)
    model = tf.keras.Model(input_layer, [x, x1, x2], name='inception_v1')
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
    x = ConvBlock(prev_Layer=x, filters=[128, 128, 256], strides=2)
    x = IdentityBlock(prev_Layer=x, filters=[128, 128, 256])

    # Stage 6
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x, name='ResNet50')
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
    input = tf.keras.layers.Input(shape=IMG_SIZE)
    x = tf.keras.layers.Lambda(lambda i: i/255.0)(input)
    x = data_augmentation(x)
    x = entry_flow(x)
    x = middle_flow(x)
    output = exit_flow(x)

    model = tf.keras.Model(inputs=input, outputs=output)
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
        model_output = tf.keras.layers.Dense(NUM_CLASSES,
                                             activation='softmax')(pooling)

        model = tf.keras.models.Model(model_input, model_output)
    else:
        model = tf.keras.models.Model(model_input, block_3)
    return model


def Alexnet(input_shape=IMG_SIZE):
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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda i: i/255.0, input_shape=IMG_SIZE),
        data_augmentation,
        tf.keras.layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(
            4, 4), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(
            5, 5), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(
            3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(
            1, 1), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(
            1, 1), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def ZFnet():
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
    x = tf.keras.layers.Conv2D(
        64, (7, 7), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1)(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.image.per_image_standardization(x))(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), strides=1, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.image.per_image_standardization(x))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)

    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32)(x)

    x = tf.keras.layers.Dense(32)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def VGG():
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
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda i: i/255.0, input_shape=IMG_SIZE))
    model.add(data_augmentation)
    model.add(tf.keras.layers.Conv2D(input_shape=IMG_SIZE, filters=64,
                                     kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                     padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), name='vgg16'))
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='fc1'))
    model.add(tf.keras.layers.Dense(128, activation='relu', name='fc2'))
    model.add(tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='output'))
    return model


def fractal_block(input_tensor, filters, block_id):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu', name='conv_' + str(block_id))(input_tensor)
    x = tf.keras.layers.BatchNormalization(name='bn_' + str(block_id))(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu', name='conv_' + str(block_id + 1))(x)
    x = tf.keras.layers.BatchNormalization(name='bn_' + str(block_id + 1))(x)
    return x

# Define the fractal net model


def fractal_net(input_shape=(64, 64, 3), num_classes=10):
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
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda i: i/255.0)(inputs)
    x = data_augmentation(x)

    # Create the first recursive block
    x = fractal_block(x, 64, 0)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool_1')(x)
    x = fractal_block(x, 128, 2)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool_2')(x)

    # Create the second recursive block
    y = fractal_block(x, 128, 4)
    y = tf.keras.layers.UpSampling2D(size=2, name='upsample_1')(y)
    y = fractal_block(y, 64, 6)
    y = tf.keras.layers.UpSampling2D(size=2, name='upsample_2')(y)

    # Combine the two recursive blocks
    x = tf.keras.layers.Concatenate()([x, y])
    x = fractal_block(x, 64, 8)
    x = tf.keras.layers.Conv2D(NUM_CLASSES, 1, name='final_conv')(x)
    outputs = tf.keras.layers.Activation('softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def lenet5_model(input_shape=(64, 64, 3)):
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
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(lambda i: i/255.0, input_shape=IMG_SIZE))
    model.add(data_augmentation)
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(
        3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    model.add(tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax'))
    return model


def pyramidal_module(x, filters, strides, name=None):
    y = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides,
                               padding='same', name=name+'_conv_1x1')(x)
    y = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=name+'_bn_1x1')(y)
    y = tf.keras.layers.Activation('relu', name=name+'_relu_1x1')(y)

    y = tf.keras.layers.Conv2D(filters, (3, 3), strides=1,
                               padding='same', name=name+'_conv_3x3')(y)
    y = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=name+'_bn_3x3')(y)
    y = tf.keras.layers.Activation('relu', name=name+'_relu_3x3')(y)

    y = tf.keras.layers.Conv2D(filters * 4, (1, 1), strides=1,
                               padding='same', name=name+'_conv_1x1_bottleneck')(y)
    y = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=name+'_bn_1x1_bottleneck')(y)

    if strides != 1 or x.shape[3] != filters * 4:
        x = tf.keras.layers.Conv2D(filters * 4, (1, 1), strides=strides,
                                   padding='same', name=name+'_conv_1x1_shortcut')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=3, epsilon=1.001e-5, name=name+'_bn_1x1_shortcut')(x)

    y = tf.keras.layers.add([x, y], name=name+'_add')
    y = tf.keras.layers.Activation('relu', name=name+'_relu_add')(y)
    return y


def PyramidalNet(input_shape=IMG_SIZE):
    input_layer = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(
        2, 2), padding='same', name='conv_1_3x3/2')(input_layer)
    x = tf.keras.layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name='bn_1_3x3/2')(x)
    x = tf.keras.layers.Activation('relu', name='relu_1_3x3/2')(x)

    x = pyramidal_module(x, filters=64, strides=2, name='pyramidal_3')
    x = pyramidal_module(x, filters=128, strides=2, name='pyramidal_4')
    x = pyramidal_module(x, filters=128, strides=2, name='pyramidal_5')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_6_3x3/1')(x)

    x = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='output')(x)

    model = tf.keras.Model(input_layer, x, name='pyramidal_net')
    return model


if __name__ == '__main__':
    models = [MobileNet, DenseNet, InceptionNet,
          ResNext, XCeption, SE_ResNet, Alexnet, ZFnet,  VGG, lenet5_model, PyramidalNet]
    for model in models:
        m = model
        model = m()
        model.summary()
    # tf.keras.utils.plot_model(model, to_file=m.__name__+'.png')
