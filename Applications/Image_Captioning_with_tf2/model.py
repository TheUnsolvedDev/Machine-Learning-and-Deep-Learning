import tensorflow as tf
import numpy as np

from param import *


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


def model(input_size=IMAGE_SIZE):
    inputs = tf.keras.layers.Input(input_size)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = ConvBlock(prev_Layer=x, filters=[64, 64, 128], strides=1)
    x = IdentityBlock(prev_Layer=x, filters=[64, 64, 128])
    x = ConvBlock(prev_Layer=x, filters=[128, 128, 256], strides=2)
    x = IdentityBlock(prev_Layer=x, filters=[128, 128, 256])
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    num_words = VOCAB
    state_size = STATE_SIZE
    embedding_size = EMBEDDING_SIZE

    decoder_transfer_map = tf.keras.layers.Dense(state_size,
                                                 activation='tanh',
                                                 name='decoder_transfer_map')(x)
    decoder_input = tf.keras.layers.Input(shape=(None, ), name='decoder_input')
    decoder_embedding = tf.keras.layers.Embedding(input_dim=num_words,
                                                  output_dim=embedding_size,
                                                  name='decoder_embedding')(decoder_input)
    decoder_gru1 = tf.keras.layers.GRU(state_size, name='decoder_gru1',
                                       return_sequences=True)([decoder_embedding, decoder_transfer_map])
    decoder_gru2 = tf.keras.layers.GRU(state_size, name='decoder_gru2',
                                       return_sequences=True)([decoder_gru1, decoder_transfer_map])
    outputs = tf.keras.layers.Dense(num_words,
                                    activation='softmax',
                                    name='decoder_output')(decoder_gru2)
    m = tf.keras.models.Model(inputs=[inputs, decoder_input],
                              outputs=outputs)
    m.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
    return m


if __name__ == '__main__':
    m = model()
    m.summary()
    m.save('ker_m.h5')
    tf.keras.utils.plot_model(m,  show_shapes=True, to_file="model.png")
