import tensorflow as tf
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_cnn_sent_classifier.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

num_words = 10000
maxlen = 200
embedding_dim = 50
num_filters = 64
filter_sizes = [3, 4, 5]


def cnn_model():
    inputs = tf.keras.layers.Input(shape=(maxlen,))
    embedding = tf.keras.layers.Embedding(
        num_words, embedding_dim, input_length=maxlen)(inputs)
    reshape = tf.keras.layers.Reshape((maxlen, embedding_dim, 1))(embedding)

    conv_0 = tf.keras.layers.Conv2D(num_filters, kernel_size=(
        filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)
    conv_1 = tf.keras.layers.Conv2D(num_filters, kernel_size=(
        filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)
    conv_2 = tf.keras.layers.Conv2D(num_filters, kernel_size=(
        filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='he_uniform', activation='relu')(reshape)

    maxpool_0 = tf.keras.layers.MaxPool2D(pool_size=(
        maxlen - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(
        maxlen - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(
        maxlen - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = tf.keras.layers.Concatenate(
        axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = tf.keras.layers.Flatten()(concatenated_tensor)
    dropout = tf.keras.layers.Dropout(0.5)(flatten)
    dense = tf.keras.layers.Dense(64, activation='relu')(dropout)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=num_words)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=maxlen)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=maxlen)
    model = cnn_model()
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=cnn_model.__name__+'.png', show_shapes=True, expand_nested=True)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=50,
              validation_data=(x_test, y_test),
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(x_test, y_test)
