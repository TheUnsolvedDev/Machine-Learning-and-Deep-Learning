import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='my_shakespeare_model.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def dataset(sequence, length, shuffle=False, seed=None, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


def text_model(n_tokens=39):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dense(n_tokens, activation="softmax")
    ])


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text], verbose=0)[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


def extend_text(text, n_chars=150, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


if __name__ == '__main__':
    filepath = tf.keras.utils.get_file(
        'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    with open(filepath) as f:
        shakespeare_text = f.read()

    if '.pkl' not in os.listdir('./'):
        text_vec_layer.adapt([shakespeare_text])
        pickle.dump({'config': text_vec_layer.get_config(),
                     'weights': text_vec_layer.get_weights()}, open("tv_layer.pkl", "wb"))
    else:
        from_disk = pickle.load(open("tv_layer.pkl", "rb"))
        text_vec_layer.from_config(from_disk['config'])
        # You have to call `adapt` with some dummy data (BUG in Keras)
        text_vec_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        text_vec_layer.set_weights(from_disk['weights'])

    encoded = text_vec_layer([shakespeare_text])[0]

    encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
    n_tokens = text_vec_layer.vocabulary_size() - 2
    dataset_size = len(encoded)

    print(dataset_size)
    print(list(dataset(text_vec_layer(["To be"])[0], length=4)))

    length = 100
    train_set = dataset(encoded[:4_000_000], length=length, shuffle=True,
                        seed=42)
    valid_set = dataset(encoded[4_000_000:4_060_000], length=length)
    test_set = dataset(encoded[4_060_000:], length=length)

    model = text_model(n_tokens)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                  metrics=["accuracy"])
    # history = model.fit(train_set, validation_data=valid_set, epochs=10,
    #                     callbacks=callbacks)
    model.load_weights('my_shakespeare_model.h5')

    shakespeare_model = tf.keras.Sequential([
        text_vec_layer,
        tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
        model
    ])

    print(extend_text("To be or not to be", temperature=0.01))
    print(extend_text("To be or not to be", temperature=0.1))
    print(extend_text("To be or not to be", temperature=1))
