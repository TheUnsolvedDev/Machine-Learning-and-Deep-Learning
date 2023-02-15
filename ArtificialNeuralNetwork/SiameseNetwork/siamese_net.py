import itertools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def make_paired_dataset(X, y):
    X_pairs, y_pairs = [], []

    tuples = [(x1, y1) for x1, y1 in zip(X, y)]

    for t in tqdm.tqdm(itertools.product(tuples, tuples)):
        pair_A, pair_B = t
        img_A, label_A = t[0]
        img_B, label_B = t[1]

        new_label = int(label_A == label_B)

        X_pairs.append([img_A, img_B])
        y_pairs.append(new_label)

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    return X_pairs, y_pairs


def siamese_model():
    img_A_inp = tf.keras.layers.Input((28, 28), name='img_A_inp')
    img_B_inp = tf.keras.layers.Input((28, 28), name='img_B_inp')

    def get_dnn_block(depth):
        return tf.keras.models.Sequential([tf.keras.layers.Dense(depth),
                           tf.keras.layers.ReLU()])

    DEPTH = 64
    cnn = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                      get_dnn_block(DEPTH),
                      get_dnn_block(DEPTH*2),
                      tf.keras.layers.Dense(64, activation='relu')])

    feature_vector_A = cnn(img_A_inp)
    feature_vector_B = cnn(img_B_inp)

    concat = tf.keras.layers.Concatenate()([feature_vector_A, feature_vector_B])

    dense = tf.keras.layers.Dense(64, activation='relu')(concat)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.Model(inputs=[img_A_inp, img_B_inp], outputs=output)

    model.summary()
    return model


if __name__ == '__main__':
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.mnist.load_data()
    random_indices = np.random.choice(train_data.shape[0], 1000, replace=False)
    X_train_sample, y_train_sample = train_data[random_indices], train_labels[random_indices]

    random_indices2 = np.random.choice(train_data.shape[0], 500, replace=False)
    train_datas, train_label = make_paired_dataset(
        X_train_sample, y_train_sample)
    X_train_sample, y_train_sample = train_data[random_indices2], train_labels[random_indices2]
    test_data, test_label = make_paired_dataset(X_train_sample, y_train_sample)

    dnn = siamese_model()
    dnn.summary()
    tf.keras.utils.plot_model(dnn, to_file='model.png', show_shapes=True)
    dnn.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    dnn.fit(x = [train_datas[:, 0, :, :],train_datas[:, 1, :, :]], y = train_label, epochs=30,
            validation_data=([test_data[:, 0, :, :],test_data[:, 1, :, :]], test_label))
