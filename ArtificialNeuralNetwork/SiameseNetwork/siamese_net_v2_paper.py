import tensorflow as tf
import random
import numpy as np
np.random.seed(1337)  # for reproducibility

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def euclidean_distance(vects):
    x, y = vects
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return tf.math.sqrt(tf.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1


def contrastive_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * tf.keras.backend.square(y_pred) + (1 - y_true) * tf.keras.backend.square(1 - y_pred))

def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def create_base_network_dense(input_dim):
    seq = tf.keras.models.Sequential()
    seq.add(tf.keras.layers.Dense(
        32, input_shape=input_dim, activation='relu'))
    seq.add(tf.keras.layers.Dropout(0.1))
    seq.add(tf.keras.layers.Dense(32, activation='relu'))
    seq.add(tf.keras.layers.Dropout(0.1))
    seq.add(tf.keras.layers.Dense(32, activation='relu'))
    return seq


def create_base_network(input_dim):
    img_colours, img_rows, img_cols = input_dim
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(nb_filters, nb_conv, nb_conv,
                                     padding='valid',
                                     input_shape=(img_colours, img_rows, img_cols)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(nb_filters, nb_conv, nb_conv))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.1)) #0.25 #too much dropout and loss -> nan
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        64, input_shape=(input_dim,), activation='relu'))
    # model.add(Dropout(0.05)) #too much dropout and loss -> nan
    model.add(tf.keras.layers.Dense(32, activation='relu'))

    return model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # input image dimensions
    img_rows, img_cols = 28, 28
    X_train = X_train.reshape(60000, 1, img_rows, img_cols)
    X_test = X_test.reshape(10000, 1, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    X_train /= 255
    X_test /= 255
    #input_dim = 784
    input_dim = (1, img_rows, img_cols)
    nb_epoch = 12

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(X_train, digit_indices)
    tr_y = tf.cast(tr_y, tf.float32)

    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(X_test, digit_indices)
    te_y = tf.cast(te_y, tf.float32)
    base_network = create_base_network_dense(input_dim)

    input_a = tf.keras.layers.Input(shape=(1, img_rows, img_cols,))
    input_b = tf.keras.layers.Input(shape=(1, img_rows, img_cols,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = tf.keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b])

    model = tf.keras.Model(inputs=[input_a, input_b], outputs=distance)

    # train
    rms = tf.keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
              batch_size=128,
              epochs=nb_epoch)

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
