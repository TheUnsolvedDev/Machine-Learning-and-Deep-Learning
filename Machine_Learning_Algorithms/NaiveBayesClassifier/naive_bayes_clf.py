from tqdm import tqdm
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import datasets, model_selection
from mpl_toolkits.mplot3d import Axes3D


def plot_data(x, y, labels, colours):
    for y_class in np.unique(y):
        index = np.where(y == y_class)
        plt.scatter(x[index, 0], x[index, 1],
                    label=labels[y_class], c=colours[y_class])
    plt.title("Training set")
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.legend()


@tf.function
def negative_log_likelyhood(dist, x_train, y_train):
    log_probs = dist.log_prob(x_train)
    length = len(tf.unique(y_train)[0])
    y_train = tf.one_hot(y_train, depth=length)
    return -tf.reduce_mean(log_probs*y_train)


@tf.function
def get_loss_and_grads(dist, x_train, y_train):
    with tf.GradientTape() as tape:
        tape.watch(dist.trainable_variables)
        loss = negative_log_likelyhood(dist, x_train, y_train)
        grads = tape.gradient(loss, dist.trainable_variables)
    return loss, grads


def learn_parameters(x, y, mus, scales, optimizer, epochs):
    nll_loss = []
    mu_values = []
    scales_values = []
    x = tf.cast(np.expand_dims(x, axis=1), tf.float32)
    dist = tfp.distributions.MultivariateNormalDiag(loc=mus, scale_diag=scales)
    for epoch in tqdm(range(epochs)):
        loss, grads = get_loss_and_grads(dist, x, y)
        optimizer.apply_gradients(zip(grads, dist.trainable_variables))
        nll_loss.append(loss)
        mu_values.append(mus.numpy())
        scales_values.append(scales.numpy())
    nll_loss, mu_values, scales_values = np.array(
        nll_loss), np.array(mu_values), np.array(scales_values)
    return (nll_loss, mu_values, scales_values, dist)


def get_prior(y):
    """
    This function takes training labels as a numpy array y of shape (num_samples,) as an input,
    and builds a Categorical Distribution object with empty batch shape and event shape,
    with the probability of each class.
    """
    counts = np.bincount(y)
    dist = tfp.distributions.Categorical(probs=counts/len(y))
    return dist


def predict_class(prior, class_conditionals, x):
    def predict_fn(myx):
        class_probs = class_conditionals.prob(tf.cast(myx, dtype=tf.float32))
        prior_probs = tf.cast(prior.probs, dtype=tf.float32)
        class_times_prior_probs = class_probs * prior_probs
        # Technically, this step
        Q = tf.reduce_sum(class_times_prior_probs)
        # and this one, are not necessary.
        P = tf.math.divide(class_times_prior_probs, Q)
        Y = tf.cast(tf.argmax(P), dtype=tf.float64)
        return Y
    y = tf.map_fn(predict_fn, x)
    return y


def get_meshgrid(x0_range, x1_range, n_points=100):
    x0 = np.linspace(x0_range[0], x0_range[1], n_points)
    x1 = np.linspace(x1_range[0], x1_range[1], n_points)
    return np.meshgrid(x0, x1)


def contour_plot(x0_range, x1_range, prob_fn, batch_shape, levels=None, n_points=100):
    X0, X1 = get_meshgrid(x0_range, x1_range, n_points=n_points)
    # X0.shape = (100, 100)
    # X1.shape = (100, 100)
    X_values = np.expand_dims(np.array([X0.ravel(), X1.ravel()]).T, axis=1)
    # X_values.shape = (1000, 1, 2)
    Z = prob_fn(X_values)
    # Z.shape = (10000, 3)
    Z = np.array(Z).T.reshape(batch_shape, *X0.shape)
    # Z.shape = (3, 100, 100)
    for batch in np.arange(batch_shape):
        plt.contourf(X0, X1, Z[batch], alpha=0.3, levels=levels)


if __name__ == '__main__':
    # Load the dataset
    iris = datasets.load_iris()

    # Use only the first two features: sepal length and width
    data = iris.data[:, :2]
    targets = iris.target

    # Randomly shuffle the data and make train and test splits
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(data, targets, test_size=0.2)

    # Plot the training data
    labels = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}
    label_colours = ['blue', 'red', 'green']

    plt.figure(figsize=(8, 5))
    plot_data(x_train, y_train, labels, label_colours)
    plt.show()

    mus = tf.Variable([[1., 1.], [1., 1.], [1., 1.]])
    scales = tf.Variable([[1., 1.], [1., 1.], [1., 1.]])
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    epochs = 5000*2
    nlls, mu_arr, scales_arr, class_conditionals = \
        learn_parameters(x_train, y_train, mus, scales, opt, epochs)

    # Plot the loss and convergence of the standard deviation parameters
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(nlls)
    ax[0].set_title("Loss vs. epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Negative log-likelihood")
    for k in [0, 1, 2]:
        ax[1].plot(mu_arr[:, k, 0])
        ax[1].plot(mu_arr[:, k, 1])
    ax[1].set_title("ML estimates for model's\nmeans vs. epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Means")
    for k in [0, 1, 2]:
        ax[2].plot(scales_arr[:, k, 0])
        ax[2].plot(scales_arr[:, k, 1])
    ax[2].set_title("ML estimates for model's\nscales vs. epoch")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Scales")
    plt.show()

    x0_range = x_train[:, 0].min(), x_train[:, 0].max()
    x1_range = x_train[:, 1].min(), x_train[:, 1].max()
    X0, X1 = get_meshgrid(x0_range, x1_range, n_points=300)
    X_v = np.expand_dims(np.array([X0.ravel(), X1.ravel()]).T, axis=1)
    Z = class_conditionals.prob(X_v)
    Z = np.array(Z).T.reshape(3, *X0.shape)

    print("Class conditional means:")
    print(class_conditionals.loc.numpy())
    print("\nClass conditional standard deviations:")
    print(class_conditionals.stddev().numpy())

    prior = get_prior(y_train)
    print(prior.probs)

    predictions = predict_class(prior, class_conditionals, x_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Test accuracy: {:.4f}".format(accuracy))

    plt.figure(figsize=(10, 6))
    plot_data(x_train, y_train, labels=labels, colours=label_colours)
    contour_plot(x0_range, x1_range,
                 lambda x: predict_class(prior, class_conditionals, x),
                 1, n_points=3, levels=[-0.5, 0.5, 1.5, 2])
    plt.title("Training set with decision regions")
    plt.show()
