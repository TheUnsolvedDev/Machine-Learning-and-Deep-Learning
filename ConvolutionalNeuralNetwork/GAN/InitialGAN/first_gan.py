import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import sys
from silence_tensorflow import silence_tensorflow
from params import *


silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def gen_noise(batch_size, z_dim):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


def make_discriminaor(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Input(IMAGE_SHAPE),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,  activation=None, input_shape=input_shape),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(64,  activation=None),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def make_generator(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(784, activation='tanh'),
        tf.keras.layers.Reshape(IMAGE_SHAPE)
    ])


@tf.function
def d_loss_fn(real_logits, fake_logits):
    return -tf.reduce_mean(tf.math.log(real_logits + 1e-10) + tf.math.log(1. - fake_logits + 1e-10))


@tf.function
def g_loss_fn(fake_logits):
    return -tf.reduce_mean(tf.math.log(fake_logits + 1e-10))


@tf.function
def train_step(gen, disc, batch_size, images, z_dim):
    noise = gen_noise(batch_size, z_dim)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise)
        disc_fake_pred = disc(generated_images)
        disc_real_pred = disc(images)

        disc_loss = d_loss_fn(disc_real_pred, disc_fake_pred)
        gen_loss = g_loss_fn(disc_fake_pred)

    disc_gradients = disc_tape.gradient(
        disc_loss, disc.trainable_variables)
    disc_optimizer.apply_gradients(
        zip(disc_gradients, disc.trainable_variables))

    gen_gradients = gen_tape.gradient(
        gen_loss, gen.trainable_variables)
    gen_optimizer.apply_gradients(
        zip(gen_gradients, gen.trainable_variables))
    return gen_loss, disc_loss


fig = plt.figure(figsize=(4, 4))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.ion()
    plt.clf()
    for i in range(predictions.shape[0]):
        prediction = np.array(predictions[i]).reshape(28, 28)
        plt.subplot(4, 4, i+1)
        plt.imshow(prediction * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.savefig('first_gan/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close("all")
    # plt.show(block='False')


g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')


def train_model(train, val, epochs=ITERATION, batch_size=BATCH_SIZE):
    gen = make_generator((Z_DIM,))
    gen.summary()
    disc = make_discriminaor((IMAGE_SIZE,))
    disc.summary()

    try:
        gen.load_weights('first_gan_weights/generator_first_gan.h5')
        disc.load_weights('first_gan_weights/discriminator_first_gan.h5')
    except FileNotFoundError:
        pass

    train = train.batch(batch_size=batch_size)
    val = val.batch(batch_size=batch_size)

    min_loss_gen = np.inf
    for epoch in range(epochs+1):
        for ind, images in enumerate(train):
            curr_batch_size = images.shape[0]
            g_loss, d_loss = train_step(
                gen, disc, curr_batch_size, images, Z_DIM)
            g_loss_metrics(g_loss)
            d_loss_metrics(d_loss)
            total_loss_metrics(g_loss + d_loss)

        template = '\r[{}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
        print(template.format(epoch, ITERATION, d_loss_metrics.result(),
                              g_loss_metrics.result(), total_loss_metrics.result()), end=' ')
        sys.stdout.flush()

        if epoch % EVERY_STEP == 0:
            noise = gen_noise(16, Z_DIM)
            generate_and_save_images(gen,
                                     epoch + 1,
                                     noise)

            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            total_loss_metrics.reset_states()

            loss = g_loss
            if loss <= min_loss_gen:
                disc.save_weights('first_gan_weights/discriminator_first_gan.h5')
                gen.save_weights('first_gan_weights/generator_first_gan.h5')
                loss = min_loss_gen


if __name__ == '__main__':
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.mnist.load_data()

    train_data = (train_data.reshape(-1, 28, 28, 1) - 127.5)/127.5
    test_data = (test_data.reshape(-1, 28, 28, 1) - 127.5)/127.5
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_optimizer = tf.keras.optimizers.Adam(G_LR)
    disc_optimizer = tf.keras.optimizers.Adam(D_LR)

    train_data = tf.data.Dataset.from_tensor_slices(
        train_data).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)
    test_data = tf.data.Dataset.from_tensor_slices(
        test_data).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)

    train_model(train_data, test_data)
