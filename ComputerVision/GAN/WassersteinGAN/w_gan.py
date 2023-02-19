import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from silence_tensorflow import silence_tensorflow
from params import *


silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def make_discriminaor(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', input_shape=input_shape),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])


def make_generator(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*256, use_bias=False,
                              input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(
            1, 1), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(
            2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(
            2, 2), padding='same', use_bias=False, activation='tanh')
    ])


def gen_noise(batch_size, z_dim):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


def d_loss_fn(real_logits, fake_logits):
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)


def g_loss_fn(fake_logits):
    return -tf.reduce_mean(fake_logits)


@tf.function
def train_step(gen, disc, batch_size, images, z_dim):
    noise = gen_noise(batch_size, z_dim)
    noise += tf.random.normal(shape=tf.shape(noise), mean=0.0, stddev=0.1)
    with tf.GradientTape(persistent=True) as tape:
        generated_images = gen(noise)
        disc_fake_pred = disc(generated_images)
        disc_real_pred = disc(images)

        disc_loss = d_loss_fn(disc_real_pred, disc_fake_pred)
        gen_loss = g_loss_fn(disc_fake_pred)

    disc_gradients = tape.gradient(
        disc_loss, disc.trainable_variables)
    disc_optimizer.apply_gradients(
        zip(disc_gradients, disc.trainable_variables))

    gen_gradients = tape.gradient(
        gen_loss, gen.trainable_variables)
    gen_optimizer.apply_gradients(
        zip(gen_gradients, gen.trainable_variables))
    return gen_loss, disc_loss


@tf.function
def train_discriminator(gen, disc, batch_size, images, z_dim):
    noise = gen_noise(batch_size, z_dim)

    with tf.GradientTape() as disc_tape:
        generated_imgs = gen(noise, training=True)
        generated_output = disc(generated_imgs, training=True)
        real_output = disc(images, training=True)

        disc_loss = d_loss_fn(real_output, generated_output)

    grad_disc = disc_tape.gradient(
        disc_loss, disc.trainable_variables)
    disc_optimizer.apply_gradients(
        zip(grad_disc, disc.trainable_variables))

    for param in disc.trainable_variables:
        # Except gamma and beta in Batch Normalization
        if param.name.split('/')[-1].find('gamma') == -1 and param.name.split('/')[-1].find('beta') == -1:
            param.assign(tf.clip_by_value(param, -0.01, 0.01))
    return disc_loss


@tf.function
def train_generator(gen, disc, batch_size, images, z_dim):
    noise = gen_noise(batch_size, z_dim)

    with tf.GradientTape() as gen_tape:
        generated_imgs = gen(noise, training=True)
        generated_output = disc(generated_imgs, training=True)
        real_output = disc(images, training=True)

        gen_loss = g_loss_fn(generated_output)

    grad_gen = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gen_optimizer.apply_gradients(zip(grad_gen, gen.trainable_variables))

    return gen_loss


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
    plt.savefig('w_gan/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close("all")
    # plt.show(block='False')


def train_model(train, val, epochs=ITERATION, batch_size=BATCH_SIZE):
    gen = make_generator((Z_DIM,))
    gen.summary()
    disc = make_discriminaor(IMAGE_SHAPE)
    disc.summary()
    n_critic = N_CRITIC

    try:
        gen.load_weights('w_gan_weights/generator_w_gan.h5')
        disc.load_weights('w_gan_weights/discriminator_w_gan.h5')
    except FileNotFoundError:
        pass

    train = train.batch(batch_size=batch_size)
    val = val.batch(batch_size=batch_size)

    min_loss_gen = np.inf
    for epoch in range(epochs+1):
        total_gen_loss = 0
        total_disc_loss = 0
        for ind, images in enumerate(train):
            curr_batch_size = images.shape[0]
            d_loss = train_discriminator(
                gen, disc, curr_batch_size, images, Z_DIM)
            total_disc_loss += d_loss

            if disc_optimizer.iterations.numpy() % n_critic == 0:
                g_loss = train_generator(
                    gen, disc, curr_batch_size, images, Z_DIM)
                total_gen_loss += g_loss

        template = '\r[{}/{}] D_loss={:.5f} G_loss={:.5f} '
        print(template.format(epoch, ITERATION, total_disc_loss/BATCH_SIZE,
                              total_gen_loss/BATCH_SIZE), end=' ')
        sys.stdout.flush()

        if epoch % EVERY_STEP == 0:
            noise = gen_noise(16, Z_DIM)
            generate_and_save_images(gen,
                                     epoch + 1,
                                     noise)

            loss = g_loss
            if loss <= min_loss_gen:
                disc.save_weights('w_gan_weights/discriminator_w_gan.h5')
                gen.save_weights('w_gan_weights/generator_w_gan.h5')
                loss = min_loss_gen


if __name__ == '__main__':
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.mnist.load_data()
    train_data = (train_data.reshape(-1, 28, 28, 1) - 127.5)/127.5
    test_data = (test_data.reshape(-1, 28, 28, 1) - 127.5)/127.5
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_optimizer = tf.keras.optimizers.RMSprop(G_LR/4)
    disc_optimizer = tf.keras.optimizers.RMSprop(D_LR/4)

    train_data=tf.data.Dataset.from_tensor_slices(
        train_data).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)
    test_data=tf.data.Dataset.from_tensor_slices(
        test_data).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)

    train_model(train_data, test_data)
