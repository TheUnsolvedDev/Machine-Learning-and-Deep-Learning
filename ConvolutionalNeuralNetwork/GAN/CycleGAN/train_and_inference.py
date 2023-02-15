import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from silence_tensorflow import silence_tensorflow

from params import *
from model import *
from dataset import *

silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def save_imgs(epoch, generator, real_x):
    gene_imgs = generator(real_x, training=False)

    gene_imgs = ((gene_imgs.numpy() + 1) * 127.5).astype(np.uint8)
    real_x = ((real_x.numpy() + 1) * 127.5).astype(np.uint8)

    fig = plt.figure(figsize=(8, 12))
    tmp = 0
    for i in range(0, 4):
        plt.subplot(4, 2, i + 1 + tmp)
        plt.imshow(real_x[i])
        plt.axis('off')
        plt.subplot(4, 2, i + 2 + tmp)
        plt.imshow(gene_imgs[i])
        plt.axis('off')
        tmp += 1

    fig.savefig("trial_images/result_{}.png".format(str(epoch).zfill(5)))


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def cycle_loss(loss_object, real_image, cycled_image, _lambda=10):
    return loss_object(real_image, cycled_image) * _lambda


def identity_loss(loss_object, real_image, same_image, _lambda=10):
    return loss_object(real_image, same_image) * 0.5 * _lambda


@tf.function
def train_step(gene_G, gene_F, disc_X, disc_Y, real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = gene_G(real_x)
        rec_x = gene_F(fake_y)

        fake_x = gene_F(real_y)
        rec_y = gene_G(fake_x)

        same_x = gene_G(real_x)
        same_y = gene_F(real_y)

        disc_real_x = disc_X(real_x)
        disc_real_y = disc_Y(real_y)

        disc_fake_x = disc_X(fake_x)
        disc_fake_y = disc_Y(fake_y)

        # Loss Func.
        disc_x_loss = discriminator_loss(
            cross_entropy, disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(
            cross_entropy, disc_real_y, disc_fake_y)

        gene_g_loss = generator_loss(cross_entropy, disc_fake_y)
        gene_f_loss = generator_loss(cross_entropy, disc_fake_x)

        cycle_x_loss = cycle_loss(mae, real_x, rec_x)
        cycle_y_loss = cycle_loss(mae, real_y, rec_y,)

        total_cycle_loss = cycle_x_loss + cycle_y_loss
        total_gene_g_loss = gene_g_loss + total_cycle_loss + \
            identity_loss(mae, real_y, same_y)
        total_gene_f_loss = gene_f_loss + total_cycle_loss + \
            identity_loss(mae, real_x, same_x)

    grad_gene_G = tape.gradient(total_gene_g_loss, gene_G.trainable_variables)
    grad_gene_F = tape.gradient(total_gene_f_loss, gene_F.trainable_variables)

    grad_disc_X = tape.gradient(disc_x_loss, disc_X.trainable_variables)
    grad_disc_Y = tape.gradient(disc_y_loss, disc_Y.trainable_variables)

    gene_g_optimizer.apply_gradients(
        zip(grad_gene_G, gene_G.trainable_variables))
    gene_f_optimizer.apply_gradients(
        zip(grad_gene_F, gene_F.trainable_variables))

    disc_x_optimizer.apply_gradients(
        zip(grad_disc_X, disc_X.trainable_variables))
    disc_y_optimizer.apply_gradients(
        zip(grad_disc_Y, disc_Y.trainable_variables))

    return total_gene_g_loss, total_gene_f_loss, disc_x_loss, disc_y_loss


def train_model(train, epochs=ITERATION, batch_size=BATCH_SIZE):
    gene_G = Generator()
    gene_F = Generator()
    disc_X = Discriminator()
    disc_Y = Discriminator()

    gene_G.build([None, *IMAGE_SHAPE])
    gene_F.build([None, *IMAGE_SHAPE])
    gene_F.summary()
    gene_G.summary()

    disc_X.build([None, *IMAGE_SHAPE])
    disc_Y.build([None, *IMAGE_SHAPE])
    disc_X.summary()
    disc_Y.summary()

    try:
        gene_G.load_weights('trial_weights/gene_G.h5')
        gene_F.load_weights('trial_weights/gene_F.h5')
    except FileNotFoundError:
        pass

    print('Training...')
    for epoch in range(493, epochs+1):
        total_gene_g_loss = 0
        total_gene_f_loss = 0
        total_disc_x_loss = 0
        total_disc_y_loss = 0

        for ind, images in enumerate(train):
            real_x, real_y = images
            gene_g_loss, gene_f_loss, disc_x_loss, disc_y_loss = train_step(
                gene_G, gene_F, disc_X, disc_Y, real_x, real_y)
            total_gene_g_loss += gene_g_loss
            total_gene_f_loss += gene_f_loss
            total_disc_x_loss += disc_x_loss
            total_disc_y_loss += disc_y_loss

        template = '\r[{}/{}] G gene_loss = {}, F gene_loss = {}, D x_loss = {}, D y_loss = {}'
        print(template.format(epoch, ITERATION, total_gene_g_loss / batch_size,
                              total_gene_f_loss / batch_size,
                              total_disc_x_loss / batch_size,
                              total_disc_y_loss / batch_size), end=' ')
        sys.stdout.flush()

        if epoch % EVERY_STEP == 0:
            save_imgs(epoch + 1, gene_F, real_y)
            gene_G.save_weights('trial_weights/gene_G.h5')
            gene_F.save_weights('trial_weights/gene_F.h5')


def test_model(data):
    gene_F = Generator()
    gene_F.build([None, *IMAGE_SHAPE])
    gene_F.summary()
    gene_F.load_weights('trial_weights/gene_F.h5')

    for image in data:
        real_x, real_y = image
        break

    her_image = cv2.imread('/home/shuvrajeet/Downloads/try.jpg')
    her_image = cv2.resize(cv2.cvtColor(
        her_image, cv2.COLOR_BGR2RGB), (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    her_image = normalize_img(her_image, dtype=tf.float32)

    her_image = tf.expand_dims(her_image, axis=0)
    her_gen_image = gene_F(her_image, training=False)

    gene_imgs = gene_F(real_y, training=False)
    gene_imgs = ((gene_imgs.numpy() + 1) * 127.5).astype(np.uint8)
    real_y = ((real_y.numpy() + 1) * 127.5).astype(np.uint8)

    her_image = ((her_image.numpy() + 1) * 127.5).astype(np.uint8)
    her_gen_image = ((her_gen_image.numpy() + 1) * 127.5).astype(np.uint8)

    for i in range(0, len(her_gen_image)):
        image = np.hstack([her_image[i], her_gen_image[i]])
        plt.imshow(image)
        plt.savefig("trial_images/final_result_{}.png".format(str(i).zfill(5)))
        plt.show()


if __name__ == '__main__':
    gene_g_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    gene_f_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    disc_x_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    disc_y_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mae = tf.keras.losses.MeanAbsoluteError()
    data, test_data = dataset()

    train_model(data)
    # test_model(test_data)
