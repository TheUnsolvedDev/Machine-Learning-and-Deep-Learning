import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2
from silence_tensorflow import silence_tensorflow
import time
import gc

from params import *
from model import *
from dataset import *


silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def gradient_penalty_loss(averaged_output, x_hat):
    gradients = tf.gradients(averaged_output, x_hat)[0]
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)

    gradient_penalty = tf.square(gradients_l2_norm - 1)

    return tf.reduce_mean(gradient_penalty)


def discriminator_loss(real_output, fake_output, averaged_output, interpolated_img, lamb_gp=10):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    gp_loss = gradient_penalty_loss(averaged_output, interpolated_img)
    total_loss = real_loss + fake_loss + gp_loss * lamb_gp
    return total_loss


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def reconstrution_loss(loss_object, real_image, recon_image, lamb_rec=10):
    return loss_object(real_image, recon_image) * lamb_rec


def domain_classification_loss(loss_object, category, output, lamb_cls=1):
    return loss_object(category, output) * lamb_cls


def random_weighted_average(inputs):
    alpha = tf.random.uniform((inputs[0].shape[0], 1, 1, 1), dtype=tf.float32)
    # inputs = tf.cast(inputs, dtype=tf.float32)
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


@tf.function
def train_generator(generator, discriminator, images, ori_labels, tar_labels):
    with tf.GradientTape(persistent=True) as tape:
        fake_images = generator([images, tar_labels])
        recon_images = generator([fake_images, ori_labels])
        fake_output, fake_class = discriminator(fake_images)
        gen_loss = generator_loss(fake_output)
        fake_class_loss = domain_classification_loss(
            bce, tar_labels, fake_class)
        recon_loss = reconstrution_loss(l1, images, recon_images)
        total_gen_loss = gen_loss + fake_class_loss + recon_loss

    grad_gen = tape.gradient(total_gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

    return fake_images, gen_loss, fake_class_loss, recon_loss


@tf.function
def train_discriminator(generator, discriminator, images, ori_labels, tar_labels):
    with tf.GradientTape(persistent=True) as tape:
        real_output, real_class = discriminator(images)
        fake_images = generator([images, tar_labels])
        fake_output, fake_class = discriminator(fake_images)
        interpolated_img = random_weighted_average([images, fake_images])
        averaged_output, _ = discriminator(interpolated_img)

        disc_loss = discriminator_loss(
            real_output, fake_output, averaged_output, interpolated_img)
        real_class_loss = domain_classification_loss(
            bce, ori_labels, real_class)
        total_disc_loss = disc_loss + real_class_loss

    grad_disc = tape.gradient(
        total_disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(
        zip(grad_disc, discriminator.trainable_variables))

    return real_class_loss, disc_loss


def save_imgs(epoch, generator, real_x):
    tags = np.array([[0, 1, 0, 1, 0]
                    for i in range(len(real_x))], dtype=np.uint8)
    gene_imgs = generator([real_x, tags], training=False)

    gene_imgs = ((gene_imgs.numpy() + 1) * 127.5).astype(np.uint8)
    real_x = ((real_x.numpy() + 1) * 127.5).astype(np.uint8)

    fig = plt.figure(figsize=(4, 8))

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


def train_model(train, epochs=ITERATION, batch_size=BATCH_SIZE):
    generator = Generator()
    generator.build([(None, *IMAGE_SHAPE), (None, 5)])
    generator.summary()
    tf.keras.utils.plot_model(
        generator.show_model(), to_file=Generator.__name__+'.png', show_shapes=True, expand_nested=True)
    discriminator = Discriminator(IMAGE_SHAPE, NUM_CLASS)
    discriminator.build([None, *IMAGE_SHAPE])
    discriminator.summary()
    tf.keras.utils.plot_model(
        discriminator.show_model(), to_file=Discriminator.__name__+'.png', show_shapes=True, expand_nested=True)

    try:
        generator.load_weights('trial_weights/generator.h5')
        discriminator.load_weights('trial_weights/discriminator.h5')
    except FileNotFoundError:
        pass

    print('Training...')
    for epoch in range(56, epochs+1):
        start = time.time()
        gc.collect()
        total_real_cls_loss = 0
        total_disc_loss = 0

        total_gen_loss = 0
        total_fake_cls_loss = 0
        total_recon_loss = 0

        for images, ori_labels, tar_labels in tqdm.tqdm(train):
            real_cls_loss, disc_loss = train_discriminator(
                generator, discriminator, images, ori_labels, tar_labels)
            total_real_cls_loss += real_cls_loss
            total_disc_loss += disc_loss

            if epoch % N_CRITIC == 0:
                fake_images, gen_loss, fake_cls_loss, recon_loss = train_generator(
                    generator, discriminator, images, ori_labels, tar_labels)
                total_gen_loss += gen_loss
                total_fake_cls_loss += fake_cls_loss
                total_recon_loss += recon_loss

        log = 'Time for epoch {}/{} is {} sec : - disc_loss = {}, real_cls_loss = {}'.format(
            epoch + 1, epochs, time.time() - start, total_disc_loss/batch_size, total_real_cls_loss/batch_size)
        log += 'gen_loss = {}, fake_cls_loss = {}, recon_loss = {}'.format(
            total_gen_loss/batch_size, total_fake_cls_loss/batch_size, total_recon_loss/batch_size)

        if epoch % EVERY_STEP == 0:
            for idx, (orig_img, fake_img) in enumerate(zip(images, fake_images)):
                tmp1 = np.asarray((orig_img.numpy() + 1)
                                  * 127.5, dtype=np.uint8)
                tmp2 = np.asarray((fake_img.numpy() + 1)
                                  * 127.5, dtype=np.uint8)
                img = np.hstack([tmp1, tmp2])
                cv2.imwrite('trial_images/Step{}_Batch{}_Ori{}_Tar{}.png'.format(str(epoch).zfill(
                    6), str(idx).zfill(3), str(ori_labels[idx].numpy()), str(tar_labels[idx].numpy()).replace(' ', '')), img[..., ::-1])
            generator.save_weights('trial_weights/generator.h5')
            discriminator.save_weights('trial_weights/discriminator.h5')
        print(log)


if __name__ == '__main__':
    data = create_dataset()
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    l1 = tf.keras.losses.MeanAbsoluteError()
    gen_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
    train_model(data)
