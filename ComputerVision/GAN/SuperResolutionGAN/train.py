from dataset import *
from model import *
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
import time

silence_tensorflow()


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def total_variation_loss(x):
    h, w = x.shape[1], x.shape[2]
    a = tf.math.square(x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :])
    b = tf.math.square(x[:, :h - 1, :w - 1, :] - x[:, :w - 1, 1:, :])
    return tf.math.reduce_sum(tf.math.pow(a + b, 1.25))


def content_loss(loss_object, hr_feat, sr_feat):
    total_loss = loss_object(hr_feat, sr_feat)
    return total_loss  # * 0.006


def train(train_dataset):
    generator = Generator()
    generator.build([None, 32, 32, 3])
    discriminator = Discriminator()
    discriminator.build([None, 128, 128, 3])

    gen_optimizer = tf.keras.optimizers.Adam(0.0005, 0.9)
    disc_optimizer = tf.keras.optimizers.Adam(0.0005, 0.9)

    content_layer = 'block5_conv4'  # SRGAN-VGG54
    extractor = ContentModel(content_layer)
    extractor.trainable = False

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()

    try:
        generator.load_weights('trial_weights/generator.h5')
        discriminator.load_weights('trial_weights/discriminator.h5')
    except FileNotFoundError:
        pass

    @tf.function
    def train_step(lr_images, hr_images):
        with tf.GradientTape(persistent=True) as tape:
            sr_images = generator(lr_images)  # sr -> super resolution

            real_output = discriminator(hr_images)
            fake_output = discriminator(sr_images)

            # adversarial loss
            gen_loss = generator_loss(cross_entropy, fake_output) * 1e-3
            disc_loss = discriminator_loss(
                cross_entropy, real_output, fake_output) * 1e-3

            # content loss
            hr_feat = extractor(hr_images)
            sr_feat = extractor(sr_images)
            cont_loss = content_loss(mse, hr_feat, sr_feat) * 0.006

            perc_loss = cont_loss + gen_loss

        grad_gen = tape.gradient(perc_loss, generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(
            zip(grad_gen, generator.trainable_variables))
        disc_optimizer.apply_gradients(
            zip(grad_disc, discriminator.trainable_variables))

        return gen_loss, disc_loss, cont_loss

    total_iter = 0
    epochs = 800
    batch_size = 4
    save_interval = 5
    for epoch in range(20, epochs + 1):
        start = time.time()
        total_gen_loss = 0
        total_disc_loss = 0
        total_cont_loss = 0

        for i, (lr_images, hr_images) in enumerate(train_dataset, 1):
            total_iter += 1
            gen_loss, disc_loss, cont_loss = train_step(lr_images, hr_images)

            if i % 100 == 0:
                print(
                    f'Batch:{i}({total_iter}) -> gen_loss: {gen_loss}, disc_loss: {disc_loss}, cont_loss: {cont_loss}')
                save_imgs(epoch, generator, lr_images, hr_images)

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            total_cont_loss += cont_loss

        print('Time for epoch {} is {} sec -> gen_loss: {}, disc_loss: {}, cont_loss: {}'.format(epoch,
                                                                                                 time.time() - start,
                                                                                                 total_gen_loss / i,
                                                                                                 total_disc_loss / i,
                                                                                                 total_cont_loss / i))
        if epoch % save_interval == 0:
            generator.save_weights('trial_weights/generator.h5')
            discriminator.save_weights('trial_weights/discriminator.h5')


if __name__ == '__main__':
    train_dataset = create_dataset()
    train(train_dataset)
