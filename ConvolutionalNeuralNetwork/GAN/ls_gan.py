import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from silence_tensorflow import silence_tensorflow
from params import *

if not os.path.exists('trial_images/'+__file__.split('/')[-1].replace('.py', '')+'/'):
    os.system("mkdir -p "+'trial_images/' +
              __file__.split('/')[-1].replace('.py', '')+'/')
    os.system("mkdir -p "+'trial_weights/' +
              __file__.split('/')[-1].replace('.py', '')+'/')

silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Generator(tf.keras.Model):
    def __init__(self, channels=1, method='transpose'):
        super(Generator, self).__init__()
        self.channels = channels
        self.method = method

        self.dense = tf.keras.layers.Dense(256 * 7 * 7, use_bias=False)

        self.reshape = tf.keras.layers.Reshape((7, 7, 256))

        if self.method == 'transpose':
            self.convT_1 = tf.keras.layers.Conv2DTranspose(
                128, (5, 5), padding='same', use_bias=False)
            self.convT_2 = tf.keras.layers.Conv2DTranspose(
                64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
            self.convT_3 = tf.keras.layers.Conv2DTranspose(self.channels, (5, 5), strides=(
                2, 2), padding='same', use_bias=False, activation='tanh')
        elif self.method == 'upsample':
            self.conv_1 = tf.keras.layers.Conv2D(
                128, (3, 3), padding='same', use_bias=False)
            self.upsample2d_1 = tf.keras.layers.UpSampling2D()
            self.conv_2 = tf.keras.layers.Conv2D(
                64, (3, 3), padding='same', use_bias=False)
            self.upsample2d_2 = tf.keras.layers.UpSampling2D()
            self.conv_3 = tf.keras.layers.Conv2D(
                self.channels, (3, 3), padding='same', use_bias=False, activation='tanh')

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.leakyrelu_1 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_2 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_3 = tf.keras.layers.LeakyReLU()

    def call(self, inputs, training=True):

        if self.method == 'transpose':
            x = self.dense(inputs)
            x = self.batch_norm_1(x, training)
            x = self.leakyrelu_1(x)

            x = self.reshape(x)

            x = self.convT_1(x)
            x = self.batch_norm_2(x, training)
            x = self.leakyrelu_2(x)

            x = self.convT_2(x)
            x = self.batch_norm_3(x, training)
            x = self.leakyrelu_3(x)

            return self.convT_3(x)

        elif self.method == 'upsample':
            x = self.dense(inputs)
            x = self.batch_norm_1(x, training)
            x = self.leakyrelu_1(x)

            x = self.reshape(x)

            x = self.conv_1(x)
            x = self.batch_norm_2(x, training)
            x = self.leakyrelu_2(x)

            x = self.upsample2d_1(x)
            x = self.conv_2(x)
            x = self.batch_norm_3(x, training)
            x = self.leakyrelu_3(x)

            x = self.upsample2d_2(x)
            return self.conv_3(x)

    def summary(self):
        x = tf.keras.layers.Input(shape=(Z_DIM,))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.out = tf.keras.layers.Dense(1)

        self.leakyrelu_1 = tf.keras.layers.LeakyReLU()
        self.leakyrelu_2 = tf.keras.layers.LeakyReLU()

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.3)

    def call(self, inputs, training=True):
        x = self.conv_1(inputs)
        x = self.leakyrelu_1(x)
        x = self.dropout_1(x, training)

        x = self.conv_2(x)
        x = self.leakyrelu_2(x)
        x = self.dropout_2(x, training)

        x = self.flatten(x)

        return self.out(x)

    def summary(self):
        x = tf.keras.layers.Input(shape=(IMAGE_SHAPE))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


def gen_noise(batch_size, z_dim):
    return tf.random.normal(shape=[batch_size, z_dim])


def d_loss_fn(real_logits, fake_logits):
    real_loss = criterion(tf.ones_like(real_logits), real_logits)
    fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def g_loss_fn(fake_logits):
    return criterion(tf.ones_like(fake_logits), fake_logits)


@tf.function
def train_step(gen, disc, batch_size, images, z_dim):
    noise = gen_noise(batch_size, z_dim)

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
    plt.savefig('trial_images/'+__file__.split('/')[-1].replace('.py', '')+'/' +
                'image_at_epoch_{:04d}.png'.format(epoch))
    plt.close("all")
    # plt.show(block='False')


def train_model(train, val, epochs=ITERATION, batch_size=BATCH_SIZE):
    gen = Generator()
    gen.build((None, Z_DIM))
    gen.summary()

    disc = Discriminator()
    disc.build((None, 28, 28, 1))
    disc.summary()

    try:
        gen.load_weights('trial_weights/'+__file__.split('/')
                         [-1].replace('.py', '')+'/' +
                         'generator_'+__file__.split('/')
                         [-1].replace('.py', '')+'.h5')
        disc.load_weights('trial_weights/'+__file__.split('/')
                          [-1].replace('.py', '')+'/' +
                          'discriminator_'+__file__.split('/')
                          [-1].replace('.py', '')+'.h5')
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
            g_loss, d_loss = train_step(
                gen, disc, curr_batch_size, images, Z_DIM)
            total_gen_loss += g_loss
            total_disc_loss += d_loss

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
                disc.save_weights('trial_weights/'+__file__.split('/')
                                  [-1].replace('.py', '')+'/' +
                                  'discriminator_'+__file__.split('/')
                                  [-1].replace('.py', '')+'.h5')
                gen.save_weights('trial_weights/'+__file__.split('/')
                                 [-1].replace('.py', '')+'/' +
                                 'generator_'+__file__.split('/')
                                 [-1].replace('.py', '')+'.h5')
                loss = min_loss_gen


def normalize(x):
    image = tf.cast(x, tf.float32)
    image = (image / 127.5) - 1
    return image


if __name__ == '__main__':
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.mnist.load_data()
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    criterion = tf.keras.losses.MeanSquaredError()

    gen_optimizer = tf.keras.optimizers.Adam(G_LR, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(D_LR, 0.5)

    train_data = tf.data.Dataset.from_tensor_slices(
        train_data).map(normalize).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)
    test_data = tf.data.Dataset.from_tensor_slices(
        test_data).map(normalize).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)

    train_model(train_data, test_data)
