import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from silence_tensorflow import silence_tensorflow
import tensorflow_datasets as tfds
from params import *

silence_tensorflow()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MaxoutDense(tf.keras.layers.Layer):
    def __init__(self, units, k, activation, drop_prob=0.5):
        self.dense = tf.keras.layers.Dense(units * k, activation=activation)
        self.dropout = tf.keras.layers.Dropout(drop_prob)
        self.reshape = tf.keras.layers.Reshape((-1, k, units))
        super(MaxoutDense, self).__init__()

    def call(self, inputs, training=True):
        x = self.dense(inputs)
        x = self.dropout(x, training)
        x = self.reshape(x)
        return tf.reduce_max(x, axis=1)


class Generator(tf.keras.Model):
    def __init__(self, num_classes, channels=1):
        super(Generator, self).__init__()
        self.channels = channels
        self.dense_z = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_z = tf.keras.layers.Dropout(0.5)
        self.dense_y = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_y = tf.keras.layers.Dropout(0.5)
        self.combined_dense = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_x = tf.keras.layers.Dropout(0.5)
        self.final_dense = tf.keras.layers.Dense(
            28 * 28 * self.channels, activation='tanh')
        self.reshape = tf.keras.layers.Reshape((28, 28, self.channels))

    def call(self, inputs, training=True):
        inputs, labels = inputs[0], inputs[1]
        z = self.dense_z(inputs)
        y = self.dense_y(labels)
        combined_x = self.combined_dense(tf.concat([z, y], axis=-1))
        x = self.final_dense(combined_x)
        return self.reshape(x)

    def summary(self):
        x = tf.keras.layers.Input(shape=(Z_DIM,))
        y = tf.keras.layers.Input(shape=(10,))
        model = tf.keras.Model(inputs=[x, y], outputs=self.call([x, y]))
        return model.summary()


class Discriminator(tf.keras.Model):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.maxout_z = MaxoutDense(240, k=5, activation='relu', drop_prob=0.5)
        self.maxout_y = MaxoutDense(50, k=5, activation='relu', drop_prob=0.5)
        self.maxout_x = MaxoutDense(240, k=4, activation='relu', drop_prob=0.5)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=True):
        inputs, labels = inputs[0], inputs[1]
        z = self.flatten(inputs)
        z = self.maxout_z(z, training)
        y = self.maxout_y(labels, training)
        x = self.maxout_x(tf.concat([z, y], axis=-1))
        return self.out(x)

    def summary(self):
        x = tf.keras.layers.Input(shape=IMAGE_SHAPE)
        y = tf.keras.layers.Input(shape=(10,))
        model = tf.keras.Model(inputs=[x, y], outputs=self.call([x, y]))
        return model.summary()


def gen_noise(batch_size, z_dim):
    return tf.random.normal(shape=[batch_size, z_dim])


def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(loss_object, fake_output):
    return loss_object(tf.ones_like(fake_output), fake_output)


def convert_onehot(label, num_classes):
    return tf.one_hot(label, num_classes)


fig = plt.figure(figsize=(4, 4))


def generate_and_save_images(model, epoch, test_input):
    labels = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 7
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 8
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 12
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 13
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 14
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 15
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 16
              ]
    labels = tf.convert_to_tensor(labels)
    predictions = model([test_input, labels], training=False)
    plt.ion()
    plt.clf()
    for i in range(predictions.shape[0]):
        prediction = np.array(predictions[i]).reshape(28, 28)
        plt.subplot(4, 4, i+1)
        plt.imshow(prediction * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.savefig('c_gan/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close("all")


@tf.function
def train_step(gen, disc, batch_size, images, labels, z_dim):
    noise = gen_noise(batch_size, z_dim)

    with tf.GradientTape(persistent=True) as tape:
        generated_images = gen([noise, labels])
        disc_fake_pred = disc([generated_images, labels])
        disc_real_pred = disc([images, labels])

        disc_loss = discriminator_loss(
            cross_entropy, disc_real_pred, disc_fake_pred)
        gen_loss = generator_loss(cross_entropy, disc_fake_pred)

    disc_gradients = tape.gradient(
        disc_loss, disc.trainable_variables)
    disc_optimizer.apply_gradients(
        zip(disc_gradients, disc.trainable_variables))

    gen_gradients = tape.gradient(
        gen_loss, gen.trainable_variables)
    gen_optimizer.apply_gradients(
        zip(gen_gradients, gen.trainable_variables))
    return gen_loss, disc_loss


def train_model(train,  epochs=ITERATION):
    gen = Generator(10)
    gen.build([(None, Z_DIM), (None, 10)])
    gen.summary()
    disc = Discriminator(10)
    disc.build([(None, *IMAGE_SHAPE), (None, 10)])
    disc.summary()

    try:
        gen.load_weights('c_gan_weights/generator_c_gan.h5')
        disc.load_weights('c_gan_weights/discriminator_c_gan.h5')
    except FileNotFoundError:
        pass

    min_loss_gen = np.inf
    for epoch in range(epochs+1):
        total_gen_loss = 0
        total_disc_loss = 0
        for images, labels in train:
            curr_batch_size = images.shape[0]
            g_loss, d_loss = train_step(
                gen, disc, curr_batch_size, images, labels, Z_DIM)
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
                disc.save_weights('c_gan_weights/discriminator_c_gan.h5')
                gen.save_weights('c_gan_weights/generator_c_gan.h5')
                loss = min_loss_gen


def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))
    return [X, labels], y


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def resize(image, height, width):
    image = tf.image.resize(
        image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def preprocess_image(data, img_shape, num_classes):
    image = data['image']
    image = resize(image, img_shape[0], img_shape[1])
    image = normalize(image)
    label = convert_onehot(data['label'], num_classes)
    return image, label


if __name__ == '__main__':
    data, info = tfds.load("mnist", with_info=True,
                           data_dir='/home/shuvrajeet/datasets/')
    train_data = data['train']

    gen_optimizer = tf.keras.optimizers.Adam(G_LR, 0.5)
    disc_optimizer = tf.keras.optimizers.Adam(D_LR, 0.5)

    train_dataset = train_data.map(lambda x: preprocess_image(
        x, IMAGE_SHAPE, 10)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_model(train_dataset)
