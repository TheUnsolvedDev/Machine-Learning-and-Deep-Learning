import tensorflow as tf

from params import *


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, norm_layer, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

    def build_conv_block(self, dim, norm_layer, use_bias):
        conv_block = []
        p = 'same'
        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias),
                       norm_layer(),
                       tf.keras.layers.Activation('relu')]

        conv_block += [tf.keras.layers.Conv2D(dim, (3, 3), padding=p, use_bias=use_bias),
                       norm_layer()]
        return tf.keras.Sequential(conv_block)

    def call(self, x, training=True):
        out = x + self.conv_block(x, training=training)
        return out


class Generator(tf.keras.Model):
    def __init__(self,
                 channels=3,
                 ngf=32,
                 norm_layer=tf.keras.layers.BatchNormalization,
                 use_bias=False,
                 n_blocks=3):
        super(Generator, self).__init__()
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        model = [tf.keras.layers.Conv2D(ngf, (7, 7), padding='same', use_bias=use_bias),
                 norm_layer(),
                 tf.keras.layers.Activation('relu')]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [tf.keras.layers.Conv2D(ngf * mult * 2, (4, 4), strides=(2, 2), padding='same'),
                      norm_layer(),
                      tf.keras.layers.Activation('relu')]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [tf.keras.layers.Conv2DTranspose(int(ngf * mult / 2),
                                                      (4, 4), strides=(2, 2),
                                                      padding='same'),
                      norm_layer(),
                      tf.keras.layers.Activation('relu')]
        model += [tf.keras.layers.Conv2D(channels, (7, 7), padding='same'),
                  tf.keras.layers.Activation('tanh')]
        self.model = tf.keras.Sequential(model)

    def call(self, inputs, training=True):
        inputs, domains = inputs[0], inputs[1]
        domains = tf.reshape(domains, (-1, 1, 1, domains.shape[1]))
        domains = tf.tile(domains, tf.constant(
            [1, inputs.shape[1], inputs.shape[2], 1]))
        domains = tf.cast(domains, tf.float32)
        x = self.concat([inputs, domains])
        return self.model(x, training=training)

    def summary(self):
        x1 = tf.keras.layers.Input(shape=(IMAGE_SHAPE))
        x2 = tf.keras.layers.Input(shape=(5,))
        self.ker_model = tf.keras.Model(
            inputs=[x1, x2], outputs=self.call([x1, x2]))
        return self.ker_model.summary(expand_nested=True)

    def show_model(self):
        return self.ker_model


class Discriminator(tf.keras.Model):
    def __init__(self, img_shape, nd, ndf=32, n_layers=3, use_bias=False):
        super(Discriminator, self).__init__()
        h, w = img_shape[0:2]
        kw = 4
        model = [tf.keras.layers.Conv2D(ndf, (kw, kw), strides=(2, 2), padding='same'),
                 tf.keras.layers.LeakyReLU(0.01)]
        nf_mult = 1
        for n in range(1, n_layers + 1):
            nf_mult = 2 ** n
            model += [
                tf.keras.layers.Conv2D(
                    ndf * nf_mult, (kw, kw), strides=(2, 2), padding='same'),
                tf.keras.layers.LeakyReLU(0.01)
            ]
        self.model = tf.keras.Sequential(model)
        self.src = tf.keras.layers.Conv2D(1, (3, 3), strides=(
            1, 1), padding='same', use_bias=use_bias)
        self.cls = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    nd, (h // 64, w // 64), strides=(1, 1), padding='valid', use_bias=use_bias),
                tf.keras.layers.GlobalAveragePooling2D()
            ])

    def call(self, inputs, training=True):
        x = self.model(inputs, training=training)
        return self.src(x), tf.squeeze(self.cls(x))

    def summary(self):
        x = tf.keras.layers.Input(shape=(IMAGE_SHAPE))
        self.ker_model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return self.ker_model.summary(expand_nested=True)

    def show_model(self):
        return self.ker_model


if __name__ == '__main__':
    generator = Generator()
    generator.summary()
    discriminator = Discriminator(IMAGE_SHAPE, NUM_CLASS)
    discriminator.summary()
