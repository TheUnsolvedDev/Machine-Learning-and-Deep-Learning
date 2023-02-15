import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

ROOT_DIR_IMAGES = '/home/shuvrajeet/datasets/leftImg8bit_trainvaltest/leftImg8bit/'
ROOT_DIR_LABELS = '/home/shuvrajeet/datasets/gtFine_trainvaltest/gtFine/'
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 34
BATCH_SIZE = 16
BUFFER_SIZE = 500

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_gated_scnn.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def read_files(type_of='train'):
    images = []
    for city in tqdm.tqdm(os.listdir(ROOT_DIR_IMAGES + type_of + '/')):
        for image in os.listdir(ROOT_DIR_IMAGES + type_of + '/'+city):
            if image.endswith('.png'):
                images.append(ROOT_DIR_IMAGES + type_of +
                              '/' + city + '/'+image)
    labels = []
    for city in tqdm.tqdm(os.listdir(ROOT_DIR_LABELS + type_of + '/')):
        for image in os.listdir(ROOT_DIR_LABELS + type_of + '/'+city):
            if image.endswith('gtFine_labelIds.png') and not image.startswith('.'):
                labels.append(ROOT_DIR_LABELS + type_of +
                              '/' + city + '/'+image)
    return sorted(images), sorted(labels)


class Dataset:
    def __init__(self):
        train_data, train_labels = read_files('train')
        validation_data, validation_labels = read_files('val')
        self.train = tf.data.Dataset.from_tensor_slices(
            (train_data, train_labels))
        self.validation = tf.data.Dataset.from_tensor_slices(
            (validation_data, validation_labels)).shuffle(BUFFER_SIZE)

    @tf.function
    def read_image(self, path1, path2):
        img = tf.io.read_file(path1)
        img = tf.io.decode_png(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)

        lbl = tf.io.read_file(path2)
        lbl = tf.io.decode_png(lbl, channels=1)
        lbl = tf.image.resize(lbl, IMAGE_SIZE)
        lbl = tf.cast(lbl, dtype=tf.uint8)
        lbl = tf.squeeze(lbl)
        lbl = tf.one_hot(lbl, depth=NUM_CLASSES, axis=-1)
        return tf.cast(img, dtype=tf.uint8), lbl

    def make_dataset(self):
        self.data_train = self.train.prefetch(tf.data.AUTOTUNE).map(
            self.read_image, num_parallel_calls=BATCH_SIZE).shuffle(BUFFER_SIZE)
        self.data_validation = self.validation.prefetch(tf.data.AUTOTUNE).map(
            self.read_image, num_parallel_calls=BATCH_SIZE).shuffle(BUFFER_SIZE)
        return self.data_train.batch(BATCH_SIZE), self.data_validation.batch(BATCH_SIZE)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.keras.backend.sum(
        y_true, [1, 2, 3])+tf.keras.backend.sum(y_pred, [1, 2, 3])-intersection
    iou = tf.keras.backend.mean(
        (intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.keras.backend.sum(
        y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    dice = tf.keras.backend.mean(
        (2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = tf.keras.backend.sum(tf.keras.backend.round(
            tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        Positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))

        recall = TP / (Positives+tf.keras.backend.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = tf.keras.backend.sum(tf.keras.backend.round(
            tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = tf.keras.backend.sum(
            tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+tf.keras.backend.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))


def total_loss(y_true, y_pred):
    loss1 = DiceLoss()
    loss2 = tf.keras.losses.CategoricalCrossentropy()
    return loss1(y_true, y_pred) + loss2(y_true, y_pred)


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tf.keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size,
                      padding=padding, strides=strides)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(
        1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tf.keras.layers.Add()([conv, shortcut])
    return output


def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation("relu")(x)
    return x


def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size,
                     padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size,
                     padding=padding, strides=1)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(
        1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tf.keras.layers.Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c


def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = tf.keras.layers.Input((*IMAGE_SIZE, 3))
    x = tf.keras.layers.Lambda(lambda i: i/255)(inputs)
    # Encoder
    e0 = x
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    # Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    # Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = tf.keras.layers.Conv2D(
        NUM_CLASSES, (1, 1), padding="same", activation="softmax")(d4)
    model = tf.keras.models.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    dataset = Dataset()
    train, validation = dataset.make_dataset()
    model = ResUNet()
    model.compile(
        optimizer='adam', loss=total_loss, metrics=['accuracy', iou_coef, dice_coef, f1])
    tf.keras.utils.plot_model(
        model, to_file=ResUNet.__name__+'.png', show_shapes=True, show_layer_activations=True)
    history = model.fit(train, epochs=50, validation_data=validation,
                        callbacks=callbacks)
