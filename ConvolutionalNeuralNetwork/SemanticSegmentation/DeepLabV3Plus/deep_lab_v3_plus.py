
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_data_dir = '/home/shuvrajeet/datasets/instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing/'
IMAGE_SIZE = 128
BATCH_SIZE = 8
NUM_CLASSES = 20

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_deepLabV3Plus_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]
colormap = loadmat(base_data_dir + "human_colormap.mat")["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


""" Atrous Spatial Pyramid Pooling """


def ASPP(inputs):
    shape = inputs.shape

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(
        shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = tf.keras.layers.Conv2D(filters=32, kernel_size=1,
                                    padding='same', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = tf.keras.layers.Activation('relu', name=f'relu_1')(y_pool)
    y_pool = tf.keras.layers.UpSampling2D((shape[1], shape[2]),
                                          interpolation="bilinear")(y_pool)

    y_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, dilation_rate=1,
                                 padding='same', use_bias=False)(inputs)
    y_1 = tf.keras.layers.BatchNormalization()(y_1)
    y_1 = tf.keras.layers.Activation('relu')(y_1)

    y_6 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=6,
                                 padding='same', use_bias=False)(inputs)
    y_6 = tf.keras.layers.BatchNormalization()(y_6)
    y_6 = tf.keras.layers.Activation('relu')(y_6)

    y_12 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=12,
                                  padding='same', use_bias=False)(inputs)
    y_12 = tf.keras.layers.BatchNormalization()(y_12)
    y_12 = tf.keras.layers.Activation('relu')(y_12)

    y_18 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, dilation_rate=18,
                                  padding='same', use_bias=False)(inputs)
    y_18 = tf.keras.layers.BatchNormalization()(y_18)
    y_18 = tf.keras.layers.Activation('relu')(y_18)

    y = tf.keras.layers.Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = tf.keras.layers.Conv2D(filters=32, kernel_size=1, dilation_rate=1,
                               padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    return y


def DeepLabV3Plus(shape):
    """ Inputs """
    inputs = tf.keras.layers.Input(shape)

    """ Pre-trained ResNet50 """
    base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                include_top=False, input_tensor=inputs)

    """ Pre-trained ResNet50 Output """
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    """ Get low-level features """
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = tf.keras.layers.Conv2D(filters=32, kernel_size=1,
                                 padding='same', use_bias=False)(x_b)
    x_b = tf.keras.layers.BatchNormalization()(x_b)
    x_b = tf.keras.layers.Activation('relu')(x_b)

    x = tf.keras.layers.Concatenate()([x_a, x_b])

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same',
                               activation='relu', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same',
                               activation='relu', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(x)

    """ Outputs """
    x = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), name='output_layer')(x)
    x = tf.keras.layers.Activation('softmax')(x)

    """ Model """
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        # image.set_shape([None, None, 1])
        image = tf.image.resize(
            images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = tf.squeeze(image)
        image = tf.cast(image, dtype=tf.uint8)
        image = tf.one_hot(image, axis=-1, depth=NUM_CLASSES)
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image


class Dataset:
    def __init__(self) -> None:
        self.train_images = sorted(
            glob(os.path.join(base_data_dir+'/Training/', "Images/*")))
        self.train_mask = sorted(
            glob(os.path.join(base_data_dir+'/Training/', "Category_ids/*")))
        self.validation_images = sorted(
            glob(os.path.join(base_data_dir+'/Validation/', "Images/*")))
        self.validation_mask = sorted(
            glob(os.path.join(base_data_dir+'/Validation/', "Category_ids/*")))

    def load_data(self, image_list, mask_list):
        image = read_image(image_list)
        mask = read_image(mask_list, mask=True)
        return image, mask

    def data_generator(self, image_list, mask_list):
        dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(
            self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        return dataset


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(
                tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(
            prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )


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


if __name__ == "__main__":
    input_shape = (128, 128, 3)
    model = DeepLabV3Plus(input_shape)
    model.summary()
    tf.keras.utils.plot_model(model, to_file=DeepLabV3Plus.__name__+'.png',
                              show_shapes=True, show_layer_names=True, show_layer_activations=True)
    input('>')
    dataset = Dataset()
    train_dataset = dataset.data_generator(
        dataset.train_images, dataset.train_mask)
    val_dataset = dataset.data_generator(
        dataset.validation_images, dataset.validation_mask)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=total_loss,
        metrics=["accuracy", iou_coef, dice_coef, f1]
    )

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=25,
                        callbacks=callbacks)

    plt.plot(history.history["loss"], label='Training Loss')
    plt.plot(history.history["val_loss"], label='Validation Loss')
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["accuracy"], label='Training accuracy')
    plt.plot(history.history["val_accuracy"], label='Validation accuracy')
    plt.title("Training Accuracy")
    plt.legend()
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()

    plot_predictions(dataset.train_images[:4], colormap, model=model)
