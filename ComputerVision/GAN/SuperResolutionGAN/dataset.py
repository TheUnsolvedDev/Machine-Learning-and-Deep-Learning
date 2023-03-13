import tensorflow as tf
import cv2

IMG_H_SIZE = 128
IMG_W_SIZE = 128


def save_imgs(epoch, generator, lr_images, hr_images):
    gen_imgs = generator(lr_images, training=False)

    for i in range(gen_imgs.shape[0]):
        cv2.imwrite('./images/sr_{}_{}.png'.format(epoch, i),
                    (gen_imgs[i].numpy()[..., ::-1] + 1) * 127.5)
        cv2.imwrite('./images/hr_{}_{}.png'.format(epoch, i),
                    (hr_images[i].numpy()[..., ::-1] + 1) * 127.5)
        cv2.imwrite('./images/lr_{}_{}.png'.format(epoch, i),
                    (lr_images[i].numpy()[..., ::-1] + 1) * 127.5)

        resized = cv2.resize(
            (lr_images[i].numpy()[..., ::-1] + 1) * 127.5, (IMG_W_SIZE, IMG_H_SIZE))
        cv2.imwrite('./images/re_{}_{}.png'.format(epoch, i), resized)


def preprocess_data(file_path, ratio=4):
    image = process_path(file_path)
    resized_image = resize(image, (IMG_H_SIZE//ratio, IMG_W_SIZE//ratio))
    image = resize(image, (IMG_H_SIZE, IMG_W_SIZE))
    image = normalize(image)
    resized_image = normalize(resized_image)

    return resized_image, image


def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    image = (image / 127.5) - 1
    return image


def resize(image, size):
    h, w = size
    image = tf.image.resize(
        image, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    return img


def create_dataset():
    image_ds = tf.data.Dataset.list_files(
        '/home/shuvrajeet/datasets/data512x512/*', shuffle=True)
    train_dataset = image_ds.map(lambda x: preprocess_data(
        x), num_parallel_calls=tf.data.AUTOTUNE).batch(8).prefetch(tf.data.AUTOTUNE).cache()
    return train_dataset
