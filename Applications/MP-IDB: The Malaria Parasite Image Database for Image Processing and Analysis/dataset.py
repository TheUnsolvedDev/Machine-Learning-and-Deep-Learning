from params import *
import tensorflow as tf
import numpy as np
from silence_tensorflow import silence_tensorflow
import os
from PIL import Image
import cv2
import tqdm
silence_tensorflow()


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, path='/home/shuvrajeet/datasets/archive/data/', shuffle=True):
        self.path = path
        self.shuffle = shuffle
        self.classes = []
        for class_ in os.listdir(self.path):
            self.classes.append(class_)

        self.image_paths = []

        for class_ in self.classes:
            for file in os.listdir(self.path+class_+'/img/'):
                self.image_paths.append(self.path+class_+'/img/'+file)

        self.indices = np.arange(len(self.image_paths))
        self.batch_size = BATCH_SIZE

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_image_paths = [self.image_paths[index] for index in indexes]
        data, labels = self.read_images(list_image_paths)
        return data, tf.one_hot(labels, depth=NUM_CLASSES)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def read_images(self, paths):
        data, labels = [], []
        for elem in paths:
            image = np.array(Image.open(elem), dtype=np.uint8)
            image = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))
            data.append(image)
            label = self.classes.index(elem.split('/')[-3])
            labels.append(label)
        return np.array(data, dtype=np.uint8), np.array(labels, dtype=np.uint8)


if __name__ == '__main__':
    dataset = Dataset()
    for _ in tqdm.tqdm(range(dataset.__len__())):
        dataset.__getitem__(_)
