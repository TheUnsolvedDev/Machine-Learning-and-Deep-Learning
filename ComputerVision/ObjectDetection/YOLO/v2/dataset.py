import numpy as np
import tensorflow as tf

from params import *
from utils import *


class SimpleBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, config, norm=None, shuffle=True):
        self.config = config
        self.config["BOX"] = int(len(self.config['ANCHORS'])/2)
        self.config["CLASS"] = len(self.config['LABELS'])
        self.images = images
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config['ANCHORS'])
        self.imageReader = ImageReader(
            config['IMAGE_H'], config['IMAGE_W'], norm=norm)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros(
            (r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1,
                           self.config['TRUE_BOX_BUFFER'], 4))
        y_batch = np.zeros(
            (r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))

        for train_instance in self.images[l_bound:r_bound]:
            img, all_objs = self.imageReader.fit(train_instance)
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x, center_y = rescale_centerxy(obj, self.config)

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx = self.config['LABELS'].index(obj['name'])
                        center_w, center_h = rescale_centerwh(obj, self.config)
                        box = [center_x, center_y, center_w, center_h]
                        best_anchor, max_iou = self.bestAnchorBoxFinder.find(
                            center_w, center_h)
                        y_batch[instance_count, grid_y,
                                grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y,
                                grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x,
                                best_anchor, 5+obj_indx] = 1

                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            x_batch[instance_count] = img
            instance_count += 1
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)


if __name__ == '__main__':
    _ANCHORS01 = np.array([0.08285376, 0.13705531,
                           0.20850361, 0.39420716,
                           0.80552421, 0.77665105,
                           0.42194719, 0.62385487])
    GRID_H,  GRID_W = 13, 13
    ANCHORS = _ANCHORS01
    ANCHORS[::2] = ANCHORS[::2]*GRID_W
    ANCHORS[1::2] = ANCHORS[1::2]*GRID_H

    IMAGE_H, IMAGE_W = IMG_SIZE[0], IMG_SIZE[1]
    BATCH_SIZE = 16
    TRUE_BOX_BUFFER = 50
    generator_config = {
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'GRID_H': GRID_H,
        'GRID_W': GRID_W,
        'LABELS': LABELS,
        'ANCHORS': ANCHORS,
        'BATCH_SIZE': BATCH_SIZE,
        'TRUE_BOX_BUFFER': TRUE_BOX_BUFFER,
    }
    train_image, seen_train_labels = parse_annotation(
        train_annot_folder, train_image_folder, labels=LABELS)
    train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                                 norm=normalize, shuffle=True)

    [x_batch, b_batch], y_batch = train_batch_generator.__getitem__(idx=3)
