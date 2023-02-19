import tensorflow as tf
import numpy as np
import os
import keras_cv
from luketils import visualization

from params import *


def visualize_dataset(dataset, bounding_box_format):
    example = next(iter(dataset))
    images, boxes = example["images"], example["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=boxes,
        scale=4,
        rows=3,
        cols=3,
        show=True,
        thickness=4,
        font_scale=1,
        class_mapping=class_mapping,
    )


if __name__ == '__main__':
    train_ds, ds_info = keras_cv.datasets.pascal_voc.load(
        split='train', bounding_box_format='xywh', batch_size=8
    )
    print(ds_info)

    dataset, dataset_info = keras_cv.datasets.pascal_voc.load(
        split="train", bounding_box_format="xywh", batch_size=BATCH_SIZE
    )

    class_ids = [
        "Aeroplane",
        "Bicycle",
        "Bird",
        "Boat",
        "Bottle",
        "Bus",
        "Car",
        "Cat",
        "Chair",
        "Cow",
        "Dining Table",
        "Dog",
        "Horse",
        "Motorbike",
        "Person",
        "Potted Plant",
        "Sheep",
        "Sofa",
        "Train",
        "Tvmonitor",
        "Total",
    ]
    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    visualize_dataset(dataset, bounding_box_format="xywh")
