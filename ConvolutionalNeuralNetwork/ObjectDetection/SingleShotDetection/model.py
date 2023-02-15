import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from params import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def iou(box1, box2):
    box1 = tf.cast(box1, dtype=tf.float32)
    box2 = tf.cast(box2, dtype=tf.float32)

    x1 = tf.math.maximum(box1[:, None, 0], box2[:, 0])
    y1 = tf.math.maximum(box1[:, None, 1], box2[:, 1])
    x2 = tf.math.minimum(box1[:, None, 2], box2[:, 2])
    y2 = tf.math.minimum(box1[:, None, 3], box2[:, 3])

    # Intersection area
    intersectionArea = tf.math.maximum(0.0, x2-x1)*tf.math.maximum(0.0, y2-y1)

    # Union area
    box1Area = (box1[:, 2]-box1[:, 0])*(box1[:, 3]-box1[:, 1])
    box2Area = (box2[:, 2]-box2[:, 0])*(box2[:, 3]-box2[:, 1])

    unionArea = tf.math.maximum(
        1e-10, box1Area[:, None]+box2Area-intersectionArea)
    iou = intersectionArea/unionArea
    return tf.clip_by_value(iou, 0.0, 1.0)


def total_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    pos_mask = tf.cast(tf.equal(tf.squeeze(y_true[:, :, 4:5], axis=-1), 0.0),
                       tf.float32)
    num_pos = tf.maximum(
        1.0, tf.cast(tf.math.count_nonzero(pos_mask, axis=-1), tf.float32))
    loc_loss = tf.compat.v1.losses.huber_loss(labels=y_true[:, :, :4],
                                              predictions=y_pred[:, :, :4],
                                              reduction="none")

    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    loc_loss = tf.where(tf.equal(pos_mask, 1.0), loc_loss, 0.0)
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    loc_loss = loc_loss / num_pos

    cce = tf.losses.CategoricalCrossentropy(from_logits=True,
                                            reduction=tf.losses.Reduction.NONE)
    cross_entropy = cce(y_true[:, :, 4:], y_pred[:, :, 4:])

    # neg:pos 3:1
    num_neg = 3.0 * num_pos

    # Negative Mining
    neg_cross_entropy = tf.where(tf.equal(pos_mask, 0.0), cross_entropy, 0.0)
    sorted_dfidx = tf.cast(tf.argsort(neg_cross_entropy,
                                      direction='DESCENDING', axis=-1), tf.int32)
    rank = tf.cast(tf.argsort(sorted_dfidx, axis=-1), tf.int32)
    num_neg = tf.cast(num_neg, dtype=tf.int32)
    neg_loss = tf.where(rank < tf.expand_dims(num_neg, axis=1),
                        neg_cross_entropy, 0.0)

    pos_loss = tf.where(tf.equal(pos_mask, 1.0), cross_entropy, 0.0)
    clas_loss = tf.reduce_sum(pos_loss + neg_loss, axis=-1)
    clas_loss = clas_loss / num_pos
    totalloss = loc_loss + clas_loss
    return totalloss


def conv_layer(filter, kernel_size,
               layer, strides=1,
               padding='same',
               activation='linear', pool=False,
               poolsize=2, poolstride=2, conv=True):
    if conv == True:
        layer = tf.keras.layers.Conv2D(filters=filter,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       activation=activation,
                                       padding=padding,
                                       kernel_initializer='he_normal')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
    elif pool == True:
        layer = tf.keras.layers.MaxPool2D(pool_size=(poolsize, poolsize),
                                          strides=poolstride, padding='same')(layer)
    return layer


def ssd_model():
    outputs = []
    mobile_net = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE,
        include_top=False)
    # Feature Layer 1
    layer = mobile_net.get_layer('block_3_depthwise_relu').output
    output = tf.keras.layers.Conv2D(filters=4*(4+NUM_CLASSES+1),
                                    kernel_size=3,
                                    padding='same',
                                    kernel_initializer='glorot_normal')(layer)
    output = tf.keras.layers.Reshape([-1, 4+NUM_CLASSES+1])(output)
    outputs.append(output)
    # Feature Layer 2
    layer = mobile_net.get_layer('block_8_depthwise_relu').output
    output = tf.keras.layers.Conv2D(filters=6*(4+NUM_CLASSES+1),
                                    kernel_size=3,
                                    padding='same',
                                    kernel_initializer='glorot_normal')(layer)
    output = tf.keras.layers.Reshape([-1, 4+NUM_CLASSES+1])(output)
    outputs.append(output)
    # Feature Layer 3
    layer = mobile_net.get_layer('out_relu').output
    output = tf.keras.layers.Conv2D(filters=6*(4+NUM_CLASSES+1),
                                    kernel_size=3,
                                    padding='same',
                                    kernel_initializer='glorot_normal')(layer)
    output = tf.keras.layers.Reshape([-1, 4+NUM_CLASSES+1])(output)
    outputs.append(output)
    # Feature Layer 4
    layer = conv_layer(128, 1, layer)
    layer = conv_layer(256, 3, layer, strides=2)
    output = tf.keras.layers.Conv2D(filters=6*(4+NUM_CLASSES+1),
                                    kernel_size=3,
                                    padding='same',
                                    kernel_initializer='glorot_normal')(layer)
    output = tf.keras.layers.Reshape([-1, 4+NUM_CLASSES+1])(output)
    outputs.append(output)
    out = tf.keras.layers.Concatenate(axis=1)(outputs)
    model = tf.keras.models.Model(mobile_net.input, out, name='SSD')
    model.summary()
    return model


if __name__ == '__main__':
    model = ssd_model()
    tf.keras.utils.plot_model(model, to_file=ssd_model.__name__+'.png')
    out = model.predict(tf.random.normal(shape=(16, 128, 128, 3)))
    print(out)
    print(out.shape)
