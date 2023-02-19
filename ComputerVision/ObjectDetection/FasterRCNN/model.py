import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from params import *


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(
        filter, (3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ResNet34(shape=(256, 256, 3), classes=10):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same')(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [2, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet34")
    return model


class ROIPoolingLayer(tf.keras.layers.Layer):
    """ Implements Region Of Interest Max Pooling 
        for channel-first images and relative bounding box coordinates

        # Constructor parameters
            pooled_height, pooled_width (int) -- 
              specify height and width of layer outputs

        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]

        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    """

    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

        super(ROIPoolingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output

            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative 
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x):
            return ROIPoolingLayer._pool_rois(x[0], x[1],
                                              self.pooled_height,
                                              self.pooled_width)

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi):
            return ROIPoolingLayer._pool_roi(feature_map, roi,
                                             pooled_height, pooled_width)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        # Compute the region of interest
        feature_map_height = int(feature_map.shape[0])
        feature_map_width = int(feature_map.shape[1])

        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width * roi[1], 'int32')
        h_end = tf.cast(feature_map_height * roi[2], 'int32')
        w_end = tf.cast(feature_map_width * roi[3], 'int32')

        region = feature_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width = w_end - w_start
        h_step = tf.cast(region_height / pooled_height, 'int32')
        w_step = tf.cast(region_width / pooled_width, 'int32')

        areas = [[(
            i*h_step,
            j*w_step,
            (i+1)*h_step if i+1 < pooled_height else region_height,
            (j+1)*w_step if j+1 < pooled_width else region_width
        )
            for j in range(pooled_width)]
            for i in range(pooled_height)]

        # take the maximum of each area and stack the result
        def pool_area(x):
            return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0, 1])

        pooled_features = tf.stack(
            [[pool_area(x) for x in row] for row in areas])
        return pooled_features


def classifier_layer(base_layers, input_rois, num_rois, nb_classes=20):
    # base_layers: vgg
    # input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # num_rois: number of rois to be processed in one time (4 in here)
    input_shape = (num_rois, 7, 7, 512)
    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = ROIPoolingLayer(16, 16)([
        base_layers, input_rois])
    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    # expanded into a vector with 25,088 (7×7×512) channels.
    out = tf.keras.layers.Flatten(name='flatten')(out_roi_pool)
    out = tf.keras.layers.Dense(1024, activation='relu', name='fc1')(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(1024, activation='relu', name='fc2')(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    # two output layer- classifier and regressor
    # for classify the class name of the object
    out_class = tf.keras.layers.Dense(nb_classes, activation='softmax',
                                      kernel_initializer='zero', name='dense_class_{}'.format(nb_classes))(out)
    # for bboxes coordinates regression
    out_regr = tf.keras.layers.Dense(4 * (nb_classes-1), activation='linear',
                                     kernel_initializer='zero', name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]


def detector_with_rpn():
    model = ResNet34()
    inputs = model.input
    partial_outputs = model.output
    output = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="rpn_conv")(partial_outputs)
    rpn_cls_output = tf.keras.layers.Conv2D(
        hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = tf.keras.layers.Conv2D(
        hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = tf.keras.Model(inputs=inputs, outputs=[
                               rpn_reg_output, rpn_cls_output])

    inputs = rpn_model.input
    base_layers = rpn_model.output
    input_rois = (1, 4, 4)
    num_rois = 4
    print(base_layers)
    outputs = classifier_layer(base_layers, input_rois, num_rois, 20)
    final_model = tf.keras.Model(inputs, outputs)
    return final_model


if __name__ == '__main__':
    m = ResNet34()
    m.summary()
    m2 = detector_with_rpn()
    m2.summary()
