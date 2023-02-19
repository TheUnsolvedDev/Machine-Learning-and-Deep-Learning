import tensorflow as tf

from params import *


def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)


def ConvBatchLReLu(x, filters, kernel_size, index, trainable):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                               padding='same', name='conv_{}'.format(index),
                               use_bias=False, trainable=trainable)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_{}'.format(index),
                                           trainable=trainable)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return (x)


def ConvBatchLReLu_loop(x, index, convstack, trainable):
    for para in convstack:
        x = ConvBatchLReLu(x, para["filters"],
                           para["kernel_size"], index, trainable)
        index += 1
    return (x)


def define_YOLOv2(IMAGE_H, IMAGE_W, GRID_H, GRID_W, TRUE_BOX_BUFFER, BOX, CLASS, trainable=False):
    convstack3to5 = [{"filters": 128, "kernel_size": (3, 3)},  # 3
                     {"filters": 64,  "kernel_size": (1, 1)},  # 4
                     {"filters": 128, "kernel_size": (3, 3)}]  # 5

    convstack6to8 = [{"filters": 256, "kernel_size": (3, 3)},  # 6
                     {"filters": 128, "kernel_size": (1, 1)},  # 7
                     {"filters": 256, "kernel_size": (3, 3)}]  # 8

    convstack9to13 = [{"filters": 512, "kernel_size": (3, 3)},  # 9
                      {"filters": 256, "kernel_size": (1, 1)},  # 10
                      {"filters": 512, "kernel_size": (3, 3)},  # 11
                      {"filters": 256, "kernel_size": (1, 1)},  # 12
                      {"filters": 512, "kernel_size": (3, 3)}]  # 13

    convstack14to20 = [{"filters": 1024, "kernel_size": (3, 3)},  # 14
                       {"filters": 512,  "kernel_size": (1, 1)},  # 15
                       {"filters": 1024, "kernel_size": (3, 3)},  # 16
                       {"filters": 512,  "kernel_size": (1, 1)},  # 17
                       {"filters": 1024, "kernel_size": (3, 3)},  # 18
                       {"filters": 1024, "kernel_size": (3, 3)},  # 19
                       {"filters": 1024, "kernel_size": (3, 3)}]  # 20

    input_image = tf.keras.layers.Input(
        shape=(IMAGE_H, IMAGE_W, 3), name="input_image")
    true_boxes = tf.keras.layers.Input(
        shape=(1, 1, 1, TRUE_BOX_BUFFER, 4), name="input_hack")
    # Layer 1
    x = ConvBatchLReLu(input_image, filters=32, kernel_size=(
        3, 3), index=1, trainable=trainable)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="maxpool1_416to208")(x)
    # Layer 2
    x = ConvBatchLReLu(x, filters=64, kernel_size=(3, 3),
                       index=2, trainable=trainable)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="maxpool1_208to104")(x)

    # Layer 3 - 5
    x = ConvBatchLReLu_loop(x, 3, convstack3to5, trainable)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="maxpool1_104to52")(x)

    # Layer 6 - 8
    x = ConvBatchLReLu_loop(x, 6, convstack6to8, trainable)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="maxpool1_52to26")(x)

    # Layer 9 - 13
    x = ConvBatchLReLu_loop(x, 9, convstack9to13, trainable)

    skip_connection = x
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), name="maxpool1_26to13")(x)

    # Layer 14 - 20
    x = ConvBatchLReLu_loop(x, 14, convstack14to20, trainable)

    # Layer 21
    skip_connection = ConvBatchLReLu(skip_connection, filters=64,
                                     kernel_size=(1, 1), index=21, trainable=trainable)
    skip_connection = tf.keras.layers.Lambda(
        space_to_depth_x2)(skip_connection)

    x = tf.keras.layers.Concatenate()([skip_connection, x])

    # Layer 22
    x = ConvBatchLReLu(x, filters=1024, kernel_size=(3, 3),
                       index=22, trainable=trainable)

    # Layer 23
    x = tf.keras.layers.Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1),
                               padding='same', name='conv_23')(x)
    output = tf.keras.layers.Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS),
                                     name="final_output")(x)

    output = tf.keras.layers.Lambda(lambda args: args[0], name="hack_layer")(
        [output, true_boxes])

    model = tf.keras.Model([input_image, true_boxes], output)
    return (model, true_boxes)


def get_cell_grid(GRID_W, GRID_H, BATCH_SIZE, BOX):
    cell_x = tf.cast(tf.reshape(
        tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    cell_grid = tf.tile(
        tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, BOX, 1])
    return (cell_grid)


def adjust_scale_prediction(y_pred, cell_grid, ANCHORS):
    BOX = int(len(ANCHORS)/2)
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * \
        np.reshape(ANCHORS, [1, 1, 1, BOX, 2])
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    pred_box_class = y_pred[..., 5:]
    return (pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class)


def extract_ground_truth(y_true):
    true_box_xy = y_true[..., 0:2]
    true_box_wh = y_true[..., 2:4]
    true_box_conf = y_true[..., 4]
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    return (true_box_xy, true_box_wh, true_box_conf, true_box_class)


def calc_loss_xywh(true_box_conf,
                   COORD_SCALE,
                   true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):
    coord_mask = tf.expand_dims(true_box_conf, axis=-1) * COORD_SCALE
    nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32))
    loss_xy = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)
                            * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)
                            * coord_mask) / (nb_coord_box + 1e-6) / 2.
    return (loss_xy + loss_wh, coord_mask)


def calc_loss_class(true_box_conf, CLASS_SCALE, true_box_class, pred_box_class):
    class_mask = true_box_conf * CLASS_SCALE  # L_{i,j}^obj * lambda_class

    nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32))
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class,
                                                                logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    return (loss_class)


def get_intersect_area(true_xy, true_wh,
                       pred_xy, pred_wh):
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    return (iou_scores)


def calc_IOU_pred_true_assigned(true_box_conf,
                                true_box_xy, true_box_wh,
                                pred_box_xy,  pred_box_wh):
    iou_scores = get_intersect_area(true_box_xy, true_box_wh,
                                    pred_box_xy, pred_box_wh)
    true_box_conf_IOU = iou_scores * true_box_conf
    return (true_box_conf_IOU)


def calc_IOU_pred_true_best(pred_box_xy, pred_box_wh, true_boxes):
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    iou_scores = get_intersect_area(true_xy,
                                    true_wh,
                                    pred_xy,
                                    pred_wh)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    return (best_ious)


def get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU, LAMBDA_NO_OBJECT, LAMBDA_OBJECT):
    conf_mask = tf.cast(best_ious < 0.6, dtype=tf.float32) * \
        (1 - true_box_conf) * LAMBDA_NO_OBJECT
    conf_mask = conf_mask + true_box_conf_IOU * LAMBDA_OBJECT
    return (conf_mask)


def calc_loss_conf(conf_mask, true_box_conf_IOU, pred_box_conf):
    nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, dtype=tf.float32))
    loss_conf = tf.reduce_sum(tf.square(
        true_box_conf_IOU-pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    return (loss_conf)


def custom_loss_core(y_true,
                     y_pred,
                     true_boxes,
                     GRID_W,
                     GRID_H,
                     BATCH_SIZE,
                     ANCHORS,
                     LAMBDA_COORD,
                     LAMBDA_CLASS,
                     LAMBDA_NO_OBJECT, 
                     LAMBDA_OBJECT):
    BOX = int(len(ANCHORS)/2)    
    cell_grid   = get_cell_grid(GRID_W,GRID_H,BATCH_SIZE,BOX)
    pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_prediction(y_pred,cell_grid,ANCHORS)
    true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true)
    loss_xywh, coord_mask = calc_loss_xywh(true_box_conf,LAMBDA_COORD,
                                           true_box_xy, pred_box_xy,true_box_wh,pred_box_wh)
    loss_class  = calc_loss_class(true_box_conf,LAMBDA_CLASS,
                                   true_box_class,pred_box_class)
    true_box_conf_IOU = calc_IOU_pred_true_assigned(true_box_conf,
                                                    true_box_xy, true_box_wh,
                                                    pred_box_xy, pred_box_wh)
    best_ious = calc_IOU_pred_true_best(pred_box_xy,pred_box_wh,true_boxes)
    conf_mask = get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU,LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
    loss_conf = calc_loss_conf(conf_mask,true_box_conf_IOU, pred_box_conf)
    loss = loss_xywh + loss_conf + loss_class
    return(loss)


if __name__ == "__main__":
    IMAGE_H, IMAGE_W = 416, 416  # IMG_SIZE[0], IMG_SIZE[1]
    GRID_H,  GRID_W = 13, 13
    TRUE_BOX_BUFFER = 50
    BOX = int(len(ANCHORS)/2)
    CLASS = len(LABELS)
    model, true_boxes = define_YOLOv2(IMAGE_H, IMAGE_W, GRID_H, GRID_W, TRUE_BOX_BUFFER, BOX, CLASS,
                                      trainable=False)

    model.summary()
    print('Memory consumption is:',
          get_model_memory_usage(model, BATCH_SIZE), 'GB')
