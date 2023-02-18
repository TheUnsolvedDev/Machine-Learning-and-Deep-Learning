import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

IMG_SIZE = (64, 64, 3)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_ResNext_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def conv_block(inputs, filters, kernel_size, strides, cardinality):
    group_channels = filters // cardinality
    groups = tf.split(inputs, cardinality, axis=-1)
    conv_outputs = []
    for i in range(cardinality):
        conv_outputs.append(tf.keras.layers.Conv2D(
            filters=group_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding='same')(groups[i]))
    concat = tf.keras.layers.Concatenate(axis=-1)(conv_outputs)
    outputs = tf.keras.layers.BatchNormalization()(concat)
    outputs = tf.keras.layers.Activation('relu')(outputs)
    return outputs

def resnext_block(inputs, filters, kernel_size, strides, cardinality):
    conv_outputs = conv_block(inputs, filters, kernel_size, strides, cardinality)
    if inputs.shape[-1] != filters:
        skip_connection = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=strides,
            padding='same')(inputs)
        skip_connection = tf.keras.layers.BatchNormalization()(skip_connection)
    else:
        skip_connection = inputs
    outputs = tf.keras.layers.Add()([conv_outputs, skip_connection])
    outputs = tf.keras.layers.Activation('relu')(outputs)
    return outputs

def ResNeXt(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # ResNeXt blocks
    x = resnext_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2), cardinality=32)
    x = resnext_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2), cardinality=32)
    x = resnext_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2), cardinality=32)
    
    # Global average pooling and dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


if __name__ == '__main__':
    model = ResNeXt()
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=ResNeXt.__name__+'.png', show_shapes=True, expand_nested=True)

    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (validation_images, validation_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(
        validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=64, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=64, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=64, drop_remainder=True))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
