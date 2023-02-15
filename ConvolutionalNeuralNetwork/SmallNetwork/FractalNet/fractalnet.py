import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
Fractal nets are a type of deep neural network architecture that are designed to 
improve the representational capacity and efficiency of the network. They achieve 
this by using a hierarchical structure that consists of recursive blocks, which 
are similar to the branching patterns found in fractals.

One of the main benefits of fractal nets is that they can be made very deep without 
requiring a large number of parameters. This is because the recursive blocks in a 
fractal net can be shared across different layers of the network, which allows the 
network to learn more complex features using fewer parameters. This makes fractal 
nets particularly useful for tasks that require a high level of abstraction and 
spatial reasoning, such as image classification, segmentation, and object detection.

Another advantage of fractal nets is that they can be trained using standard 
gradient-based optimization algorithms, such as stochastic gradient descent (SGD), 
without the need for special techniques like residual learning or skip connections. 
This makes them relatively easy to implement and train.

Overall, fractal nets are a promising alternative to traditional deep neural network 
architectures, and have shown promising results in a variety of tasks. However, more 
research is needed to fully understand their capabilities and limitations.
'''

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_fractalnet.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, label


def solo(x_inp, filters, depth):
    x_inp = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu')(x_inp)
    x_inp = tf.keras.layers.BatchNormalization()(x_inp)

    x_inp = tf.keras.layers.MaxPool2D(4**(depth), 4**(depth))(x_inp)
    return x_inp


def fractal_block(input_tensor, filters, depth=2):
    if depth == 1:
        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same', activation='relu')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        y = solo(input_tensor, filters, depth)
        return tf.keras.layers.Add()([x, y])

    new1 = fractal_block(input_tensor, filters, depth - 1)
    new2 = fractal_block(new1, filters, depth - 1)
    new3 = solo(input_tensor, filters, depth)
    x = tf.keras.layers.Add()(
        [new2, new3])

    return x

# Define the fractal net model


def fractal_net(input_shape=(64, 64, 3), num_classes=10):
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    inputs = tf.keras.layers.Input(shape=input_shape)
    x_inp = tf.keras.layers.Lambda(lambda i: i/255.0)(inputs)
    x_inp = data_augmentation(x_inp)
    x = x_inp

    # Create the first recursive block
    x = fractal_block(x, 64)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Activation('softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
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
                .batch(batch_size=8, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=8, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=8, drop_remainder=True))
    model = fractal_net()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file=fractal_net.__name__+'.png', show_shapes=True, show_layer_names=True)
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
