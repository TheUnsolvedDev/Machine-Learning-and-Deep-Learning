import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tempfile

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_deep_compress.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])


if __name__ == '__main__':
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    train_images, test_images = train_images / 255.0, test_images / 255.0
    tf.keras.utils.plot_model(
        model, to_file='compression_model.png', show_shapes=True)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=50, validation_data=(
        test_images, test_labels), callbacks=callbacks)
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                               final_sparsity=0.90,
                                                                               begin_step=0,
                                                                               end_step=10000,
                                                                               frequency=100)}
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model, **pruning_params)
    pruned_model.compile(optimizer='adam',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy'])

    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)

    tf.keras.models.save_model(
        model_for_export, 'pruned_model.h5', include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()

    with open('pruned_model.tflite', 'wb') as f:
        f.write(pruned_tflite_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()
    

    with open('quantized_and_pruned_model.tflite', 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)
