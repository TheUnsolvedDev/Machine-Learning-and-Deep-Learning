import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

generation_counter = 0


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (32, 32))
    return image, label


def create_cnn_model(input_shape, num_classes, architecture):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    if 'dense' in architecture[-1].name:
        architecture = architecture[:-1]

    for layer in architecture:
        if 'input' not in layer.name:
            x = layer(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_population(input_shape, num_classes, population_size):
    population = []
    for i in range(population_size):
        architecture = []
        num_conv_layers = np.random.randint(1, 3)
        for j in range(num_conv_layers):
            filters = np.random.randint(32, 128)
            kernel_size = np.random.randint(3, 5)
            architecture.append(tf.keras.layers.Conv2D(
                filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal'))
            if np.random.rand() > 0.5:
                pool_size = np.random.randint(2, 4)
                strides = np.random.randint(1, 3)
                architecture.append(
                    tf.keras.layers.MaxPooling2D(pool_size, strides))
        architecture.append(tf.keras.layers.Flatten())
        num_dense_layers = np.random.randint(1, 3)
        for j in range(num_dense_layers):
            units = np.random.randint(64, 128)
            architecture.append(
                tf.keras.layers.Dense(units, activation='relu', kernel_initializer='he_normal'))
        model = create_cnn_model(input_shape, num_classes, architecture)
        population.append(model)
    return population


def evaluate_population(population, train, val):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]
    scores = []
    for population_counter, model in enumerate(population):
        population_counter += 1
        tf.keras.utils.plot_model(
            model, to_file='models_arch/model_gen'+str(generation_counter)+'_pop'+str(population_counter)+'.png', show_layer_activations=True, show_layer_names=True)

        model.summary()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train, epochs=3, batch_size=32,
                  validation_data=val, callbacks=callbacks)
        # Evaluate the model
        score = model.evaluate(val, verbose=0)
        scores.append(score[1])
    return scores


def select_parents(population, scores, num_parents):
    population_sorted = [x for _, x in sorted(
        zip(scores, population), key=lambda pair: pair[0], reverse=True)]
    parents = population_sorted[:num_parents]
    return parents


def breed_next_generation(parents, num_children):
    children = []
    for parent in parents:
        print([layer.name for layer in parent.layers])

    for i in range(num_children):
        convs = []
        denses = []
        parent1 = np.random.choice(parents)
        parent2 = np.random.choice(parents)
        child_architecture = []

        for layer_idx in range(min(len(parent1.layers), len(parent2.layers))):
            if np.random.rand() > 0.5:
                child_layer = parent1.layers[layer_idx]
            else:
                child_layer = parent2.layers[layer_idx]
            if isinstance(child_layer, tf.keras.layers.Conv2D):
                convs.append(child_layer)
            elif isinstance(child_layer, tf.keras.layers.Dense):
                denses.append(child_layer)

        for child_layer in convs:
            if isinstance(child_layer, tf.keras.layers.Conv2D):
                # child_layer.filters = np.random.randint(16, 64)
                # child_layer.kernel_size = np.random.randint(3, 5)
                child_layer = tf.keras.layers.Conv2D(
                    child_layer.filters, child_layer.kernel_size, activation='relu', kernel_initializer='he_normal')
            elif isinstance(child_layer, tf.keras.layers.MaxPooling2D):
                # child_layer.pool_size = np.random.randint(2, 4)
                # child_layer.strides = np.random.randint(1, 3)
                child_layer = tf.keras.layers.MaxPooling2D(
                    child_layer.pool_size, child_layer.strides)
            child_architecture.append(child_layer)

        child_architecture.append(tf.keras.layers.Flatten())

        for child_layer in denses:
            if isinstance(child_layer, tf.keras.layers.Dense):
                # units = np.random.randint(16, 64)
                child_layer = tf.keras.layers.Dense(
                    units=child_layer.units, activation='relu', kernel_initializer='he_normal')
            child_architecture.append(child_layer)
        print([l.name for l in child_architecture], 10000)
        child = create_cnn_model(input_shape, num_classes, child_architecture)
        children.append(child)
    return children


if __name__ == '__main__':
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]
    input_shape = (32, 32, 3)
    num_classes = 10

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (validation_images, validation_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(
        validation_ds).numpy()

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

    population_size = 5
    num_parents = 3
    num_children = 3
    population = create_population(input_shape, num_classes, population_size)
    scores = evaluate_population(
        population, train_ds, validation_ds)

    num_generations = 10
    for i in range(num_generations):
        generation_counter += 1
        parents = select_parents(population, scores, num_parents)
        children = breed_next_generation(parents, num_children)
        children_scores = evaluate_population(
            children, train_ds, test_ds)
        population = parents + children
        scores = scores[:num_parents] + children_scores
    best_model = population[np.argmax(scores)]
    tf.keras.models.save_model(best_model, filepath='weights_and_model')
