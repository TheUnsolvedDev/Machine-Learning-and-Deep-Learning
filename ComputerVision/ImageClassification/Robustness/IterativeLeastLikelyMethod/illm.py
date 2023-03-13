import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def lenet5_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(
        3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    return model


loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(model, input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad


def adversarial_image(image, eps):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        label = tf.argmin(prediction, 1)
        one_hot = tf.one_hot(label, 10)
        loss = loss_object(one_hot, prediction)
    gradient = tape.gradient(loss, image)
    x_adv = image - eps*tf.sign(gradient[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train/255, axis=-1)
    x_test = tf.expand_dims(x_test/255, axis=-1)
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    model = lenet5_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
    model.save_weights('Lenet.h5')

    model.load_weights('Lenet.h5')
    image = tf.expand_dims(x_train[0], axis=0)
    label = tf.expand_dims(y_train[0], axis=0)

    perturbations = create_adversarial_pattern(model, image, label)
    plt.imshow(image[0])
    plt.show()
    plt.imshow(perturbations[0]*0.5+0.5)
    plt.show()

    epsilons = [0, 0.01, 0.1, 0.15]
    for i, eps in enumerate(epsilons):
        adv_x = adversarial_image(image, eps)
        print(np.argmax(model.predict(adv_x, verbose=0), axis=-1))
