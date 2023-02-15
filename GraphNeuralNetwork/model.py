import tensorflow as tf


def DeepNeuralNetwork(input_shape=(1433,), output_shape=7):
    inputs = tf.keras.layers.Input(shape=input_shape, name="Input Features")
    hidden = tf.keras.layers.BatchNormalization()(inputs)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    hidden = tf.keras.layers.Dense(32)(hidden)

    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    hidden = tf.keras.layers.Dense(32)(hidden)

    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.5)(hidden)
    hidden = tf.keras.layers.Dense(32)(hidden)

    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline")
    return model


if __name__ == "__main__":
    m = DeepNeuralNetwork()
    m.summary()
