import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_testing_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, mode, batch_size, resize_dim=(32, 32), shuffle=True, n_channels=3):
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.resize_dim = resize_dim
        self.n_channels = n_channels
        self.load_dataset()
        self.on_epoch_end()

    def load_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        self.images, self.labels = (
            (x_train, y_train) if self.mode == "train" else (x_test, y_test)
        )

        self.indexes = tf.range((self.images).shape[0])

    def __len__(self):
        return int(tf.math.ceil((self.images).shape[0] / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.resize_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for ind, i in enumerate(indexes):
            X[ind] = self.images[i]
            y[ind] = self.labels[i]

        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            tf.random.shuffle(self.indexes)

    def preprocess(self, image):
        image = tf.image.resize(
            image, size=self.resize_dim, method=tf.image.ResizeMethod.BILINEAR
        )
        image = image / 255.0
        image = tf.cast(image, tf.float32)
        return image


class Classification(tf.keras.models.Model):
    def __init__(self, num_gradients=16):
        super(Classification, self).__init__()
        self.num_gradients = tf.constant(num_gradients, tf.int32)
        self.accum_steps = tf.Variable(
            tf.constant(0, dtype=tf.int32),
            trainable=False
        )
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(
            v, tf.float32), trainable=False)
            if v is not None
            else v
            for v in self.trainable_variables
        ]
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def call(self, input_tensor):
        x = self.flatten(input_tensor)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs

    def reset(self):
        if not self.gradient_accumulation:
            return
        self.accum_steps = 0
        for gradient in self.gradient_accumulation:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient), read_value=False)

    def accumulate_gradients(self):
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables)
        )
        self.reset()

    @tf.function
    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            model_output = self(images, training=True)
            loss = tf.reduce_mean(self.loss(labels, model_output))
            acc = self.metric(labels, model_output)
        gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
        tf.cond(
            tf.equal(self.accum_steps, self.num_gradients),
            self.accumulate_gradients,
            lambda: None,
        )
        return {"loss": loss, "acc": acc}


if __name__ == '__main__':
    train_dataset = DataGenerator('train', shuffle=True, batch_size=10000)
    test_dataset = DataGenerator('test', shuffle=True, batch_size=10000)

    mod = Classification()
    mod.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy',tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    mod.fit(train_dataset, validation_data=test_dataset,
            epochs=100, callbacks=callbacks)
