import tensorflow as tf
import numpy as np
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3,):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions /
                                  self.temperature, axis=1),
                    tf.nn.softmax(student_predictions /
                                  self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + \
                (1 - self.alpha) * distillation_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.compiled_metrics.update_state(y, y_prediction)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


teacher = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(
            2, 2), strides=(1, 1), padding="same"),
        tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10),
    ],
    name="teacher",
)

# Create the student
student = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(
            2, 2), strides=(1, 1), padding="same"),
        tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10),
    ],
    name="student",
)

if __name__ == "__main__":
    tf.keras.utils.plot_model(
        student, to_file='student_model.png', show_shapes=True)
    tf.keras.utils.plot_model(
        teacher, to_file='teacher_model.png', show_shapes=True)

    student_scratch = tf.keras.models.clone_model(student)
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    teacher_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model_teacher.h5', save_weights_only=True, monitor='loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs_teacher', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

    teacher.fit(x_train, y_train, epochs=10, callbacks=teacher_callbacks)
    teacher.evaluate(x_test, y_test)

    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )
    distiller_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model_distiller.h5', save_weights_only=True, monitor='loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs_distiller', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

    distiller.fit(x_train, y_train, epochs=5, callbacks=distiller_callbacks)
    distiller.evaluate(x_test, y_test)

    student_scratch.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    student_scratch_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model_student.h5', save_weights_only=True, monitor='loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs_student', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

    student_scratch.fit(x_train, y_train, epochs=5,
                        callbacks=student_scratch_callbacks)
    student_scratch.evaluate(x_test, y_test)
