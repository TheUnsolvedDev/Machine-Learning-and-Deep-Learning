import tensorflow as tf
import tensorflow_probability as tfp


def mixup(x, y, alpha=0.2):
    lam = tfp.distributions.Beta(alpha, alpha).sample((x.shape[0],))
    lam = tf.reshape(lam, (x.shape[0], 1, 1, 1))
    index = tf.range(x.shape[0])
    index = tf.random.shuffle(index)

    x_gathered = tf.gather(tf.cast(x, tf.float32), index, axis=0)
    left = lam*x
    print(left.shape)
    right = (1-lam)*x_gathered

    x = left + right
    y = lam * tf.cast(y, tf.float32) + (1 - lam) * \
        tf.gather(tf.cast(y, tf.float32), index, axis=0)
    print(lam.shape, x.shape, y.shape)
    input()
    return x, y


if __name__ == "__main__":
    a = tf.ones((4, 28, 28, 1), dtype=tf.float32)
    y = tf.ones((4,), dtype=tf.float32)
    print(mixup(a, y))
