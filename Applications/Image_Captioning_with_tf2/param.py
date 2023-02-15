import tensorflow as tf

BATCH_SIZE = 16
IMAGE_SIZE = [128, 128, 3]
FULL_DATA_SIZE = 600000
VOCAB = 50000
STATE_SIZE = 64
EMBEDDING_SIZE = 128
MAX_LENGTH = 15
EPOCHS = 30


def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return inputs
