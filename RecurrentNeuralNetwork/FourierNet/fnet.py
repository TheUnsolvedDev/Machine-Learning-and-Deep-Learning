import numpy as np
import tensorflow as tf
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_fourier_text_net.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

VOCAB_SIZE = 8192
MAX_SAMPLES = 50000
BUFFER_SIZE = 20000
MAX_LENGTH = 40
EMBED_DIM = 256
LATENT_DIM = 512
NUM_HEADS = 8
BATCH_SIZE = 64


def preprocess_text(sentence):
    sentence = tf.strings.lower(sentence)
    sentence = tf.strings.regex_replace(sentence, r"([?.!,])", r" \1 ")
    sentence = tf.strings.regex_replace(sentence, r"\s\s+", " ")
    sentence = tf.strings.regex_replace(sentence, r"[^a-z?.!,]+", " ")
    sentence = tf.strings.strip(sentence)
    sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
    return sentence


def load_conversations():
    # Helper function for loading the conversation splits
    id2line = {}
    with open(path_to_movie_lines, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(path_to_movie_conversations, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            inputs.append(id2line[conversation[i]])
            outputs.append(id2line[conversation[i + 1]])
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs


def vectorize_text(inputs, outputs):
    inputs, outputs = vectorizer(inputs), vectorizer(outputs)
    outputs = tf.pad(outputs, [[0, 1]])
    return (
        {"encoder_inputs": inputs, "decoder_inputs": outputs[:-1]},
        {"outputs": outputs[1:]},
    )


class FNetEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super(FNetEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(dense_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        inp_complex = tf.cast(inputs, tf.complex64)
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class FNetDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(FNetDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(latent_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


def create_model():
    encoder_inputs = tf.keras.layers.Input(
        shape=(None,), dtype="int32", name="encoder_inputs")
    x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
    encoder_outputs = FNetEncoder(EMBED_DIM, LATENT_DIM)(x)
    encoder = tf.keras.Model(encoder_inputs, encoder_outputs)
    decoder_inputs = tf.keras.layers.Input(
        shape=(None,), dtype="int32", name="decoder_inputs")
    encoded_seq_inputs = tf.keras.layers.Input(
        shape=(None, EMBED_DIM), name="decoder_state_inputs"
    )
    x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
    x = FNetDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x, encoded_seq_inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    decoder_outputs = tf.keras.layers.Dense(
        VOCAB_SIZE, activation="softmax")(x)
    decoder = tf.keras.Model(
        [decoder_inputs, encoded_seq_inputs], decoder_outputs, name="outputs"
    )
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    fnet = tf.keras.Model([encoder_inputs, decoder_inputs],
                          decoder_outputs, name="fnet")
    return fnet


def decode_sentence(input_sentence):
    tokenized_input_sentence = vectorizer(
        tf.constant("[start] " + preprocess_text(input_sentence) + " [end]")
    )
    tokenized_target_sentence = tf.expand_dims(VOCAB.index("[start]"), 0)
    decoded_sentence = ""

    for i in range(MAX_LENGTH):
        # Get the predictions
        predictions = fnet.predict(
            {
                "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0),
                "decoder_inputs": tf.expand_dims(
                    tf.pad(
                        tokenized_target_sentence,
                        [[0, MAX_LENGTH -
                            tf.shape(tokenized_target_sentence)[0]]],
                    ),
                    0,
                ),
            }
        )
        sampled_token_index = tf.argmax(predictions[0, i, :])
        sampled_token = VOCAB[sampled_token_index.numpy()]
        if tf.equal(sampled_token_index, VOCAB.index("[end]")):
            break
        decoded_sentence += sampled_token + " "
        tokenized_target_sentence = tf.concat(
            [tokenized_target_sentence, [sampled_token_index]], 0
        )

    return decoded_sentence


if __name__ == '__main__':
    path_to_zip = tf.keras.utils.get_file(
        "cornell_movie_dialogs.zip",
        origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
        extract=True,
    )

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus"
    )
    path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
    path_to_movie_conversations = os.path.join(
        path_to_dataset, "movie_conversations.txt")

    questions, answers = load_conversations()
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (questions[:40000], answers[:40000]))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (questions[40000:], answers[40000:]))
    vectorizer = tf.keras.layers.TextVectorization(
        VOCAB_SIZE,
        standardize=preprocess_text,
        output_mode="int",
        output_sequence_length=MAX_LENGTH,
    )

    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(
        (questions + answers)).batch(128))
    train_dataset = train_dataset.map(
        vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(
        vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = (
        train_dataset.cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    fnet = create_model()
    tf.keras.utils.plot_model(fnet, to_file='fnet_model.png', show_shapes=True,
                              show_layer_activations=True, show_layer_names=True)
    fnet.compile("adam", loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    fnet.fit(train_dataset, epochs=100,
             validation_data=val_dataset, callbacks=callbacks)
    VOCAB = vectorizer.get_vocabulary()

    print(decode_sentence("Where have you been all this time?"))
    print(decode_sentence("Did you miss me?"))
    print(decode_sentence("Who are you?"))
