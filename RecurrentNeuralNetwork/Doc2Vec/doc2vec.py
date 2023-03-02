import tensorflow as tf
import numpy as np
import gensim

embedding_dim = 50
window_size = 5
num_epochs = 10
batch_size = 32


def Doc2Vec():
    input_word = tf.keras.layers.Input(shape=(1,))
    input_doc = tf.keras.layers.Input(shape=(1,))
    word_embedding = tf.keras.layers.Embedding(input_dim=len(vocab.wv.index_to_key),
                                               output_dim=embedding_dim,
                                               input_length=1,
                                               name="word_embedding")
    doc_embedding = tf.keras.layers.Embedding(input_dim=len(doc_ids),
                                              output_dim=embedding_dim,
                                              input_length=1,
                                              name="doc_embedding")
    word_embedding_reshaped = tf.keras.layers.GlobalAveragePooling1D(
        name="word_embedding_reshaped")(word_embedding(input_word))
    doc_embedding_reshaped = tf.keras.layers.GlobalAveragePooling1D(
        name="doc_embedding_reshaped")(doc_embedding(input_doc))
    merged = tf.keras.layers.concatenate(
        [word_embedding_reshaped, doc_embedding_reshaped])
    output = tf.keras.layers.Dense(
        units=len(vocab.wv.index_to_key), activation="softmax", name="output")(merged)
    model = tf.keras.Model(inputs=[input_word, input_doc], outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


if __name__ == '__main__':
    corpus = gensim.models.doc2vec.TaggedLineDocument('4153.txt')
    docs = [doc.words for doc in corpus]
    doc_ids = [doc.tags[0] for doc in corpus]
    vocab = gensim.models.Word2Vec(docs, min_count=1)

    word2id = {word: index for index, word in enumerate(vocab.wv.index_to_key)}
    id2word = {index: word for index, word in enumerate(vocab.wv.index_to_key)}
    doc_ids2index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    sequences = [[word2id[word] for word in doc] for doc in docs]

    window_size = 5
    model = Doc2Vec()

    for epoch in range(num_epochs):
        loss = 0
        for i, seq in enumerate(sequences):
            context = [seq[max(0, j - window_size): j] + seq[j + 1: j + window_size + 1]
                       for j in range(len(seq))]
            context = [c for c in context if c]
            if context:
                context_words = np.concatenate(context)
                target_word = np.repeat(seq, len(context_words))
                doc_id = doc_ids[i]
                doc_index = doc_ids2index[doc_id]
                doc_ids_batch = np.repeat(doc_index, len(target_word))
                target_word = target_word.reshape(-1, 1)
                doc_ids_batch = doc_ids_batch.reshape(-1, 1)
                x = np.concatenate([target_word, doc_ids_batch], axis=-1)
                y = tf.keras.utils.to_categorical(
                    context_words, num_classes=len(vocab.wv.index_to_key))
                loss += model.train_on_batch(x=x, y=y)
        print("Epoch:", epoch + 1, "Loss:", loss)

    doc_vectors = np.zeros((len(docs), embedding_dim))
    for i, doc in enumerate(docs):
        doc_id = doc_ids[i]
        doc_index = doc_ids2index[doc_id]
        doc_vector = model.get_layer("doc_embedding")(np.array([doc_index]))
        doc_vectors[i, :] = doc_vector.numpy().squeeze()
        doc_vectors[i, :] = doc_vector

        print(doc_vectors)
