
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import io

def pad_to_size(vec,size):
    zeros = [0]*(size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence,pad,model):
    encoder_sample_pred = encoder.encode(sentence)
    if pad:
        encoder_sample_pred = pad_to_size(encoder_sample_pred,64)
    encoder_sample_pred = tf.cast(encoder_sample_pred,tf.float32)
    predictions = model.predict(tf.expand_dims(encoder_sample_pred,0))
    return predictions[0][0]

def get_data():
    dataset, info = tfds.load('imdb_reviews/subwords8k',
                            with_info=True, as_supervised=True)
    train_data, test_data = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    padded_shapes = ([None], ())
    train_dataset = train_data.shuffle(
        BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes)
    test_dataset = test_data.shuffle(
        BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes)
    return train_dataset, test_dataset, encoder


def get_model(encoder, embedding_dim):
    embedding_dim = embedding_dim
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def plot_history(history):
    history = history.history
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, label='Training accuracy')
    plt.plot(epochs, val_acc, label='Test accuracy')
    plt.title('Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()


def retrieve_embedding(model, encoder):
    out_vectors = io.open('classif_vectors.tsv', 'w', encoding='utf8')
    out_metadata = io.open('classif_metadata.tsv', 'w', encoding='utf8')
    weights = model.layers[0].get_weights()[0]

    for num, word in enumerate(encoder.subwords):
        vec = weights[num + 1]
        out_metadata.write(word + '\n')
        out_vectors.write('\t'.join([str(x) for x in vec]) + '\n')
    out_vectors.close()
    out_metadata.close()


if __name__ == '__main__':
    train_data, test_data, encoder = get_data()
    model = get_model(encoder=encoder, embedding_dim=16)
    model.summary()
    # history = model.fit(train_data, epochs=15, validation_data=test_data)
    # model.save('sent_class.h5')
    
    model = tf.keras.models.load_model('sent_class.h5')
    # plot_history(history)
    retrieve_embedding(model, encoder=encoder)
    
    sample_text = ('This movie was awesome. The acting was good enough!!')
    pred = sample_predict(sample_text,True,model)*100
    
    print("Probability of good review:",pred)
    
    sample_text = ("The movies is bad enough! Acting was mediocre. Kind of bad things happened.")
    pred = sample_predict(sample_text,True,model)*100
    
    print("Probability of good review:",pred)
