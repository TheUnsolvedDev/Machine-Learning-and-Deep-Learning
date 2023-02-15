import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from dataset import read_data_cites
from model import DeepNeuralNetwork

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelA*.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=20, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]


def split_data(citations, papers, percentage=70):
    train_data, test_data = [], []
    for _, group_data in papers.groupby('subject'):
        random_selection = np.random.rand(
            len(group_data.index)) <= (percentage/100)
        train_data.append(group_data[random_selection])
        test_data.append(group_data[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)
    return train_data, test_data


def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["accuracy"])
    ax2.plot(history.history["val_accuracy"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


def train_model(model, train_data, train_label, test_data, test_label):
    history = model.fit(train_data, train_label, epochs=50, batch_size=100,
                        callbacks=callbacks, validation_data=(test_data, test_label))
    display_learning_curves(history)
    return history


def generate_random_instances(num_instances):
    token_probability = x_train.mean(axis=0)
    instances = []
    for _ in range(num_instances):
        probabilities = np.random.uniform(size=len(token_probability))
        instance = (probabilities <= token_probability).astype(int)
        instances.append(instance)

    return np.array(instances)


def display_class_probabilities(probabilities):
    class_values = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                    'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory']
    for instance_idx, probs in enumerate(probabilities):
        print(f"Instance {instance_idx + 1}:")
        for class_idx, prob in enumerate(probs):
            print(f"- {class_values[class_idx]}: {round(prob * 100, 2)}%")


if __name__ == '__main__':
    citations, papers = read_data_cites()

    train_data, test_data = split_data(citations, papers)
    feature_names = set(papers.columns) - {"paper_id", "subject"}
    num_features = len(feature_names)
    num_classes = 7

    # Create train and test features as a numpy array.
    x_train = np.array(train_data[feature_names])
    x_test = np.array(test_data[feature_names])
    # Create train and test targets as a numpy array.
    y_train = train_data["subject"]
    y_test = test_data["subject"]

    model = DeepNeuralNetwork()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    train_model(model, x_train, y_train, x_test, y_test)
    new_instance = generate_random_instances(num_classes)
    probs = model.predict(new_instance)

    display_class_probabilities(probs)
