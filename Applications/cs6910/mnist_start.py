import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

from model import *
from optimizers import *

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.array(x_train.reshape(-1, 28*28)/255.0, dtype=np.float32)
    x_test = x_test.reshape(-1, 28*28)/255.0

    inputs = 784
    outputs = 10
    model = NNClassifier([inputs, 128, 128, outputs])
    optimizer = Adam(model, lr=0.001)

    n_epochs = 30
    loss_history = np.zeros(n_epochs)
    accuracy_history = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        # Forward pass
        batch_size = 100
        num_batches = len(x_train)//batch_size
        indices = [i for i in range(num_batches)]
        np.random.shuffle(indices)

        for i in range(num_batches):
            ind = indices[i]
            data = x_train[ind*batch_size:(ind+1)*batch_size]
            labels = y_train[ind*batch_size:(ind+1)*batch_size]

            loss_history[epoch] = model.loss(data.T, labels)
            accuracy_history[epoch] = model.accuracy(data.T, labels)
            print('\repochs;', epoch, 'batch [', i, '/', num_batches, ']', 'loss:', loss_history[epoch],
                  'accuracy:', accuracy_history[epoch], end='')
            sys.stdout.flush()

            # Backward pass
            model.backward()
            
            # Parameter updates
            optimizer.step(epoch)

        print('\n')
        train_loss = model.loss(x_train.T, y_train)
        loss_history[epoch] = test_loss = model.loss(x_test.T, y_test)

        train_accuracy = model.accuracy(x_train.T, y_train)
        accuracy_history[epoch] = test_accuracy = model.accuracy(
            x_test.T, y_test)

        print('Train Acc:', round(train_accuracy*100, 2), 'Test Acc:', round(test_accuracy *
              100, 2), 'Train Loss:', round(train_loss, 5), 'Test Loss:', round(test_loss, 5))

    plt.plot(loss_history)
    plt.title('Final loss: {:.3f}'.format(np.min(loss_history)))
    plt.xlabel('Epochs')
    plt.show()

    plt.plot(accuracy_history)
    plt.title('Final accuracy: {:.3f}'.format(np.min(accuracy_history)))
    plt.xlabel('Epochs')
    plt.show()
