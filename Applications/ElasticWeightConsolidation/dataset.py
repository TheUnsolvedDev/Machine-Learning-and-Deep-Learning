import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        (self.train_data,self.train_labels),(self.test_data,self.test_labels) = tf.keras.datasets.mnist.load_data()
        
    def task_A(self):
        return (self.train_data[self.train_labels == 0]/255.0,self.train_labels[self.train_labels == 0])
    
    def task_B(self):
        return (self.train_data[self.train_labels == 1]/255.0,self.train_labels[self.train_labels == 1])

    def task_C(self):
        return (self.train_data[self.train_labels == 2]/255.0,self.train_labels[self.train_labels == 2])

    def task_D(self):
        return (self.train_data[self.train_labels == 3]/255.0,self.train_labels[self.train_labels == 3])
    
if __name__ == "__main__":
    data = Dataset()
    Aa,Ab = data.task_A()
    Ba,Bb = data.task_B()
    print(Aa.shape,Ab.shape)

