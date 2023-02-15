import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from model import *

if __name__ == '__main__':
    obj = Dataset()
    A1, A2, A3, A4 = obj.task_A()
    B1, B2, B3, B4 = obj.task_B()

    model_A = lenet5()
    model_A.load_weights('modelA*.h5')

    print()
    print('Training Evaluation A train task:')
    model_A.evaluate(A1, A2)
    print()

    print()
    print('Training Evaluation A test task:')
    model_A.evaluate(A3, A4)
    print()

    model_B = lenet5()
    model_B.load_weights('modelB.h5')

    print()
    print('Training Evaluation B train task:')
    model_B.evaluate(B1, B2)
    print()

    print()
    print('Training Evaluation B test task:')
    model_B.evaluate(B3, B4)
    print()
    
    print()
    print('After EWC')
    
    print()
    print('Training Evaluation A train task:')
    model_B.evaluate(A1, A2)
    print()

    print()
    print('Training Evaluation A test task:')
    model_B.evaluate(A3, A4)
    print()
    
    
