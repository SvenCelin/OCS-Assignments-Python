import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt

import Task4
import Task5
import Task6

dtype = np.float64

if __name__ == '__main__':
    # load the data set
    data_set = np.load('./data.npz')
    # get the training data
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    # get the test data
    x_test = data_set['x_test']
    y_test = data_set['y_test']

    print('Training data:', x_train.shape, y_train.shape)
    print('Test data:', x_test.shape, y_test.shape)
    
    W, b = Task4.init_params()
    activations = [Task4.sigmoid, Task4.sigmoid]

    x = np.asarray(x_test)
    y = np.asarray(y_test)

    print('W:', W)
    print('b:', b)
    
    a, z = Task4.feed_forward(x[1], W, b, activations)

    print("a: ", a)
    print("z: ", z)

    d_act = [Task5.d_sigmoid, Task5.d_sigmoid]
    dW, db, e = Task5.back_prop(y[1], W, b, d_act, a, z)

    print("d_act: ", d_act)
    print("dw: ", dW)
    print("db: ", db)
    print("e: ", e)

    W_array = np.array(W)
    b_array = np.array(b)
    print("W type: ", type(W))
    print("Warray type: ", type(W_array))
    print("W shape: ", W_array.shape)

    print("W[0]: ", W_array[0])


    
    WTrainStandard = [None] * len(W)
    bTrainStandard = [None] * len(b)

    
    print("x type: ", type(x))
    print("dW[0] type: ", type(dW[0]))
    print("dW[1] type: ", type(dW[1]))

    print("x shape: ", x.shape)
    print("dW[0] shape: ", dW[0].shape)
    print("dW[1] shape: ", dW[1].shape)

    for x in W_array[0]:
        WTrainStandard[0] = Task6.steepestDescent(W_array[0], 1000, 0.05, dW[0], True) 
    for x in W_array[1]:
        WTrainStandard[1] = Task6.steepestDescent(W_array[1], 1000, 0.05, dW[1], True) 
    for x in b_array[0]:
        bTrainStandard[0] = Task6.steepestDescent(b_array[0], 1000, 0.05, db[0], True) 
    for x in b_array[1]:
        bTrainStandard[1] = Task6.steepestDescent(b_array[1], 1000, 0.05, db[1], True) 


    WTrainArmijo = [None] * len(W)
    bTrainArmijo = [None] * len(b)

    for x in W_array[0]:
        WTrainArmijo[0] = Task6.steepestDescent(W_array[0], 1000, 0.05, dW[0], False) 
    for x in W_array[1]:
        WTrainArmijo[1] = Task6.steepestDescent(W_array[1], 1000, 0.05, dW[1], False) 
    for x in b_array[0]:
        bTrainArmijo[0] = Task6.steepestDescent(b_array[0], 1000, 0.05, db[0], False) 
    for x in b_array[1]:
        bTrainArmijo[1] = Task6.steepestDescent(b_array[1], 1000, 0.05, db[1], False) 
    

