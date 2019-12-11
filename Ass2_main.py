import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numdifftools as nd

import Task6
import Task4
import Task5

dtype = np.float64

def drawPlot(x_train, y_train):
    x_x = [None] * len(x_test)
    x_y = [None] * len(x_test)
    x_z = [None] * len(x_test)

    for i in range (0, len(x_test), 1):
        x_x[i] = x_test[i][0]
        x_y[i] = x_test[i][1]
        x_z[i] = x_test[i][2]

    acolor = [None] * len(y_test)
    for i in range (0, len(y_test), 1):
        if(y_test[i] == 0):
            acolor[i] = "red"
        if(y_test[i] == 1):
            acolor[i] = "blue"
        if(y_test[i] == 2):
            acolor[i] = "green"
        if(y_test[i] == 3):
            acolor[i] = "yellow"

    y_test2 = y_test
    amarker = [None] * len(y_test)
    for i in range (0, len(y_test), 1):
        if(y_test[i] == y_test2[i]):
            amarker[i] = '+'
        if(y_test[i] != y_test2[i]):
            amarker[i] = '_'

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_x, x_y, x_z, c=acolor, cmap='viridis', linewidth=0.5)
    for i in range(len(y_test)):
        ax.scatter(x_x[i], x_y[i], x_z[i], c="black", marker=amarker[i], cmap='viridis', linewidth=0.5)

    plt.show()

if __name__ == '__main__':
    # load the data set
    data_set = np.load('./data.npz')
    # get the training data
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    # get the test data
    x_test = data_set['x_test']
    y_test = data_set['y_test']
    
    W, b = Task4.init_params()

    y_train_mat = np.zeros((len(y_train), 4))
    #y_train_mat = [ [ 0 for i in range(4) ] for j in range(len(y_train)) ] 
    for i in range(len(y_train)):
        for j in range(4):
            index = int(y_train[i])
            y_train_mat[i][index] = 1

    print("y_train_mat = ", y_train_mat)

    y_test_mat = np.zeros((len(y_test), 4))
    #y_test_mat = [ [ 0 for i in range(4) ] for j in range(len(y_test)) ] 
    for i in range(len(y_test)):
        for j in range(4):
            index = int(y_test[i])
            y_test_mat[i][index] = 1

    print("y_test_mat = ", y_test_mat)

    Task6.train(x_train, W, b, y_test_mat, y_train_mat, 3)

    drawPlot(x_train, y_train)




