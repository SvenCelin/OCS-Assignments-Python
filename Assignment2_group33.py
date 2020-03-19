import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numdifftools as nd

import Task6
import Task4
import Task5

dtype = np.float64
rng = np.random.RandomState(42)

def initNormalDistribution():
    N_H = 4

    mu, sigma = 0, 0.05 # mean and standard deviation

    w0_vector = np.random.normal(mu, sigma, N_H*4)
    w1_vector = np.random.normal(mu, sigma, 4*N_H)
    b0 = np.random.normal(mu, sigma, N_H*1)
    b1 = np.random.normal(mu, sigma, 4*1)

    w0 = [[0 for x in range(4)] for y in range(N_H)]
    k = 0
    for i in range(N_H):
        for j in range(4):
            w0[i][j] = w0_vector[k]
            k += 1


    w1 = [[0 for x in range(N_H)] for y in range(4)]
    k = 0
    for i in range(N_H):
        for j in range(4):
            w1[i][j] = w1_vector[k]
            k += 1

    return w0, w1, b0, b1

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softPlus(x):
    return np.log(1 + np.exp(x))

def softMax(X):
    exps = np.exp(X)/np.sum(np.exp(X))
    return exps 

def feed_forward(x, W, b, act):
    x = x[:, np.newaxis]
    a = [x]
    z = [x]
    for i, (Wi, bi, acti) in enumerate(zip(W, b, act)):
        z.append(Wi @ a[i] + bi)
        a.append(acti(z[-1]))
    return a, z

def loss(y, y_tilde):
    return -(y*np.log(y_tilde) + (1-y)*np.log(1-y_tilde))

def init_params(N_H=4):
    W = [rng.rand(N_H, 3).astype(dtype),  # W0
         rng.rand(4, N_H).astype(dtype)]  # W1
    b = [rng.rand(N_H,1).astype(dtype),     # b0
         rng.rand(4,1).astype(dtype)]     # b1
    return W, b

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def d_softPlus(x):
    return np.exp(x)/(1+np.exp(x))

def d_softMax(X):
    temp = softMax(X)
    result = np.zeros((len(X), len(X)))
    i = 0
    j = 0
    while(i < len(X)):
        while (j < len(X)):
            if(i == j):
                result[i, j] = temp[i]*(1-temp[j])
            else:
                result[i, j] = -temp[i]*temp[j]
            j += 1
        i += 1
    return result

def d_loss(y,y_tilde):
    return -(y/y_tilde)

def backwards(y, W, b, d_act, a, z):
    
    temp_e = [None] * len(a)
    dW = [None] * len(a)
    db = [None] * len(a)
    y = y[:, np.newaxis]
    assert(len(a) == len(z)) 
    temp_e[-1] = d_act[-1](z[-1]) @ d_loss(y, a[-1])
    for i in range(len(a) - 2, 0, -1):
        temp_e[i] = d_act[i-1](z[i]) * (W[i].T @ temp_e[i+1])
    i = 0
    while (i < (len(a) - 1)):
        dW[i] = np.outer(temp_e[i+1], a[i])
        db[i] = temp_e[i+1]
        i += 1
    return dW, db, temp_e

def steepestDescent(x_k, K, d, pos, x, W, b, activations, inY):
    t = 0.05
    i = 0
    while (i < K):
        x_k += t * d
        i += 1
    return x_k

def train(x, W, b, y_test, y_train, K):
    W0 = W[0]
    W1 = W[1]
    b0 = b[0]
    b1 = b[1]
    i = 0
    accuracy = 0
    while ( i < len(y_train)):
        activations = [softPlus, softMax]
        a, y_tilde = feed_forward(x[i], W, b, activations)
        d_act = [d_softPlus, d_softMax]
        dW, db, e = backwards(y_train[i], W, b, d_act, a, y_tilde)
        trainingW0 = steepestDescent(W0, K, dW[0], 0, x[i], W, b, activations, y_train[i])
        trainingW1 = steepestDescent(W1, K, dW[1], 1, x[i], W, b, activations, y_train[i])
        training_b0 = steepestDescent(b0, K, db[0], 2, x[i], W, b, activations, y_train[i])
        training_b1 = steepestDescent(b1, K, db[1], 3, x[i], W, b, activations, y_train[i])

        W[0] = trainingW0
        W[1] = trainingW1
        b[0] = training_b0
        b[1] = training_b1
        
        a, y_tilde = feed_forward(x[i], W, b, activations)

        accuracy+= checkAccuracy(y_test, y_tilde)
        i += 1
    accuracy /= len(y_train)
  
def checkAccuracy(y, y_tilde):
    if(y == y_tilde):
        return 1
    else:
        return 0

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

    # y2 should be our predicted label
    y_2 = y_test
    amarker = [None] * len(y_test)
    for i in range (0, len(y_test), 1):
        if(y_test[i] == y_2[i]):
            amarker[i] = '+'
        if(y_test[i] != y_2[i]):
            amarker[i] = '_'

    ax = plt.axes(projection='3d')
    ax.scatter(x_x, x_y, x_z, c=acolor, cmap='viridis', linewidth=0.5)
    for i in range(len(y_test)):
        ax.scatter(x_x[i], x_y[i], x_z[i], c="black", marker=amarker[i], cmap='viridis', linewidth=0.5)

    plt.show()

if __name__ == '__main__':
    data_set = np.load('./data.npz')
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    x_test = data_set['x_test']
    y_test = data_set['y_test']
    
    W, b = init_params()

    y_train_mat = np.zeros((len(y_train), 4))
    y_test_mat = np.zeros((len(y_test), 4))

    i = 0
    j = 0
    while(i < len(y_train)):
        while (j < 4):
            index = int(y_train[i])
            y_train_mat[i][index] = 1
            j += 1
        i += 1
    
    i = 0
    j = 0
    while(i < len(y_train)):
        while (j < 4):
            index = int(y_train[i])
            y_train_mat[i][index] = 1
            j += 1
        i += 1


    train(x_train, W, b, y_test_mat, y_train_mat, 3)

    drawPlot(x_train, y_train)




