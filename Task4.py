import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt

rng = np.random.RandomState(42)

dtype = np.float64

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

#a= ln(1 + exp(z))
def lnAct(x):
    return np.log(1 + np.exp(x))

def softMax(X):
    y = 0

    for numer in X:
        sum = 0
        for elem in X:
            sum += np.exp(elem)
        temp = np.exp(numer)/sum
        if(temp > y):
            y = temp
    
    print("Y of softmax = ", y)
    return y
    """

    exps = np.exp(X)
    exps = exps/ np.sum(exps)

    y = 0
    for x in exps:
        if(x > y):
            y = x
    print("Y of softmax = ", y)
    return y
    
    """
    

def feed_forward(x, W, b, act):
    # convert to column vector
    x = x[:, np.newaxis]
    a = [x]
    z = [x]
    print("FEED FORWARD")
    print("x = ", x)
    print("a = ", a)
    print("z = ", z)
    print("W = ", W)
    print("b = ", b)
    for i, (Wi, bi, acti) in enumerate(zip(W, b, act)):
        print("one iteration, i = ", i)
        z.append(Wi @ a[i] + bi)
        a.append(acti(z[-1]))
    
    return a, z

def loss(y, y_tilde):
    return -(y*np.log(y_tilde) + (1-y)*np.log(1-y_tilde))

def init_params(N_H=4):
    # initialize parameters
    W = [rng.rand(N_H, 3).astype(dtype),  # W0
         rng.rand(1, N_H).astype(dtype)]  # W1
    b = [rng.rand(N_H,1).astype(dtype),     # b0
         rng.rand(1,1).astype(dtype)]     # b1
    return W, b