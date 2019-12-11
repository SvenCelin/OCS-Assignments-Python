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
    
    exps = np.exp(X)
    exps = exps/ np.sum(exps)
    return exps
    

def feed_forward(x, W, b, act):
    # convert to column vector
    x = x[:, np.newaxis]
    a = [x]
    z = [x]
    for i, (Wi, bi, acti) in enumerate(zip(W, b, act)):
        z.append(Wi @ a[i] + bi)
        a.append(acti(z[-1]))
    
    print("FORWARD")
    return a, z

def loss(y, y_tilde):
    return -(y*np.log(y_tilde) + (1-y)*np.log(1-y_tilde))

def init_params(N_H=4):
    # initialize parameters
    W = [rng.rand(N_H, 3).astype(dtype),  # W0
         rng.rand(4, N_H).astype(dtype)]  # W1
    b = [rng.rand(N_H,1).astype(dtype),     # b0
         rng.rand(4,1).astype(dtype)]     # b1
    return W, b