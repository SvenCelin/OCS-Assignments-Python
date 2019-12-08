import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt

import Task4

dtype = np.float64

def d_sigmoid(x):
    sig = Task4.sigmoid(x)
    return sig*(1-sig)

def d_lnAct(x):
    return np.exp(x)/(1+np.exp(x))

def d_softmax(X):
    sum = np.sum(np.exp(X))
    
    h = [None] * len(X)
    i = 0

    for x in X:
        res = np.exp(x)*sum + np.exp(x)*np.exp(x)
        res /= np.square(sum)
        h[i] = res
        i+=1
    return h


def d_loss(y, y_tilde):
    return -y/y_tilde + (1-y)/(1-y_tilde)

def back_prop(y, W, b, d_act, a, z):
    # compute errors e in reversed order
    print("BACK PROPAGATION")
    assert(len(a) == len(z))
    e = [None] * len(a)
    e[-1] = d_act[-1](z[-1]) * d_loss(y, a[-1]) # delta_L
    for l in range(len(a) - 2, 0, -1):
        e[l] = d_act[l-1](z[l]) * (W[l].T @ e[l+1])
            
    # compute gradient for W an b
    dW = [None] * len(a)
    db = [None] * len(a)
    for l in range(len(a) - 1):
        dW[l] = np.outer(e[l+1], a[l])
        db[l] = e[l+1]
        
    return dW, db, e