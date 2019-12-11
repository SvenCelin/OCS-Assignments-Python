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
"""
def d_softmax(X):
    sum = np.sum(np.exp(X))
    
    h = [None] * len(X)
    i = 0

    for x in X:
        res = np.exp(x)*sum - np.exp(x)*np.exp(x)
        res = res / (sum*sum)
        h[i] = res
        i+=1
    return h
    """

def d_softmax(X):
    sum = np.sum(np.exp(X))
    sumSq = sum*sum
    
    h = [ [ 0 for i in len(X) ] for j in range(len(X)) ] 
    #h = [None] * len(X)

    for i in len(X):
        for j in len(X):
            if(i != j):
                h[i][j] = -np.exp(X[j])*np.exp[i]/sumSq
            else:
                h[i][j] = (np.exp(X[j])*sum - np.exp(X[j])*np.exp(X[j]))/sumSq

    return h

#def d_loss(y, y_tilde):
#    return -y/y_tilde + (1-y)/(1-y_tilde)

def d_loss(y,dy, N_O, S):
    sum = 0
    for i in S:
        for j in N_O:
            if(dy[i][j] == 0):
                print("DIVIDE BY ZERO!!!!")
            else:
                sum += y[i][j]/dy[i][j]
    return sum/S

def back_prop(y, W, b, d_act, a, z):
    # compute errors e in reversed order
    print("BACK PROPAGATION")
    assert(len(a) == len(z))
    e = [None] * len(a)
    e[-1] = d_act[-1](z[-1]) * d_loss(y, z, len(y), 1) # delta_L
    for l in range(len(a) - 2, 0, -1):
        e[l] = d_act[l-1](z[l]) * (W[l].T @ e[l+1])
            
    # compute gradient for W an b
    dW = [None] * len(a)
    db = [None] * len(a)


    for l in range(len(a) - 1):
        dW[l] = np.outer(e[l+1], a[l])
        db[l] = e[l+1]
        
    return dW, db, e