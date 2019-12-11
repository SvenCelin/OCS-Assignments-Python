import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt

import Task4
import Task5
dtype = np.float64

nabla_f = lambda x: np.where(x > 1, 3*x/2 + 1/2, np.where(x < -1, 3*x/2 - 1/2, 2*x))
f = lambda x: np.where(np.abs(x) > 1, 3*(1-np.abs(x))**2/4 - 2*(1-np.abs(x)), x**2-1)

def armijo(K, x_k, pos, x, W, b, activations, inY):
    s = 1.
    sigma = 0.1
    beta = 0.5
    if(pos == 0):
        W[0] = x_k
    if(pos == 1):
        W[1] = x_k
    if(pos == 2):
        b[0] = x_k
    if(pos == 3):
        b[1] = x_k
    a, resY = Task4.feed_forward(x, W, b, activations)

    
    loss = Loss(inY, resY, 4, len(resY))
    d_loss = dLoss(inY, resY, 4, len(resY))

    for k in range(K):
        grad = nabla_f(x_k)
        for m_k in range(10):
             t_k = s*beta**m_k
             if(pos == 0):
                 W[0] = x_k - t_k*grad
             if(pos == 1):
                 W[1] = x_k - t_k*grad
             if(pos == 2):
                 b[0] = x_k - t_k*grad
             if(pos == 3):
                 b[1] = x_k - t_k*grad

             a, newResY = Task4.feed_forward(x, W, b, activations)
             newLoss = Loss(inY, resY, 4, len(resY))

             if  loss - newLoss >= sigma*t_k*(d_loss**2).sum():
                 break
        x_k -= t_k * grad

    return t_k
        #update_plots(k, x_k, t_k)

def steepestDescent(x_k, K, d, constantStep, pos, x, W, b, activations, inY):
    print("x_k shape: ", x_k.shape)
    print("d shape: ", d.shape)
    t = 0.01
    for k in range(K):
        if(constantStep == False):
            t = armijo(K, x_k, pos, x, W, b, activations, inY)
        x_k += t * d
    return x_k


def Loss(inY, resY, N, S):
    sum = 0
    for i in S:
        for j in N:
             sum -= inY[i][j]*np.log(resY[i][j])
    return sum/S
    
def dLoss(inY, resY, N, S):
    sum = 0
    for i in S:
        for j in N:
             sum -= inY[i][j]/(resY[i][j])
    return sum/S
    
def deriveP(inY, resY, S, W, b, d_act, a):
    # compute errors e in reversed order
    print("BACK PROPAGATION")
    assert(len(a) == len(resY))
    e = [None] * len(a)
    e[-1] = d_act[-1](resY[-1]) * dLoss(inY, resY, 4, S) # delta_L
    for l in range(len(a) - 2, 0, -1):
        e[l] = d_act[l-1](resY[l]) * (W[l].T @ e[l+1])
            
    # compute gradient for W an b
    dW = [None] * len(a)
    db = [None] * len(a)
    for l in range(len(a) - 1):
        dW[l] = np.outer(e[l+1], a[l])
        db[l] = e[l+1]
        
    return dW, db

def train(x, W, b, y, K, constantStep):
    inputY = [ [ 0 for i in range(4) ] for j in range(len(y)) ] 
    i = 0
    for elem in y:
        inputY[i][elem] = 1
        i+=1
    
    activations = [Task4.lnAct, Task4.softMax]
    a, resultY = Task4.feed_forward(x, W, b, activations)

    d_act = [Task5.d_lnAct, Task5.d_softmax]
    dW, db = deriveP(inputY, resultY, len(y), W, b, d_act, a)


    W_constant0 = steepestDescent(W[0], K, dW[0], False, 0, x, W, b, activations, inputY)
    W_constant1 = steepestDescent(W[1], K, dW[1], False, 1, x, W, b, activations, inputY)
    b_constant0 = steepestDescent(b[0], K, db[0], False, 2, x, W, b, activations, inputY)
    b_constant1 = steepestDescent(b[1], K, db[1], False, 3, x, W, b, activations, inputY)

    W_armijo0 = steepestDescent(W[0], K, dW[0], True, 0, x, W, b, activations, inputY)
    W_armijo1 = steepestDescent(W[1], K, dW[1], True, 1, x, W, b, activations, inputY)
    b_armijo0 = steepestDescent(b[0], K, db[0], True, 2, x, W, b, activations, inputY)
    b_armijo1 = steepestDescent(b[1], K, db[1], True, 3, x, W, b, activations, inputY)

def compareError(S, armijoErr, standardErr):
    sum = 0
    for s in S:
        if(armijoErr[s] == standardErr[s]):
            sum+=1
    return sum/S