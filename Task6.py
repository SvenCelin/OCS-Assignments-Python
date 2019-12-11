import numpy as np
from scipy.optimize import approx_fprime
from numpy import linalg as LA

import matplotlib.pyplot as plt

import Task4
import Task5
dtype = np.float64

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

    
    loss = Loss(inY, resY)
    d_loss = dLoss(inY, resY)

    for k in range(K):
        for m_k in range(10):
             t_k = s*beta**m_k
             if(pos == 0):
                 W[0] = x_k - t_k*d_loss
             if(pos == 1):
                 W[1] = x_k - t_k*d_loss
             if(pos == 2):
                 b[0] = x_k - t_k*d_loss
             if(pos == 3):
                 b[1] = x_k - t_k*d_loss

             a, newResY = Task4.feed_forward(x, W, b, activations)
             newLoss = Loss(inY, resY)

             if  LA.norm(loss - newLoss) >= LA.norm(sigma*t_k*(d_loss**2)):
                 break
        x_k -= t_k * d_loss

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

def Loss(y,y_tilde):

    print("Loss, y SHAPE: ", np.asarray(y).shape)
    print("Loss, y_tilde SHAPE: ", np.asarray(y_tilde).shape)
    return -(y*np.log(y_tilde))
    
def dLoss(y,y_tilde):

    print("DLoss, y SHAPE: ", np.asarray(y).shape)
    print("DLoss, y_tilde SHAPE: ", np.asarray(y_tilde).shape)
    return -(y/y_tilde)

    """
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
    e[-1] = d_act[-1](z[-1]) @ d_loss(y, a[-1]) # delta_L
    for l in range(len(a) - 2, 0, -1):
        e[l] = d_act[l-1](resY[l]) * (W[l].T @ e[l+1])
            
    # compute gradient for W an b
    dW = [None] * len(a)
    db = [None] * len(a)
    for l in range(len(a) - 1):
        dW[l] = np.outer(e[l+1], a[l])
        db[l] = e[l+1]
        
    return dW, db
    """

def train(x, W, b, y, K, constantStep):
    
    inW0 = W[0]
    inW1 = W[1]
    inb0 = b[0]
    inb1 = b[1]
    sumConstant = 0
    sumArmijo = 0
    for i in range(len(y)):
        activations = [Task4.lnAct, Task4.softMax]
        a, resultY = Task4.feed_forward(x, W, b, activations)

        d_act = [Task5.d_lnAct, Task5.d_softmax]
        dW, db, e = Task5.back_prop(y[i], W, b, d_act, a, resultY)

        W_constant0 = steepestDescent(inW0, K, dW[0], False, 0, x, W, b, activations, y[i])
        W_constant1 = steepestDescent(inW1, K, dW[1], False, 1, x, W, b, activations, y[i])
        b_constant0 = steepestDescent(inb0, K, db[0], False, 2, x, W, b, activations, y[i])
        b_constant1 = steepestDescent(inb1, K, db[1], False, 3, x, W, b, activations, y[i])

        W_armijo0 = steepestDescent(inW0, K, dW[0], True, 0, x, W, b, activations, y[i])
        W_armijo1 = steepestDescent(inW1, K, dW[1], True, 1, x, W, b, activations, y[i])
        b_armijo0 = steepestDescent(inb0, K, db[0], True, 2, x, W, b, activations, y[i])
        b_armijo1 = steepestDescent(inb1, K, db[1], True, 3, x, W, b, activations, y[i])

        W[0] = W_constant0
        W[1] = W_constant1
        b[0] = b_constant0
        b[1] = b_constant1
        
        a, resultY = Task4.feed_forward(x, W, b, activations)
        sumConstant += compareError(y[i], resultY)

        W[0] = W_armijo0
        W[1] = W_armijo1
        b[0] = b_armijo0
        b[1] = b_constant1
        
        a, resultY = Task4.feed_forward(x, W, b, activations)
        sumArmijo += compareError(y[i], resultY)

    sumConstant /= len(y)
    sumArmijo /= len(y)
        """
    activations = [Task4.lnAct, Task4.softMax]
    a, resultY = Task4.feed_forward(x, W, b, activations)

    d_act = [Task5.d_lnAct, Task5.d_softmax]
    dW, db, e = Task5.back_prop(y[0], W, b, d_act, a, resultY)


    W_constant0 = steepestDescent(W[0], K, dW[0], False, 0, x, W, b, activations, y)
    W_constant1 = steepestDescent(W[1], K, dW[1], False, 1, x, W, b, activations, y)
    b_constant0 = steepestDescent(b[0], K, db[0], False, 2, x, W, b, activations, y)
    b_constant1 = steepestDescent(b[1], K, db[1], False, 3, x, W, b, activations, y)

    W_armijo0 = steepestDescent(W[0], K, dW[0], True, 0, x, W, b, activations, y)
    W_armijo1 = steepestDescent(W[1], K, dW[1], True, 1, x, W, b, activations, y)
    b_armijo0 = steepestDescent(b[0], K, db[0], True, 2, x, W, b, activations, y)
    b_armijo1 = steepestDescent(b[1], K, db[1], True, 3, x, W, b, activations, y)
    """

def compareError(y, y_tilde):
    if(y == y_tilde):
        return 1
    else:
        return 0