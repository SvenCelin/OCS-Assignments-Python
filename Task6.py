import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt

import Task4
import Task5
dtype = np.float64

nabla_f = lambda x: np.where(x > 1, 3*x/2 + 1/2, np.where(x < -1, 3*x/2 - 1/2, 2*x))
f = lambda x: np.where(np.abs(x) > 1, 3*(1-np.abs(x))**2/4 - 2*(1-np.abs(x)), x**2-1)

def armijo(K, x_k):
    s = 1.
    sigma = 0.1
    beta = 0.5
    for k in range(K):
        grad = nabla_f(x_k)
        for m_k in range(10):
             t_k = s*beta**m_k
             temp = f(x_k) - f(x_k - t_k*grad)
             temp2 = sigma*t_k*(grad**2).sum()

             #print("temp1 type: ", type(temp))
             #print("temp2 type: ", type(temp2))
             #if  f(x_k) - f(x_k - t_k*grad) >= sigma*t_k*(grad**2).sum():
             if  temp.sum() >= sigma*t_k*(grad**2).sum():
                 break
        x_k -= t_k * grad

    return t_k
        #update_plots(k, x_k, t_k)

def steepestDescent(x_k, K, d, constantStep):
    print("x_k shape: ", x_k.shape)
    print("d shape: ", d.shape)
    t = 0.01
    for k in range(K):
        if(constantStep == False):
            t = armijo(K, x_k)
        x_k += t * d
    return x_k

def calcD_Loss(inY, resY, N, S):
    sum = 0
    for i in S:
        for j in N:
            if(resY[i][j] == 0):
                print("DIVIDE BY ZERO!!!!")
            else:
                sum += inY[i][j]/np.log(resY[i][j])
    return sum/S
    
def deriveP(inY, resY, S, W, b, d_act, a):
    # compute errors e in reversed order
    print("BACK PROPAGATION")
    assert(len(a) == len(resY))
    e = [None] * len(a)
    e[-1] = d_act[-1](resY[-1]) * calcD_Loss(inY, resY, 4, S) # delta_L
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

    W[0] = steepestDescent(W[0], K, dW[0], False)
    W[1] = steepestDescent(W[1], K, dW[1], False)
    b[0] = steepestDescent(b[0], K, db[0], False)
    b[1] = steepestDescent(b[1], K, db[1], False)

def compareError(S, armijoErr, standardErr):
    sum = 0
    for s in S:
        if(armijoErr[s] == standardErr[s]):
            sum+=1
    return sum/S