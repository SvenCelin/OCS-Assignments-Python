import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt

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

def steepestDescent(x_k, K, t, d, constantStep):
    print("x_k shape: ", x_k.shape)
    print("d shape: ", d.shape)
    for k in range(K):
        x_k += t * d
        if(constantStep == False):
            t = armijo(K, x_k)
    return x_k

def compareError(S, armijoErr, standardErr):
    sum = 0
    for s in S:
        if(armijoErr[s] == standardErr[s]):
            sum+=1
    return sum/S