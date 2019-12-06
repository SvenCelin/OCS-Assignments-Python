import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt

rng = np.random.RandomState(42)

dtype = np.float64

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def feed_forward(x, W, b, act):
    # convert to column vector
    x = x[:, np.newaxis]
    a = [x]
    z = [x]
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

#BACKWARD
def d_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def d_loss(y, y_tilde):
    return -y/y_tilde + (1-y)/(1-y_tilde)

def back_prop(y, W, b, d_act, a, z):
    # compute errors e in reversed order
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

    #TRAINING
def nabla_f(x):
    nabla = lambda x: np.where(x > 1, 3*x/2 + 1/2, np.where(x < -1, 3*x/2 - 1/2, 2*x))
    return nabla

def func(x):
    f = lambda x: np.where(np.abs(x) > 1, 3*(1-np.abs(x))**2/4 - 2*(1-np.abs(x)), x**2-1)
    return f

def armijo(K, x_k):
    s = 1.
    sigma = 0.1
    beta = 0.5
    for k in range(K):
        grad = nabla_f(x_k)
        for m_k in range(10):
             t_k = s*beta**m_k
             if func(x_k) - func(x_k - t_k*grad) >= sigma*t_k*(grad**2).sum():
                 break
        x_k -= t_k * grad

    return t_k
        #update_plots(k, x_k, t_k)

def steepestDescent(x_k, K, t, d, constantStep):
    for k in range(K):
        x_k += t * d
        t = armijo(K, x_k)
    return x_k

def compareError(S, armijoErr, standardErr):
    sum = 0
    for s in S:
        if(armijoErr[s] == standardErr[s]):
            s+=1
    return s/S

if __name__ == '__main__':
    # load the data set
    data_set = np.load('./data.npz')
    # get the training data
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    # get the test data
    x_test = data_set['x_test']
    y_test = data_set['y_test']

    print('Training data:', x_train.shape, y_train.shape)
    print('Test data:', x_test.shape, y_test.shape)
    
    W, b = init_params()
    activations = [sigmoid, sigmoid]

    x = np.asarray(x_test)
    y = np.asarray(y_test)

    print('W:', W)
    print('b:', b)
    
    a, z = feed_forward(x[1], W, b, activations)

    print("a: ", a)
    print("z: ", z)

    d_act = [d_sigmoid, d_sigmoid]
    dW, db, e = back_prop(y[1], W, b, d_act, a, z)

    print("d_act: ", d_act)
    print("dw: ", dW)
    print("db: ", db)
    print("e: ", e)

    
    WTrainStandard = [None] * len(W)
    bTrainStandard = [None] * len(b)

    for x in W[0]:
        WTrainStandard[0] = steepestDescent(x, 1000, 0.001, dW[0], True) 
    for x in W[1]:
        WTrainStandard[1] = steepestDescent(x, 1000, 0.001, dW[1], True) 
    for x in b[0]:
        bTrainStandard[0] = steepestDescent(x, 1000, 0.001, db[0], True) 
    for x in b[1]:
        bTrainStandard[1] = steepestDescent(x, 1000, 0.001, db[1], True) 

    WTrainArmijo = [None] * len(W)
    bTrainArmijo = [None] * len(b)

    for x in W[0]:
        WTrainArmijo[0] = steepestDescent(x, 1000, 0.001, dW[0], False) 
    for x in W[1]:
        WTrainArmijo[1] = steepestDescent(x, 1000, 0.001, dW[1], False) 
    for x in b[0]:
        bTrainArmijo[0] = steepestDescent(x, 1000, 0.001, db[0], False) 
    for x in b[1]:
        bTrainArmijo[1] = steepestDescent(x, 1000, 0.001, db[1], False) 
    

