import numpy as np
from scipy.optimize import approx_fprime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numdifftools as nd

import Task6
import Task4
import Task5

dtype = np.float64
rng = np.random.RandomState(42)

#Task4
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














#task5
def d_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def d_lnAct(x):
    return np.exp(x)/(1+np.exp(x))

def d_softmax(X):
    pred = softMax(X)
    result = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        for j in range(len(X)):
            if(i == j):
                result[i, j] = pred[i]*(1-pred[j])
            else:
                result[i, j] = -pred[i]*pred[j]
    return result


def d_loss(y,y_tilde):
    return -(y/y_tilde)
    #return -(y/y_tilde) + (1+y)/(1-y_tilde)

def back_prop(y, W, b, d_act, a, z):
    # compute errors e in reversed order
    print("BACK PROPAGATION")
    
    y = y[:, np.newaxis]
    assert(len(a) == len(z))
    e = [None] * len(a)
    
    e[-1] = d_act[-1](z[-1]) @ d_loss(y, a[-1]) # delta_L
    for l in range(len(a) - 2, 0, -1):
        e[l] = d_act[l-1](z[l]) * (W[l].T @ e[l+1])
            
    # compute gradient for W an b
    dW = [None] * len(a)
    db = [None] * len(a)


    for l in range(len(a) - 1):
        dW[l] = np.outer(e[l+1], a[l])
        db[l] = e[l+1]
        
    return dW, db, e















#task6
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
    a, resY = feed_forward(x, W, b, activations)

    
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

             a, newResY = feed_forward(x, W, b, activations)
             newLoss = Loss(inY, resY)

             if  LA.norm(loss - newLoss) >= LA.norm(sigma*t_k*(d_loss**2)):
                 break
        x_k -= t_k * d_loss

    return t_k
        #update_plots(k, x_k, t_k)

def steepestDescent(x_k, K, d, constantStep, pos, x, W, b, activations, inY):
    print("DESCENT DOWN THE HILL")
    t = 0.01
    for k in range(K):
        if(constantStep == False):
            t = armijo(K, x_k, pos, x, W, b, activations, inY)
        x_k += t * d
    return x_k

def Loss(y,y_tilde):
    print("y_tilde shape: ", np.asarray(y_tilde).shape)
    for i in y_tilde:
        print (np.asarray(i)[:][0])
    temp1 = np.log(y_tilde)
    return -(y*temp1)
    
def dLoss(y,y_tilde):
    return -(y/y_tilde)

def train(x, W, b, y_test, y_train, K):
    
    print("TRAINING FOR THE WORLD MARTIAL ARTS TOURNAMENT")
    inW0 = W[0]
    inW1 = W[1]
    inb0 = b[0]
    inb1 = b[1]
    sumConstant = 0
    sumArmijo = 0
    for i in range(len(y_train)):

        activations = [lnAct, softMax]
        a, resultY = feed_forward(x[i], W, b, activations)

        d_act = [d_lnAct, d_softmax]
        dW, db, e = back_prop(y_train[i], W, b, d_act, a, resultY)
        
        print("CONSTANT STEP")
        W_constant0 = steepestDescent(inW0, K, dW[0], True, 0, x[i], W, b, activations, y_train[i])
        W_constant1 = steepestDescent(inW1, K, dW[1], True, 1, x[i], W, b, activations, y_train[i])
        b_constant0 = steepestDescent(inb0, K, db[0], True, 2, x[i], W, b, activations, y_train[i])
        b_constant1 = steepestDescent(inb1, K, db[1], True, 3, x[i], W, b, activations, y_train[i])

  
def compareError(y, y_tilde):
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

    y_test2 = y_test
    amarker = [None] * len(y_test)
    for i in range (0, len(y_test), 1):
        if(y_test[i] == y_test2[i]):
            amarker[i] = '+'
        if(y_test[i] != y_test2[i]):
            amarker[i] = '_'

    ax = plt.axes(projection='3d')
    ax.scatter(x_x, x_y, x_z, c=acolor, cmap='viridis', linewidth=0.5)
    for i in range(len(y_test)):
        ax.scatter(x_x[i], x_y[i], x_z[i], c="black", marker=amarker[i], cmap='viridis', linewidth=0.5)

    plt.show()





if __name__ == '__main__':
    # load the data set
    data_set = np.load('./data.npz')
    # get the training data
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    # get the test data
    x_test = data_set['x_test']
    y_test = data_set['y_test']
    
    W, b = init_params()

    y_train_mat = np.zeros((len(y_train), 4))
    #y_train_mat = [ [ 0 for i in range(4) ] for j in range(len(y_train)) ] 
    for i in range(len(y_train)):
        for j in range(4):
            index = int(y_train[i])
            y_train_mat[i][index] = 1

    print("y_train_mat = ", y_train_mat)

    y_test_mat = np.zeros((len(y_test), 4))
    #y_test_mat = [ [ 0 for i in range(4) ] for j in range(len(y_test)) ] 
    for i in range(len(y_test)):
        for j in range(4):
            index = int(y_test[i])
            y_test_mat[i][index] = 1

    print("y_test_mat = ", y_test_mat)

    train(x_train, W, b, y_test_mat, y_train_mat, 3)

    drawPlot(x_train, y_train)




