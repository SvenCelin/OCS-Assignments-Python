import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numdifftools as nd

dtype = np.float64
rng = np.random.RandomState(42)

def init_normalDistribution():
    N_H = 4 #number of hidden layers

    mu, sigma = 0, 0.05 # mean and standard deviation

    w0_vector = np.random.normal(mu, sigma, N_H*4)
    w1_vector = np.random.normal(mu, sigma, 4*N_H)
    b0 = np.random.normal(mu, sigma, N_H*1)
    b1 = np.random.normal(mu, sigma, 4*1)

    w0 = [[0 for x in range(4)] for y in range(N_H)]
    k = 0
    for i in range(N_H):
        for j in range(4):
            w0[i][j] = w0_vector[k]
            k += 1
    #print (w0)

    w1 = [[0 for x in range(N_H)] for y in range(4)]
    k = 0
    for i in range(N_H):
        for j in range(4):
            w1[i][j] = w1_vector[k]
            k += 1
    #print (w1)

    #print (b0)
    #print (b1)

    w = np.stack((w0, w1))
    
    print("KAKO NAM IZGLEDA W ", w)
    
    #return w0, w1, b0, b1


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
        print("temp = ", temp)
        if(temp > y):
            y = temp
    
    print("Y of softmax = ", y)
    #NEKAKO VRACA 1 KADA NE SMIJE
    return y
    

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
         rng.rand(4, N_H).astype(dtype)]  # W1
    b = [rng.rand(N_H,1).astype(dtype),     # b0
         rng.rand(4,1).astype(dtype)]     # b1
    return W, b


def d_sigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def d_lnAct(x):
    return np.exp(x)/(1+np.exp(x))


def d_softmax(X):
    X = np.asarray(X).flatten()
    print("X SHAPE: ", X.shape)
    sum = np.sum(np.exp(X))
    sumSq = sum*sum
    
    h = [ [ 0 for i in range(len(X))] for j in range(len(X)) ] 
    #h = [None] * len(X)

    for i in range(len(X)):
        for j in range(len(X)):
            if(i != j):
                #h[i][j] = -softMax(X)[i]*(softmax(X)[j])
                h[i][j] = -np.exp(X[j])*np.exp(X[i])/sumSq
            else:
                #h[i][j] = softMax(X)[i]*(1 - softmax(X)[i])
                h[i][j] = (np.exp(X[j])*sum - np.exp(X[j])*np.exp(X[j]))/sumSq

    return h


def d_loss(y,y_tilde):

    print("y SHAPE: ", np.asarray(y).shape)
    print("y_tilde SHAPE: ", np.asarray(y_tilde).shape)
    return -(y/y_tilde)


def back_prop(y, W, b, d_act, a, z):
    # compute errors e in reversed order
    print("BACK PROPAGATION")
    
    #y = y[:, np.newaxis]
    assert(len(a) == len(z))
    e = [None] * len(a)
    temp1 = d_act[-1](z[-1])
    temp2 = d_loss(y, a[-1])
    print("temp1 SHAPE: ", np.asarray(temp1).shape)
    print("temp2 SHAPE: ", np.asarray(temp2).shape)
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

             a, newResY = feed_forward(x, W, b, activations)
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
    
    
    activations = [lnAct,softMax]
    a, resultY = feed_forward(x, W, b, activations)

    d_act = [d_lnAct, d_softmax]
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




def drawScatter(x_test, y_test):

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


    #fig = plt.figure()
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
    activations = [lnAct, softMax]

    x = np.asarray(x_test)
    y = np.asarray(y_test)
    a, z = feed_forward(x[0], W, b, activations)

    inputY = [ [ 0 for i in range(4) ] for j in range(len(y)) ] 
    for i in range(len(y)):
        for j in range(4):
            index = int(y[i])
            inputY[i][index] = 1

    print("inputY = ", inputY)

    d_act = [d_lnAct, d_softmax]
    dW, db, e = back_prop(inputY[0], W, b, d_act, a, z)

    print("d_act: ", d_act)
    print("dw: ", dW)
    print("db: ", db)
    print("e: ", e)



    print("DULJINA X: ", len(x_test))


    drawScatter(x_test, y_test)

    init_normalDistribution()




