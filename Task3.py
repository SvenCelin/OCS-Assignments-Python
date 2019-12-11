import numpy as np
import matplotlib.pyplot as plt

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
print (w0)

w1 = [[0 for x in range(N_H)] for y in range(4)]
k = 0
for i in range(N_H):
    for j in range(4):
        w1[i][j] = w1_vector[k]
        k += 1
print (w1)

print (b0)
print (b1)


