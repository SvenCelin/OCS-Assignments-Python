import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
import matplotlib.pyplot as plt
import time

def compute_func_grad(x, p_1, p_2, d):
    m = d.size
    g = np.zeros(m)
    nabla_g = np.zeros((2,m))
    for i in range(0,m):
        t = np.sqrt((x[0]-p_1[i])**2 + (x[1]-p_2[i])**2)
        g[i] = t-d[i]
        nabla_g[0,i] = (x[0]-p_1[i])/t
        nabla_g[1,i] = (x[1]-p_2[i])/t
    
    f = 0.5*np.linalg.norm(g)**2
    return (f,g,nabla_g) 

# My position
x = np.array([0.5, 0.25])

# Number beacons
m = 10

# Exact positions of beacons
p_1 = (np.random.rand(m)-0.5)*2
p_2 = (np.random.rand(m)-0.5)*2

# Noise level of measurements
sigma = 1

# distance measurements
dist = np.sqrt((x[0]-p_1)**2 + (x[1]-p_2)**2) + sigma*np.random.randn(m)

# generate 2D grid
x_2,x_1 = np.mgrid[-2:2:0.01, -2:2:0.01]
f_all = np.zeros_like(x_1)
for i in range(0,m):
    f_all += 0.5*(np.sqrt((x_1-p_1[i])**2 + (x_2-p_2[i])**2)-dist[i])**2
    
plt.figure(1, figsize=(7,7))
fig1 = plt.gcf()

# draw contour lines of f
levels = np.power(2.0, np.arange(-2,10,1))
plt.contour(x_1, x_2, f_all, levels)
plt.plot(x[0], x[1], '*', markersize=10, color='red')


for i in range(0,m):
    plt.plot(p_1[i], p_2[i], '+', markersize=10, color='blue')
    
    
# Gauss-Newton algorithm    
maxiter = 50
delta = 0.01
x = np.array([-2,-2])
t = 1.0
for iter in range(0,maxiter):
    f,g,nabla_g=compute_func_grad(x,p_1,p_2,dist)
    D = inv(nabla_g@nabla_g.T + delta*np.eye(2,2))
    d = -D@(nabla_g@g)
    
    x_old = x
    x = x + t*d
    
    #print("iter = ", iter, ", t = ", t, ", x = ", x, ", f = ", f)
            
    plt.plot((x_old[0],x[0]), (x_old[1], x[1]), linewidth=2.0, color="black")
    plt.plot(x[0],x[1],"*", color="black", markersize=7)
    fig1.canvas.draw()
    
    time.sleep(0.001)
plt.show()