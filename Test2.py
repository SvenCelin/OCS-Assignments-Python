import numpy as np
from numpy import log as ln
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

#Compute the gradient and the Hessian for all functions and determine the set of stationary points 
#and characterize every stationary point whether it is a saddle point, (strict) local/global minimum or maximum

#1
def f(x, y):
    return (2*(x**3) - 6*(y**2) + 3*y*(x**2))

#2
def f2(x,y):
    return ((x-2*y)**4. + 64*x*y)

#3
def f3(x,y):
    return (2*(x**2) + 3*(y**2) - 2*x*y + 2*x - 3*y)

#4
def f4(x,y):
    return (ln(1 + 0.5*(x**2. + 3*(y**2))))

def determinantX(func, x, y):
    return (func(x + 0.001, y) - func(x - 0.001, y))/0.002

def determinantY(func, x, y):
    return (func(x, y + 0.001) - func(x, y - 0.001))/0.002


def dFx(x, y):
    return 6*x**2 + 6*x*y

def dFy(x, y):
    return -12*y + 3*(x**2)

def dF2x(x, y):
    return 4*((x-2*y)**3) + 64*y

def dF2y(x, y):
    return -8*((x-2*y)**3) + 64*x

def dF3x(x, y):
    return 4*x-2*y+2

def dF3y(x, y):
    return 6*y-2*x-3

def dF4x(x, y):
    return 2*x/(2 + x**2 + 3*(y**2))

def dF4y(x, y):
    return 6*y/(2 + x**2 + 3*(y**2))

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

x = np.random.rand(1)
y = np.random.rand(1)

print("x = ")
print(x)
print("y = ")
print(y)
print("determinantX = ")
print(determinantX(f, x, y))
print("determinantY = ")
print(determinantY(f, x, y))
print("dFx = ")
print(dFx(x, y))
print("dFy = ")
print(dFy(x, y))

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
#Z = f2(X, Y)
#Z = f3(X, Y)
#Z = f4(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')

plt.show()  #***REMOVE COMMENT***

#gradient of f1
#print("New Print")
#gradF = np.gradient(f(x,y), x)

#print (f(x,y))
#print (gradF)


#print("New Print")
#gradF = np.gradient(f(x,y))

#print (f(x,y))
#print (gradF)
