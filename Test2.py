import numpy as np
from numpy import log as ln
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

#Compute the gradient and the Hessian for all functions and determine the set of stationary points 
#and characterize every stationary point whether it is a saddle point, (strict) local/global minimum or maximum

#1
def f(x, y):
    return (2*x**3. - 6*y**2. + 3*y*x**2.)

#2
def f2(x,y):
    return ((x-2*y)**4. + 64*x*y)

#3
def f3(x,y):
    return (2*x**2. + 3*y**2. - 2*x*y + 2*x - 3*y)

#4
def f4(x,y):
    return (ln(1 + 0.5*(x**2. + 3*y**2.)))

x = np.linspace(-0.1, 0.1, 100)
y = np.linspace(-0.1, 0.1, 100)





X, Y = np.meshgrid(x, y)
Z = f(X, Y)
#Z2 = f2(X, Y)
#Z3 = f3(X, Y)
#Z4 = f4(X, Y)

plt.contour(X, Y, Z, colors='black')
#plt.contour(X, Y, Z2, colors='black')
#plt.contour(X, Y, Z3, colors='black')
#plt.contour(X, Y, Z4, colors='black')


"""
fig = plt.figure()
ax = plt.axes(projection='2d')
ax.contourf(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')
"""
plt.savefig("Prva.png")
plt.show()


#gradient of f1
gradF = np.gradient(f(x,y))

print (f(x,y))
print (gradF)
