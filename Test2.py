import numpy as np
from numpy import log as ln
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

#nis
#test2
def g(x):
    return (x**2. * np.sin(20*x) + np.cos(x-np.pi))**3 + 1

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



x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)


X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')

#plt.show()  REMOVE COMMENT

#gradient of f
gradF = np.gradient(f(x,y))

print (f(x,y))
print (gradF)
