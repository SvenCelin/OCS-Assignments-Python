import numpy as np
from numpy import log as ln
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numdifftools as nd

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

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian



x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)


#X, Y = np.meshgrid(x, y)
#Z = f(X, Y)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.set_title('3D contour')

X, Y = np.meshgrid(x, y)
Z = f3(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')

plt.show()

#gradient of f
gradF = np.gradient(f(x,y))

print("Function:")
print (f(x,y))
print("Gradient:")
print (gradF)

# 2.zadatak
point = (0.5, 0.4)
#print(point[0])
eps = 0.001
x1 = 0.5
y1 = 0.4
#Function 1
result_x_f1 = (f(x1 + eps, y1) - f(x1 - eps, y1)) / (2.0 * eps)
result_y_f1 = (f(x1, y1 + eps) - f(x1, y1 - eps)) / (2.0 * eps)
print("Numerical approximation using central differences for function 1:")
print(result_x_f1)
print(result_y_f1)

#Function 2
result_x_f2 = (f2(x1 + eps, y1) - f2(x1 - eps, y1)) / (2.0 * eps)
result_y_f2 = (f2(x1, y1 + eps) - f2(x1, y1 - eps)) / (2.0 * eps)
print("Numerical approximation using central differences for function 2:")
print(result_x_f2)
print(result_y_f2)

#Function 3
result_x_f3 = (f3(x1 + eps, y1) - f3(x1 - eps, y1)) / (2.0 * eps)
result_y_f3 = (f3(x1, y1 + eps) - f3(x1, y1 - eps)) / (2.0 * eps)
print("Numerical approximation using central differences for function 3:")
print(result_x_f3)
print(result_y_f3)

#Function 4
result_x_f4 = (f4(x1 + eps, y1) - f4(x1 - eps, y1)) / (2.0 * eps)
result_y_f4 = (f4(x1, y1 + eps) - f4(x1, y1 - eps)) / (2.0 * eps)
print("Numerical approximation using central differences for function 4:")
print(result_x_f4)
print(result_y_f4)



#H = nd.Hessian(f)([1,1])
#hessian = H([1,1])
#print("Hessian")
#print(H)

#H = hessian(f)


c = [[0.5, 0.75], 
        [0.25, 1.0], 
        [0.25, 0.75], 
        [0.25, 0.5], 
        [1.0, 0.5], 
        [1.0, 0.25], 
        [0.5, 0.25], 
        [0.5, 0.25], 
        [1.0, 2.0], 
        [0.5, 1.25], 
        [1.5, 1.0], 
        [2.5, 4.0], 
        [1.0, 2.5], 
        [2.5, 3.0], 
        [3.5, 2.0]]
