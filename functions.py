import numpy as np
from numpy import log as ln
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
from matplotlib import cm

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

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
#Z2 = f2(X, Y)
#Z3 = f3(X, Y)
#Z4 = f4(X, Y)

plt.contour(X, Y, Z, 100, colors='black')
#plt.contour(X, Y, Z2, colors='black')
#plt.contour(X, Y, Z3, colors='black')
#plt.contour(X, Y, Z4, colors='black')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm, edgecolor = 'none')

plt.show()

#gradient, f1
gradF = np.gradient(f(x,y))
print (gradF)



# 2nd task
point = (0.5, 0.4)
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