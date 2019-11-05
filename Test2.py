import numpy as np
from numpy import log as ln
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import linprog
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
from matplotlib import cm

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

#START OF TASK 5
"""
def objective(x):
    return (
    0.5*x[0]  + 0.75*x[1] + 
    0.25*x[2] + 1.0*x[3] + 
    0.25*x[4] + 0.75*x[5] + 
    0.25*x[6] + 0.5*x[7] + 
    1.0*x[8]  + 0.5*x[9] + 
    1.0*x[10] + 0.25*x[11] + 
    0.5*x[12] + 0.25*x[13] + 
    0.5*x[14] + 0.25*x[15] + 
    1.0*x[16] + 2.0*x[17] + 
    0.5*x[18] + 1.25*x[19] + 
    1.5*x[20] + 1.0*x[21] + 
    2.5*x[22] + 4.0*x[23] + 
    1.0*x[24] + 2.5*x[25] + 
    2.5*x[26] + 3.0*x[27] + 
    3.5*x[28] + 2.0*x[29]
    )

def constraint1(x):
    return (
    0.5*x[0]  + 
    0.25*x[2] + 
    0.25*x[4] + 
    0.25*x[6] + 
    1.0*x[8]  + 
    1.0*x[10] + 
    0.5*x[12] + 
    0.5*x[14] + 
    1.0*x[16] + 
    0.5*x[18] + 
    1.5*x[20] + 
    2.5*x[22] + 
    1.0*x[24] + 
    2.5*x[26] + 
    3.5*x[28] -
    9
    )

def constraint2(x):
    return (
    0.75*x[1] + 
    1.0*x[3] + 
    0.75*x[5] + 
    0.5*x[7] + 
    0.5*x[9] + 
    0.25*x[11] + 
    0.25*x[13] + 
    0.25*x[15] + 
    2.0*x[17] + 
    1.25*x[19] + 
    1.0*x[21] + 
    4.0*x[23] + 
    2.5*x[25] + 
    3.0*x[27] + 
    2.0*x[29] -
    6
    )

def constraintX1(x):
    return abs(x[0] - x[1]) - 1
    
def constraintX2(x):
    return abs(x[2] - x[3]) - 1
    
def constraintX3(x):
    return abs(x[4] - x[5]) - 1
    
def constraintX4(x):
    return abs(x[6] - x[7]) - 1
    
def constraintX5(x):
    return abs(x[8] - x[9]) - 1

def constraintX6(x):
    return abs(x[10] - x[11]) - 1
    
def constraintX7(x):
    return abs(x[12] - x[13]) - 1
    
def constraintX8(x):
    return abs(x[14] - x[15]) - 1
    
def constraintX9(x):
    return abs(x[16] - x[17]) - 1
    
def constraintX10(x):
    return abs(x[18] - x[19]) - 1

def constraintX11(x):
    return abs(x[20] - x[21]) - 1
    
def constraintX12(x):
    return abs(x[22] - x[23]) - 1
    
def constraintX13(x):
    return abs(x[24] - x[25]) - 1
    
def constraintX14(x):
    return abs(x[26] - x[27]) - 1
    
def constraintX15(x):
    return abs(x[28] - x[29]) - 1

b = (0.0, 1.0)
bnds = (b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b)

con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
conx1 = {'type': 'eq', 'fun': constraintX1}
conx2 = {'type': 'eq', 'fun': constraintX2}
conx3 = {'type': 'eq', 'fun': constraintX3}
conx4 = {'type': 'eq', 'fun': constraintX4}
conx5 = {'type': 'eq', 'fun': constraintX5}
conx6 = {'type': 'eq', 'fun': constraintX6}
conx7 = {'type': 'eq', 'fun': constraintX7}
conx8 = {'type': 'eq', 'fun': constraintX8}
conx9 = {'type': 'eq', 'fun': constraintX9}
conx10 = {'type': 'eq', 'fun': constraintX10}
conx11 = {'type': 'eq', 'fun': constraintX11}
conx12 = {'type': 'eq', 'fun': constraintX12}
conx13 = {'type': 'eq', 'fun': constraintX13}
conx14 = {'type': 'eq', 'fun': constraintX14}
conx15 = {'type': 'eq', 'fun': constraintX15}

cons = [con1, con2, conx1, conx2, conx3, conx4, conx5, conx6, conx7, conx8, conx9, conx10, conx11, conx12, conx13, conx14, conx15]

x0 = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

sol = minimize(objective, x0, method='SLSQP', \
    bounds = bnds, constraints = cons)

print(sol)
"""

#END OF 5. TASK

#START OF 5. TASK 
# TAKE 2

c = [0.5 , 0.25, 0.25, 0.25, 1.0, 1.0 , 0.5 , 0.5 , 1.0, 0.5 , 1.5, 2.5, 1.0, 2.5, 3.5, 
     0.75, 1.0 , 0.75, 0.5 , 0.5, 0.25, 0.25, 0.25, 2.0, 1.25, 1.0, 4.0, 2.5, 3.0, 2.0 ]

a_ub = [
    [0.5 , 0.25, 0.25, 0.25, 1.0, 1.0 , 0.5 , 0.5 , 1.0, 0.5 , 1.5, 2.5, 1.0, 2.5, 3.5, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0.75, 1.0 , 0.75, 0.5 , 0.5, 0.25, 0.25, 0.25, 2.0, 1.25, 1.0, 4.0, 2.5, 3.0, 2.0 ]
]

b_ub = [9, 6]

a_eq = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
     #6.
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
     #11.
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
]

b_eq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

b = [0, 1]
bnds = (b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b, b)

sol = linprog(c, A_ub= a_ub, b_ub= b_ub, A_eq= a_eq, b_eq= b_eq, bounds=bnds,method='revised simplex')
print(sol)

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

x = np.random.rand(1)
y = np.random.rand(1)

"""
print("x = ")
print(x)
print("y = ")
print(y)
print("determinantX = ")
print(determinantX(f4, x, y))
print("determinantY = ")
print(determinantY(f4, x, y))
print("dFx = ")
print(dF4x(x, y))
print("dFy = ")
print(dF4y(x, y))
"""

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
#Z = f(X, Y)
Z = f2(X, Y)
#Z = f3(X, Y)
#Z = f4(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 500, cmap='autumn_r')
#ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm, edgecolor = 'none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')

#plt.show()  #***REMOVE COMMENT***

#gradient of f1
#print("New Print")
#gradF = np.gradient(f(x,y), x)

#print (f(x,y))
#print (gradF)


#print("New Print")
#gradF = np.gradient(f(x,y))

#print (f(x,y))
#print (gradF)
