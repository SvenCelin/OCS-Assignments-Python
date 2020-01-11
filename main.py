import numpy as np
import matplotlib.pyplot as plt


def printHCCt(H1, H2, H3, C1, C2, C3, C1t, C2t, C3t, CtC1, CtC2, CtC3):
    print("H1 data: ", H1)
    print("H1 shape: ", np.asarray(H1).shape)
    print("H2 data: ", H2)
    print("H2 shape: ", np.asarray(H2).shape)
    print("H3 data: ", H3)
    print("H3 shape: ", np.asarray(H3).shape)
    
    print("C1 data: ", C1)
    print("C1 shape: ", np.asarray(C1).shape)
    print("C2 data: ", C2)
    print("C2 shape: ", np.asarray(C2).shape)
    print("C3 data: ", C3)
    print("C3 shape: ", np.asarray(C3).shape)
    
    print("C1t data: ", C1t)
    print("C1t shape: ", np.asarray(C1t).shape)
    print("C2t data: ", C2t)
    print("C2t shape: ", np.asarray(C2t).shape)
    print("C3t data: ", C3t)
    print("C3t shape: ", np.asarray(C3t).shape)
    
    print("CtC1 data: ", CtC1)
    print("CtC1 shape: ", np.asarray(CtC1).shape)
    print("CtC2 data: ", CtC2)
    print("CtC2 shape: ", np.asarray(CtC2).shape)
    print("CtC3 data: ", CtC3)
    print("CtC3 shape: ", np.asarray(CtC3).shape)

def printHCt(HCt1, HCt2, HCt3):
    print("HCt1 data: ", HCt1)
    print("HCt1 shape: ", np.asarray(HCt1).shape)
    print("HCt2 data: ", HCt2)
    print("HCt2 shape: ", np.asarray(HCt2).shape)
    print("HCt3 data: ", HCt3)
    print("HCt3 shape: ", np.asarray(HCt3).shape)

def printZCX(ZCX1, ZCX2, ZCX3):
    print("ZCX1 data: ", ZCX1)
    print("ZCX1 shape: ", np.asarray(ZCX1).shape)
    print("ZCX2 data: ", ZCX2)
    print("ZCX2 shape: ", np.asarray(ZCX2).shape)
    print("ZCX3 data: ", ZCX3)
    print("ZCX3 shape: ", np.asarray(ZCX3).shape)

def printGZ(GZ1, GZ2, GZ3):
    print("GZ1 data: ", GZ1)
    print("GZ1 shape: ", np.asarray(GZ1).shape)
    print("GZ2 data: ", GZ2)
    print("GZ2 shape: ", np.asarray(GZ2).shape)
    print("GZ3 data: ", GZ3)
    print("GZ3 shape: ", np.asarray(GZ3).shape)

def VectorElementMult(C, x):
    res = C[0]*x[0] + C[1]*x[1]
    return res

def twoVecMult(X, Y):
    a = X[0]*Y[0]
    b = X[0]*Y[1]
    c = X[1]*Y[0]
    d = X[1]*Y[1]
    res = [
        [a, b],
        [c, d]
    ]
    return res

def g(x, l):
    res = l[1] - x[1]
    res /= (l[0] - x[0])
    res = np.arctan(res)

    #print("RES = ", res)

    return res

def dg(x, l):
    top = l[1] - x[1]
    bottom = l[0] - x[0]

    y= top/bottom
    
    temp1 = 1/(1+ y*y)
    derx1 = top/(bottom*bottom)
    derx2 = -1/bottom

    derx1 *= temp1
    derx2 *= temp1

    res = [-derx1, -derx2]
    return res

def g_motion(x, l):
    print("x shape", np.asarray(x).shape)
    x1 = l[0] - x[0]
    x2 = l[1] - x[1]
    res = np.asarray([x1, x2])
    res = np.sqrt(res)
    return res

def dg_motion(x, l):
    g = g_motion(x, l)
    x1 = (x[0] - l[0])/g
    x2 = (x[1] - l[1])/g
    res = [x1, x2]
    return res

def plotContour(x_array, towers, size):
    levels=[10, 30, 50]
    #plt.figure(1, figsize=(7,7))
    fig1 = plt.gcf()

    #Create a contour grid
    x_array = np.asarray(x_array)
    x = x_array[:, 0]
    y = x_array[:, 1]
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    fig, ax = plt.subplots()
    cp = ax.contour(X, Y, Z, )

    #mark the starting position with a red star
    plt.plot(x_array[0][0], x_array[0][1], '*', markersize=10, color='red')

    #mark the towers with a blue + sign
    for i in range(0,2):
        plt.plot(towers[i][0], towers[i][1], '+', markersize=10, color='blue')
        
    #draw the lines of our estimated position
    for i in range(1, size):
        plt.plot((x_array[i-1][0],x_array[i][0]), (x_array[i-1][1],x_array[i][1]), linewidth=2.0, color="black")
        plt.plot(x_array[i][0],x_array[i][1],"*", color="black", markersize=7)
        fig1.canvas.draw()
    plt.show()

def estimate_position(towers, z):
    xi = [[2.3,2.3]]
    H1 = np.eye(2) * 0.01 
    lamda = 0.6
    for i in range(60):
        x = xi[-1]
        for j in range(0, 3):
            H1 = lamda * H1 + np.outer(dg(x, towers[:,j]), dg(x, towers[:,j])) 
            x = x + np.linalg.inv(H1).dot(dg(x, towers[:,j])).dot(g(x, towers[:,j]) - z[i, j])

        xi.append(x)

    plotContour(xi, towers, 60)
    pass


def estimate_motion(towers, z):
    xi = [[-5,-15]]
    v = [[-5,-15]]
    x_new1 = [-5,-15]
    x_new2 = [-5,-15]
    x_new3 = [-5,-15]
    #  H = np.zeros((2,2))
    H1 = np.eye(2) * 0.01 
    H2 = np.eye(2) * 0.01 
    H3 = np.eye(2) * 0.01 
    lamda = 0.6
    #plt.figure(2)
    for i in range(2):
        print("\n\nNEW MOTION ITERATION")
        x = xi[-1] 

        C1 = dg_motion(x, towers[:,0])
        C2 = dg_motion(x, towers[:,1])
        C3 = dg_motion(x, towers[:,2])
        
        C1t = np.transpose(C1)
        C2t = np.transpose(C2)
        C3t = np.transpose(C3)

        CtC1 = twoVecMult(C1t, C1)
        CtC2 = twoVecMult(C2t, C2)
        CtC3 = twoVecMult(C3t, C3)

        H1 = lamda * H1 + CtC1  
        H2 = lamda * H2 + CtC2
        H3 = lamda * H3 + CtC3 

        #printHCCt(H1, H2, H3, C1, C2, C3, C1t, C2t, C3t, CtC1, CtC2, CtC3)

        HCt1 = np.linalg.inv(H1)@(C1t)
        HCt2 = np.linalg.inv(H2)@(C2t)
        HCt3 = np.linalg.inv(H3)@(C3t)
        #printHCt(HCt1, HCt2, HCt3)

        #ZCX1 = np.outer(dg_motion(x, towers[:,0]), x)
        #ZCX2 = np.outer(dg_motion(x, towers[:,1]), x)
        #ZCX3 = np.outer(dg_motion(x, towers[:,2]), x)
        ZCX1 = z[i, 0] + VectorElementMult(C1, x)
        ZCX2 = z[i, 1] + VectorElementMult(C2, x)
        ZCX3 = z[i, 2] + VectorElementMult(C3, x)
        #printZCX(ZCX1, ZCX2, ZCX3)

        GZ1 = g_motion(x, towers[:,0]) - z[i, 0]
        GZ2 = g_motion(x, towers[:,1]) - z[i, 1]
        GZ3 = g_motion(x, towers[:,2]) - z[i, 2]
        #printGZ(GZ1, GZ2, GZ3)

        x_new1 = x - HCt1 @ GZ1
        x_new2 = x - HCt2 @ GZ2
        x_new3 = x - HCt3 @ GZ3

        x_new = (x_new1 + x_new2 + x_new3)/3

        x_temp = [xi[0][0]*i, xi[0][1]*i]
        v = x_new - x_temp
    
        xi.append(x_new)
        print("x_new shape", np.asarray(x_new).shape)
        #print("v", v)
    #plt.show()
    pass

def printAngles(z):
    t1 = 0
    t2 = 0
    t3 = 0
    for i in range(60):
        t1 += z[i][0]
        t2 += z[i][1]
        t3 += z[i][2]
    t1 /=60
    t2 /=60
    t3 /=60

    print("tower1 angle: ", t1)
    print("tower2 angle: ", t2)
    print("tower3 angle: ", t3)

if __name__ == '__main__':
    # load the data
    data = np.load('./data_position.npz')
    # towers
    towers = data['towers']
    # measurements
    origZ = data['z']
    z = data['z'] * np.pi/180

    #print('Towers:', towers.shape)
    #print('Measurements:', z.shape)
    print('Towers = ', towers)
    print("z = ", np.degrees(z))
    printAngles(np.degrees(z))
    #print("z = ", origZ)

    print("START OF ESTIMATE POSITION!!\n")
    estimate_position(towers, z)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    print("START OF ESTIMATE MOTION!!\n")
    #estimate_motion(towers, z)
