import numpy as np
import matplotlib.pyplot as plt


def printHValues(H, C, Ct, CtC):
    print("H data: ", H)
    print("C data: ", C)
    print("Ct data: ", Ct)
    print("CtC data: ", CtC)

def printKalmanValues(HCt1, ZCX, GZ):
    print("HCt data: ", HCt1)
    print("ZCX data: ", ZCX)
    print("GZ data: ", GZ)

def printHShapes(H, C, Ct, CtC):
    print("H shape: ", np.asarray(H).shape)
    print("C shape: ", np.asarray(C).shape)
    print("Ct shape: ", np.asarray(Ct).shape)
    print("CtC shape: ", np.asarray(CtC).shape)

def printKalmanShapes(HCt1, ZCX, GZ):
    print("HCt shape: ", np.asarray(HCt1).shape)
    print("ZCX shape: ", np.asarray(ZCX).shape)
    print("GZ shape: ", np.asarray(GZ).shape)


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

def printDistances(z, step):
    t1 = 0
    t2 = 0
    t3 = 0
    j = 0
    z1 = []

    for i in range(200):
        t1 += z[i][0]
        t2 += z[i][1]
        t3 += z[i][2]

        j = j+1
        if(j == step):
            z1.append([t1/step, t2/step, t3/step])
            j = 0
            t1 = 0
            t2 = 0
            t3 = 0
    print("for step = ", step)
    print("z1 = \n", np.asarray(z1))

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
    #print("x shape", np.asarray(x).shape)
    #print("l[0] - x[0] type = ", type(l[0] - x[0]))
    res = np.square(l[0] - x[0]) + np.square(l[1] - x[1])
    res = np.sqrt(res)
    return res

def dg_motion(x, l):
    #Nebi smio uzeti vector i podijeliti scalare sa vectorom
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
    for i in range(0,3):
        plt.plot(towers[0][i], towers[1][i], '+', markersize=10, color='blue')
        
    #draw the lines of our estimated position
    for i in range(1, size):
        plt.plot((x_array[i-1][0],x_array[i][0]), (x_array[i-1][1],x_array[i][1]), linewidth=2.0, color="black")
        plt.plot(x_array[i][0],x_array[i][1],"*", color="black", markersize=7)
        fig1.canvas.draw()
    """
    """
    plt.show()

def estimate_position(towers, z):
    xi = [[2.3,2.3]]
    H1 = np.eye(2) * 0.01 

    plot_z_meas_0 = []
    plot_z_est_0 = []
    plot_z_meas_1 = []
    plot_z_est_1 = []
    plot_z_meas_2 = []
    plot_z_est_2 = []

    lamda = 0.9
    for i in range(60):
        x = xi[-1]
        for j in range(0, 3):
            H1 = lamda * H1 + np.outer(dg(x, towers[:,j]), dg(x, towers[:,j])) 
            x = x + np.linalg.inv(H1).dot(dg(x, towers[:,j])).dot(g(x, towers[:,j]) - z[i, j])

        zest = [g(x, towers[:, 0]), g(x, towers[:, 1]), g(x, towers[:, 2])]
        #print("\nz_estimate = ", np.asarray(zest))
        #print("z_measured = ", np.asarray(z[i]))

        """
        j = 0
        plot_z_meas_0.append(z[i, j])
        plot_z_est_0.append(zest[j])

        plt.figure(5, figsize=(6,6))
        plt.clf()
        fig0 = plt.gcf()
        plt.plot(plot_z_meas_0[:len(plot_z_est_0)], label='measurements', lw=2)
        plt.plot(plot_z_est_0[1:], label='estimate', lw=2)
        plt.legend(loc='best')
        fig0.canvas.draw()

        j = 1
        plot_z_meas_1.append(z[i, j])
        plot_z_est_1.append(zest[j])

        plt.figure(6, figsize=(6,6))
        plt.clf()
        fig1 = plt.gcf()
        plt.plot(plot_z_meas_1[:len(plot_z_est_1)], label='measurements', lw=2)
        plt.plot(plot_z_est_1[1:], label='estimate', lw=2)
        plt.legend(loc='best')
        fig1.canvas.draw()

        j = 2
        plot_z_meas_2.append(z[i, j])
        plot_z_est_2.append(zest[j])

        plt.figure(7, figsize=(6,6))
        plt.clf()
        fig2 = plt.gcf()
        plt.plot(plot_z_meas_2[:len(plot_z_est_2)], label='measurements', lw=2)
        plt.plot(plot_z_est_2[1:], label='estimate', lw=2)
        plt.legend(loc='best')
        fig2.canvas.draw()
        """

        xi.append(x)

    plotContour(xi, towers, 60)
    #plt.tight_layout()
    #plt.show()
    pass


def estimate_motion(towers, z):
    xi = [[-4,-20]]
    v = [0, 0]
    #  H = np.zeros((2,2))
    H1 = np.eye(2) * 0.01 

    plot_z_meas_0 = []
    plot_z_est_0 = []
    plot_z_meas_1 = []
    plot_z_est_1 = []
    plot_z_meas_2 = []
    plot_z_est_2 = []

    lamda = 0.9
    #plt.figure(2)
    for i in range(0, 200):
        #print("\n\nNEW MOTION ITERATION")
        x = xi[-1] 

        for j in range(0, 3):
            H1 = lamda * H1 + np.outer(dg_motion(x, towers[:,j]), dg_motion(x, towers[:,j])) 
            x = x - np.linalg.inv(H1).dot(dg_motion(x, towers[:,j])).dot(g_motion(x, towers[:,j]) - z[i, j])
            
        zest = [g_motion(x, towers[:, 0]), g_motion(x, towers[:, 1]), g_motion(x, towers[:, 2])]
        #print("\nz_estimate = ", np.asarray(zest))
        #print("z_measured = ", np.asarray(z[i]))

        """
        j = 0
        plot_z_meas_0.append(z[i, j])
        plot_z_est_0.append(zest[j])

        plt.figure(5, figsize=(6,6))
        plt.clf()
        fig0 = plt.gcf()
        plt.plot(plot_z_meas_0[:len(plot_z_est_0)], label='measurements', lw=2)
        plt.plot(plot_z_est_0[1:], label='estimate', lw=2)
        plt.legend(loc='best')
        fig0.canvas.draw()

        j = 1
        plot_z_meas_1.append(z[i, j])
        plot_z_est_1.append(zest[j])

        plt.figure(6, figsize=(6,6))
        plt.clf()
        fig1 = plt.gcf()
        plt.plot(plot_z_meas_1[:len(plot_z_est_1)], label='measurements', lw=2)
        plt.plot(plot_z_est_1[1:], label='estimate', lw=2)
        plt.legend(loc='best')
        fig1.canvas.draw()

        j = 2
        plot_z_meas_2.append(z[i, j])
        plot_z_est_2.append(zest[j])

        plt.figure(7, figsize=(6,6))
        plt.clf()
        fig2 = plt.gcf()
        plt.plot(plot_z_meas_2[:len(plot_z_est_2)], label='measurements', lw=2)
        plt.plot(plot_z_est_2[1:], label='estimate', lw=2)
        plt.legend(loc='best')
        fig2.canvas.draw()
        """

        
        #x0 = [xi[0][0], xi[0][1]]
        #v = (x - x0)/(i+1)

        xi.append(x)
        #print("x = ", x)
        
    plotContour(xi, towers, 200)
    #plt.tight_layout()
    #plt.show()
    pass

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
    #print("z = ", np.degrees(z))
    #printAngles(np.degrees(z))
    #print("z = ", origZ)

    print("START OF ESTIMATE POSITION!!\n")
    estimate_position(towers, z)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']
    #print("z = ", z)
    #printDistances(z, 10)

    #print('Towers:', towers.shape)
    #print('Measurements:', z.shape)

    print("START OF ESTIMATE MOTION!!\n")
    #estimate_motion(towers, z)
