import numpy as np
import matplotlib.pyplot as plt


def g(x, l):
    return np.arctan( (l[1] - x[1]) / (l[0] - x[0]) )


def dg(x, l):
    numerator = l[1] - x[1]
    denominator = l[0] - x[0]

    derx1 = numerator/(denominator*denominator)
    derx2 = -1/denominator

    derx1 *= (1 / (1 + (numerator/denominator)*(numerator/denominator)))
    derx2 *= (1 / (1 + (numerator/denominator)*(numerator/denominator)))

    res = [-derx1, -derx2]
    return res

def g_motion(x, l):
    return np.sqrt(np.square(l[0] - x[0]) + np.square(l[1] - x[1]))

def dg_motion(x, l):
    g = g_motion(x, l)
    x1 = (x[0] - l[0])/g
    x2 = (x[1] - l[1])/g
    return [x1, x2]

def plotContour(x_array, towers, size):
    #levels=[10, 30, 50]
    #plt.figure(1, figsize=(7,7))
    fig1 = plt.gcf()

    
    #Create a contour grid
    #x_array = np.asarray(x_array)
    
    #X, Y = np.meshgrid(x_array[:, 0], np.linspace(-20, 10))
    #Z = np.sqrt(X**2 + Y**2)
    #fig, ax = plt.subplots()
    #cp = ax.contour(X, Y, Z)

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
    
    plt.show()

def estimate_position(towers, z):
    xi = [[2,2]]
    H = np.eye(2) * 0.01 
    lamda = 0.6
    #lamda = 0.9
    for i in range(60):
        x = xi[-1]
        for j in range (3):
            Ci = dg(x, towers[:,j])
            CiT = np.transpose(Ci)
            CiTCi = np.outer(CiT, Ci)
            H = lamda * H + CiTCi
            HCt = np.linalg.inv(H).dot(CiT)
            x = x + HCt.dot(g(x, towers[:,j]) - z[i, j])
        xi.append(x)
    plotContour(xi, towers, 60)
    pass

def estimate_motion(towers, z):


    #      x1,  x2, v1, v2
    xi = [[-4, -20, 0, 0]]
    v = [0, 0]

    H = np.eye(2) * 0.01 


    lamda = 0.6
    #plt.figure(2)

    for i in range(200):
        x = xi[-1] 
        for j in range(3):
            Ci = dg_motion(x, towers[:,j])
            CiT = np.transpose(Ci)
            CiTCi = np.outer(CiT, Ci)
            H = lamda * H + CiTCi
            HCt = np.linalg.inv(H).dot(Ci)
            x = x - HCt.dot(g_motion(x, towers[:,j]) - z[i, j])
     
        temp = x - xi[0]
        v = temp/(i + 1)

        #x0 = [xi[0][0]*i, xi[0][1]*i]
        #v = (x - x0)/(i+1)

        xi.append(x)

        #x_temp = [xi[0][0]*i, xi[0][1]*i]
        #v = x_new - x_temp
    
        #xi.append(x_new)


        #print("x_new shape", np.asarray(x_new).shape)
        #print("x", xi)
    #plt.show()
    plotContour(xi, towers, 200)
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
    print("z = ", np.degrees(z))

    estimate_position(towers, z)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']

    print('Towers:', towers.shape)
    print('Towers but actually:', towers)
    print('Measurements:', z.shape)

    estimate_motion(towers, z)
