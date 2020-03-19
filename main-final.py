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


#from self localization python book
def plotContour(x_array, towers, size):
    fig1 = plt.gcf()

    plt.plot(x_array[0][0], x_array[0][1], '*', markersize=10, color='red')

    #towers
    for i in range(0,3):
        plt.plot(towers[0][i], towers[1][i], '+', markersize=10, color='blue')
      
    # our position
    for i in range(1, size):
        plt.plot((x_array[i-1][0],x_array[i][0]), (x_array[i-1][1],x_array[i][1]), linewidth=2.0, color="black")
        plt.plot(x_array[i][0],x_array[i][1],"*", color="black", markersize=7)
        fig1.canvas.draw()
    
    plt.show()


def estimate_position(towers, z):
    xi = [[2,2]]
    H = np.eye(2) * 0.01 
    #lamda = 0.6
    lamda = 0.9
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


def scatter_plot(v):
    v = np.asarray(v)
    x = v[:, 0]
    y = v[:, 1]
    plt.scatter(x, y, color="blue", alpha=0.5)
    plt.title('2D scatter plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def estimate_motion(towers, z):
    xi = [[-4,-20]]
    v = [[0, 0]]
    H = np.eye(2) * 0.01 
    lamda = 0.6
    #lamda = 0.9
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
        v.append(temp/(i + 1))
        xi.append(x)

    scatter_plot(v)
    plotContour(xi, towers, 200)
    print(xi)
    pass


if __name__ == '__main__':
    # load the data
    data = np.load('./data_position.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z'] * np.pi/180

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    estimate_position(towers, z)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    estimate_motion(towers, z)