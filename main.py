import numpy as np
import matplotlib.pyplot as plt

def g(x, l):
    res = l[1] - x[1]
    res /= (l[0] - x[0])
    res = np.arctan(res)

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
    res = np.square(l[0] - x[0]) + np.square(l[1] - x[1])
    res = np.sqrt(res)
    return res

def dg_motion(x, l):
    g = g_motion(x, l)
    x1 = (x[0] - l[0])/g
    x2 = (x[1] - l[1])/g

    res = [-x1, -x2]
    return res

def calculateMeshZ(z_meas, x, y, towers, lamda, step):
    res = 100000
    tempRes = 0
    for j in range(0, 3):
        for i in range(0, step):
            x_arr = [x, y]
            noise = z_meas[i, j] - g(x_arr, towers[:,j])
            noise = np.abs(noise)
            noise = np.power(noise, 2)
            lamda = np.power(lamda, (step - 1 - i))
            tempRes += lamda*noise
        if(res > tempRes):
            res = tempRes
        tempRes = 0
    
    return 0.5*res

def plotContour(x_array, z_meas, z_est, towers, lamda, size, startStep, stopStep):
    levels=[10, 30, 50]
    #plt.figure(1, figsize=(7,7))
    fig1 = plt.gcf()

    #Create a contour grid
    x = np.linspace(-20, 30, 100)
    y = np.linspace(-40, 20, 100)
    z = []
    tempZ = []

    X, Y = np.meshgrid(x, y)
    
    """
    #get the height of the noise
    for i in x:
        for j in y:
            tempZ.append(calculateMeshZ(z_meas, i, j, towers, lamda, size))
        z.append(tempZ)
        tempZ = []
    z = np.asarray(z)
    """
    

    if(size == 60):
        x_add = 4
        y_add = 20
    else:
        x_add = -23
        y_add = 6
    factor = 1
    Z = (((X+x_add)**2)/factor + ((Y+y_add)**2)/factor)
    fig, ax = plt.subplots()
    cp = ax.contour(X, Y, Z)


    #mark the towers with a blue + sign
    for i in range(0,3):
        plt.plot(towers[0][i], towers[1][i], '^', markersize=10, color='blue', label="towers")
        
    #draw the lines of our estimated position
    for i in range(startStep+1, stopStep):
        plt.plot((x_array[i-1][0],x_array[i][0]), (x_array[i-1][1],x_array[i][1]), linewidth=2.0, color="black")
        plt.plot(x_array[i-1][0],x_array[i-1][1],"*", color="black", markersize=7)
        fig1.canvas.draw()
    
    #mark the starting position with a red star
    plt.plot(x_array[startStep][0], x_array[startStep][1], '*', markersize=10, color='red', label="starting position")
    #draw the final position
    plt.plot(x_array[stopStep][0], x_array[stopStep][1], 'D', markersize=5, color='green', label="final position")

    #show the labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    if(size == 60):
        title = "Estimate position with lamda = " + str(lamda) + " and time steps: [" + str(startStep) + ", " + str(stopStep) + "]"
    else:
        title = "Estimate motion with lamda = " + str(lamda) + " and time steps: [" + str(startStep) + ", " + str(stopStep) + "]"
    plt.title(title)
    plt.show()

def scatterPlotCourse(v, lamda, startStep, stopStep):
    # Create data
    v = np.asarray(v)
    x = v[:, 0]
    y = v[:, 1]
    colors = (0,0,0)
    area = np.pi*3


    # Plot
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title("Scatter plot of the course for lamda " + str(lamda))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def estimate_position(towers, z, lamda, size, startStep, stopStep):
    xi = [[2.0,2.0]]
    H1 = np.eye(2) * 0.01 
    zest = []
    for i in range(size):
        x = xi[-1]
        for j in range(0, 3):
            H1 = lamda * H1 + np.outer(dg(x, towers[:,j]), dg(x, towers[:,j])) 
            x = x + np.linalg.inv(H1).dot(dg(x, towers[:,j])).dot(g(x, towers[:,j]) - z[i, j])

        zest.append([g(x, towers[:, 0]), g(x, towers[:, 1]), g(x, towers[:, 2])])
        xi.append(x)

    plotContour(xi, z, zest, towers, lamda, size, startStep, stopStep)
    print("x_position final = ", xi[-1])
    pass


def estimate_motion(towers, z, lamda, size, startStep, stopStep):
    xi = [[-4,-20]]
    v = [[0, 0]]
    H1 = np.eye(2) * 0.01 
    zest = []
    for i in range(0, size):
        x = xi[-1] 

        for j in range(0, 3):
            H1 = lamda * H1 + np.outer(dg_motion(x, towers[:,j]), dg_motion(x, towers[:,j])) 
            x = x + np.linalg.inv(H1).dot(dg_motion(x, towers[:,j])).dot(g_motion(x, towers[:,j]) - z[i, j])
            
        zest.append([g_motion(x, towers[:, 0]), g_motion(x, towers[:, 1]), g_motion(x, towers[:, 2])])
        x0 = [xi[0][0], xi[0][1]]
            
        v.append((x - x0)/(i+1))

        xi.append(x)
        
    plotContour(xi, z, zest, towers, lamda, size, startStep, stopStep)
    scatterPlotCourse(v, lamda, startStep, stopStep)
    print("x_motion final = ", xi[-1])
    pass

if __name__ == '__main__':
    # load the data
    data = np.load('./data_position.npz')
    # towers
    towers = data['towers']
    # measurements
    origZ = data['z']
    z = data['z'] * np.pi/180

    print("START OF ESTIMATE POSITION!!\n")
    estimate_position(towers, z, 0.6, 60, 0, 10)
    estimate_position(towers, z, 0.9, 60, 5, 20)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']

    print("START OF ESTIMATE MOTION!!\n")
    estimate_motion(towers, z, 0.6, 200, 0, 50)
    estimate_motion(towers, z, 0.9, 200, 75, 150)
