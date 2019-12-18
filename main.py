import numpy as np
import matplotlib.pyplot as plt

def g(x, l):
    res = l[1] - x[1]
    res /= (l[0] - x[0])
    res = np.arctan(res)

    return res

def dg(x, l):
    y = l[1] - x[1]
    y /= (l[0] - x[0])
    
    temp1 = 1/(1+ y*y)
    derx1 = (l[1] - x[1])/((l[0] - x[0])*(l[0] - x[0]))
    derx2 = -1/(l[0] - x[0])

    derx1 *= temp1
    derx2 *= temp1

    res = [derx1, derx2]
    return res


def estimate_position(towers, z):
    xi = [[2,2]]
    zest = [0]
    #  H = np.zeros((2,2))
    H1 = np.eye(2) * 0.01 
    H2 = np.eye(2) * 0.01 
    H3 = np.eye(2) * 0.01 
    lamda = 0.6
    for i in range(60):
        x = xi[-1]
        H1 = lamda * H1 + np.outer(dg(x, towers[:,0]), dg(x, towers[:,0]))   
        H2 = lamda * H2 + np.outer(dg(x, towers[:,1]), dg(x, towers[:,1]))  
        H3 = lamda * H3 + np.outer(dg(x, towers[:,2]), dg(x, towers[:,2]))  
        
        #x_new1 = x - np.linalg.inv(H1).dot(dg(x, towers[:,0])).dot(g(x, towers[:,0]) - z[i, 0])
        #x_new2 = x - np.linalg.inv(H2).dot(dg(x, towers[:,1])).dot(g(x, towers[:,1]) - z[i, 1])
        #x_new3 = x - np.linalg.inv(H3).dot(dg(x, towers[:,2])).dot(g(x, towers[:,2]) - z[i, 2])
        
        x_new1 = x + np.linalg.inv(H1).dot(np.transpose(dg(x, towers[:,0]))).dot(z[i, 0] - np.outer(dg(x, towers[:,0]), x))
        x_new2 = x + np.linalg.inv(H2).dot(np.transpose(dg(x, towers[:,1]))).dot(z[i, 1] - np.outer(dg(x, towers[:,1]), x))
        x_new3 = x + np.linalg.inv(H3).dot(np.transpose(dg(x, towers[:,2]))).dot(z[i, 2] - np.outer(dg(x, towers[:,2]), x))

        x_new = (x_new1 + x_new2 + x_new3)/3
    
        xi.append(x_new)
        print (x)
        #zest.append(g(x_new, towers[:,0]), g(x_new, towers[:,1]), g(x_new, towers[:,2]))    
    
        #plt.figure(5, figsize=(10,6))
        #plt.figure(5, figsize=(6,6))
        #plt.clf()
        #fig = plt.gcf()
        #plt.plot(z[:len(zest)], label='measurements', lw=2)
        #plt.plot(zest[1:], label='estimate', lw=2)
        #plt.legend(loc='best')
        #fig.canvas.draw()
    #plt.tight_layout()
    pass


def estimate_motion(towers, z):
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
