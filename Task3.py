import numpy as np
import matplotlib.pyplot as plt

class norm1:
    def __init__(self, a1, b1, c1):
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        
    def dist_curve(self):
        plt.plot(self.c1, 1/(self.b1 * np.sqrt(2 * np.pi)) *
            np.exp( - (self.c1 - self.a1)**2 / (2 * self.b1**2) ), linewidth=2, color='y')
        plt.show()

#Vary the mean and SD to generate different plots

mu, sigma = 0, 0.05 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)


mean1 = 0 
sd1 = 0.05

c = np.random.normal(mean1, sd1, 3000)
        
w1, b1, z1 = plt.hist(s, 100, density=True) #hist

print(w1)
print(b1)
#print(z1)

#hist1 = norm1(mean1, sd1, x1)
#hist1.dist_curve()

#plt.plot(z1, 1/(x1 * np.sqrt(2 * np.pi)) * np.exp( - (z1 - w1)**2 / (2 * x1**2) ), linewidth=2, color='y')
#plt.show()