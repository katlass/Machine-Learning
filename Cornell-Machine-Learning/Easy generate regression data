import numpy as np
from pylab import *
import matplotlib.pyplot as plt
%matplotlib inline 

N = 40 # 
X = np.random.rand(N,1) # Sample N points randomly along X-axis
X=np.hstack((X,np.ones((N,1))))  # Add a constant dimension
w = np.array([3, 4]) # defining a linear function 
y = X@w + np.random.randn(N) * 0.1 # defining labels
plt.plot(X[:, 0],y,".")

Learning Using Closed Form Solution
Recall the closed form solution:
𝐰=(𝐗𝑇𝐗)−1𝐗𝑇𝐲

w_closed = np.linalg.inv(X.T@X)@X.T@y
plt.plot(X[:, 0],y,".") # plot the points
z=np.array([[0,1],      # define two points with X-value 0 and 1 (and constant dimension)
            [1,1]])
plt.plot(z[:,0], z@w_closed, 'r') # draw line w_closed through these two points
w_closed = np.linalg.solve(X.T@X,X.T@y)
