import numpy as np
import matplotlib 
import sys
import matplotlib.pyplot as plt
import time
sys.path.append('/home/codio/workspace/.guides/hf')
from helper import *


%matplotlib notebook
print('You\'re running python %s' % sys.version.split(' ')[0])
def perceptron_update(x,y,w):
    """
    function w=perceptron_update(x,y,w);
    
    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions
    
    Output:
    w : weight vector after updating (d)
    """
    # YOUR CODE HERE
    w=w+(x*y)
    #w=ww,np.product(x,y))
    return(w)
    raise NotImplementedError()
    
# little test
x = np.random.rand(10)
y = -1
w = np.random.rand(10)
w1 = perceptron_update(x,y,w)
def perceptron(xs,ys):
    """
    function w=perceptron(xs,ys);
    
    Implementation of a Perceptron classifier
    Input:
    xs : n input vectors of d dimensions (nxd)
    ys : n labels (-1 or +1)
    
    Output:
    w : weight vector (1xd)
    b : bias term
    """

    n, d = xs.shape     # so we have n input vectors, of d dimensions each
    w = np.zeros(d)
    b = 0.0
    
    # YOUR CODE HERE
    Count=0
    while True:
        m=0
        random_indeces = np.random.permutation(len(xs))
        for x in random_indeces:
            #print(ys[x])
           # print(xs[x])
            if ys[x]*(np.dot(np.transpose(w),xs[x]))+b<=0:
            #if np.product(ys[x],np.dot(np.transpose(w),xs[x]))<=0:
                b=b+ys[x]
                w1 = perceptron_update(xs[x],ys[x],w)
                w = w + w1
                m=m+1
        Count=Count+1
        if m == 0 or Count==100:
            break
    return (w,b)
    raise NotImplementedError()
    
  def classify_linear(xs,w,b=None):
    """
    function preds=classify_linear(xs,w,b)
    
    Make predictions with a linear classifier
    Input:
    xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
    w : weight vector of dimensionality d
    b : bias (scalar)
    
    Output:
    preds: predictions (1xn)
    """    
    w = w.flatten()    
    predictions=np.zeros(xs.shape[0])
    
    # YOUR CODE HERE
    W=np.transpose(w)
    mult=xs.dot(W)+b
    predictions=np.where(mult>0,1,-1)
    return(predictions)
    raise NotImplementedError()
    
    def test_linear1():
    xs = np.random.rand(50000,20)-0.5 # draw random data 
    w0 = np.random.rand(20)
    b0 =- 0.1 # with bias -0.1
    ys = classify_linear(xs,w0,b0)
    uniquepredictions=np.unique(ys) # check if predictions are only -1 or 1
    return set(uniquepredictions)==set([-1,1])
    test_linear1()
