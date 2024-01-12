import numpy as np
from scipy.stats import mode
from scipy import stats
import sys
%matplotlib notebook
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])

xTr,yTr,xTe,yTe=loaddata("faces.mat")


#Visualizing the Data
plt.figure(figsize=(11,8))
plotfaces(xTr[:9, :])


def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #
    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"
    # YOUR CODE HERE
    S=np.dot(X,np.transpose(X))
    G1=np.dot(X,np.transpose(Z))
    G2=np.dot(Z,np.transpose(X))
    R=np.dot(Z,np.transpose(Z))
    D=S-G1-G2+R
    D=np.sqrt(D)
    return(D)
     

def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """ 
    D1 = l2distance(xTr,xTe)
    indices=np.argsort(D1,axis=0)[0:k]
    dists=np.sort(D1,axis=0)
    dists=dists[0:k,:]
    return(indices,dists)
    
def accuracy(truth,preds):
    """
    function output=accuracy(truth,preds)         
    Analyzes the accuracy of a prediction against the ground truth
    
    Input:
    truth = n-dimensional vector of true class labels
    preds = n-dimensional vector of predictions
    
    Output:
    accuracy = scalar (percent of predictions that are correct)
    """
    
    truth = truth.flatten()
    preds = preds.flatten()

    # YOUR CODE HERE
    Mask=truth==preds
    Accurate=Mask.astype(int)
    accuracy=np.sum(Accurate)/len(Accurate)
    return(accuracy)
    raise NotImplementedError()
    
def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    # YOUR CODE HERE
    # fix array shapes
    yTr = yTr.flatten()
    Ig,Dg = findknn(xTr,xTe,k)
    Yvals=yTr[Ig]
    preds = stats.mode(Yvals)
    preds=preds[0]
    preds=preds.reshape(-1,)
    return(preds)

#Accuracy
print("Face Recognition: (1-nn)")
xTr,yTr,xTe,yTe=loaddata("faces.mat") # load the data
t0 = time.time()
preds = knnclassifier(xTr,yTr,xTe,1)
result=accuracy(yTe,preds)
t1 = time.time()
print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" % (result*100.0,t1-t0))

