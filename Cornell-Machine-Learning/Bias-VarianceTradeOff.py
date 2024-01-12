import numpy as np
from numpy.matlib import repmat
import matplotlib
import matplotlib.pyplot as plt
from helper import *

%matplotlib inline

print('You\'re running python %s' % sys.version.split(' ')[0])

OFFSET = 1.75
X, y = toydata(OFFSET, 1000)

# Visualize the generated data"
ind1 = y == 1
ind2 = y == 2
plt.figure(figsize=(10,6))
plt.scatter(X[ind1, 0], X[ind1, 1], c='r', marker='o', label='Class 1')
plt.scatter(X[ind2, 0], X[ind2, 1], c='b', marker='o', label='Class 2')
plt.legend();

def computeybar(xTe, OFFSET):
    """
    function [ybar]=computeybar(xTe, OFFSET);

    computes the expected label 'ybar' for a set of inputs x
    generated from two standard Normal distributions (one offset by OFFSET in
    both dimensions.)

    INPUT:
    xTe       : nx2 array of n vectors with 2 dimensions
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    ybar : a nx1 vector of the expected labels for vectors xTe
    noise: 
    """
    n, d = xTe.shape
    ybar = np.zeros(n)
    
    # Feel free to use the following function to compute p(x|y)
    # By default, mean is 0 and std. deviation is 1.
    normpdf = lambda x, mu, sigma: np.exp(-0.5 * np.power((x - mu) / sigma, 2)) / (np.sqrt(2 * np.pi) * sigma)
    Class1=normpdf(xTe,0,1)
    Class2=normpdf(xTe,OFFSET,1)
    ProbXGivenYClass1=np.prod(Class1,axis=1)
    ProbXGivenYClass2=np.prod(Class2,axis=1)
    ProbYGivenXClass1=(ProbXGivenYClass1)/((ProbXGivenYClass1)+(ProbXGivenYClass2))
    ProbYGivenXClass2=(ProbXGivenYClass2)/((ProbXGivenYClass2)+(ProbXGivenYClass1))
    ybar=(1.0*ProbYGivenXClass1)+(2*ProbYGivenXClass2)
    return ybar
   
def computenoise(xTe, yTe, OFFSET):
    """
    function noise=computenoise(xTe, OFFSET);

    computes the noise, or square mean of ybar - y, for a set of inputs x
    generated from two standard Normal distributions (one offset by OFFSET in
    both dimensions.)

    INPUT:
    xTe       : nx2 array of n vectors with 2 dimensions
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    noise:    : a scalar representing the noise component of the error of xTe
    """
    noise = 0
    
    # YOUR CODE HERE
    n , d = xTe.shape
    ybar = computeybar(xTe, OFFSET)
    Diff=np.sum(np.power(np.subtract(ybar,yTe),2),axis=0)
    noise=(1.0/n)*Diff
    return noise
OFFSET = 1.75
np.random.seed(1)
xTe, yTe = toydata(OFFSET, 1000)

# compute Bayes Error
ybar = computeybar(xTe, OFFSET)
predictions = np.round(ybar)
errors = predictions != yTe
err = errors.sum() / len(yTe) * 100
print('Error of Bayes classifier: %.2f%%.' % err)

# print out the noise
print('Noise: %.4f' % computenoise(xTe, yTe, OFFSET))

# plot data
ind1 = yTe == 1
ind2 = yTe == 2
plt.figure(figsize=(10,6))
plt.scatter(xTe[ind1, 0], xTe[ind1, 1], c='r', marker='o')
plt.scatter(xTe[ind2, 0], xTe[ind2, 1], c='b', marker='o')
plt.scatter(xTe[errors, 0], xTe[errors, 1], c='k', s=100, alpha=0.2)
plt.title("Plot of data (misclassified points highlighted)")
plt.show()

def computehbar(xTe, depth, Nsmall, NMODELS, OFFSET):
    """
    function [hbar]=computehbar(xTe, sigma, lmbda, NSmall, NMODELS, OFFSET);

    computes the expected prediction of the average regression tree (hbar)
    for data set xTe. 

    The regression tree should be trained using data of size Nsmall and is drawn from toydata with OFFSET 
    

    The "infinite" number of models is estimated as an average over NMODELS. 

    INPUT:
    xTe       | nx2 matrix, of n column-wise input vectors (each 2-dimensional)
    depth     | Depth of the tree 
    NSmall    | Number of points to subsample
    NMODELS   | Number of Models to average over
    OFFSET    | The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.
    OUTPUT:
    hbar | nx1 vector with the predictions of hbar for each test input
    """
    n = xTe.shape[0]
    hbar = np.zeros(n)
    # YOUR CODE HERE
    for i in range(0,NMODELS):
        xTr, yTr = toydata(OFFSET, Nsmall)
        tree = RegressionTree(depth=depth)
        tree.fit(xTr, yTr)
        pred = tree.predict(xTe)
        hbar = hbar+pred
    hbar=hbar/NMODELS
    return hbar
def computebias(xTe, depth, Nsmall, NMODELS, OFFSET):
    """
    function bias = computebias(xTe, sigma, lmbda, NSmall, NMODELS, OFFSET);

    computes the bias for data set xTe. 

    The regression tree should be trained using data of size Nsmall and is drawn from toydata with OFFSET 
    

    The "infinite" number of models is estimated as an average over NMODELS. 

    INPUT:
    xTe       | nx2 matrix, of n column-wise input vectors (each 2-dimensional)
    depth     | Depth of the tree 
    NSmall    | Number of points to subsample
    NMODELS   | Number of Models to average over
    OFFSET    | The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.
    OUTPUT:
    bias | a scalar representing the bias of the input data
    """
    noise = 0
    
    # YOUR CODE HERE
    n, d = xTe.shape
    hbar = computehbar(xTe, depth, Nsmall, NMODELS, OFFSET)
    ybar = computeybar(xTe, OFFSET)
    SqDiff=np.power(np.subtract(hbar,ybar),2)
    SumSq=np.sum(SqDiff,axis=0)
    bias=SumSq/float(n)
    return bias
def computevariance(xTe, depth, hbar, Nsmall, NMODELS, OFFSET):
    """
    function variance=computevbar(xTe,sigma,lmbda,hbar,Nsmall,NMODELS,OFFSET)

    computes the variance of classifiers trained on data sets from
    toydata.m with pre-specified "OFFSET" and 
    with kernel regression with sigma and lmbda
    evaluated on xTe. 
    the prediction of the average classifier is assumed to be stored in "hbar".

    The "infinite" number of models is estimated as an average over NMODELS. 

    INPUT:
    xTe       : nx2 matrix, of n column-wise input vectors (each 2-dimensional)
    depth     : Depth of the tree 
    hbar      : nx1 vector of the predictions of hbar on the inputs xTe
    Nsmall    : Number of samples drawn from toyData for one model
    NModel    : Number of Models to average over
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.
    
    OUTPUT:
    vbar      : nx1 vector of the difference between each model prediction and the
                average model prediction for each input
                
    """
    n = xTe.shape[0]
    vbar = np.zeros(n)
    variance = 0
    
    # YOUR CODE HERE
    for i in range(0,NMODELS):
        xTr, yTr = toydata(OFFSET, Nsmall)
        tree = RegressionTree(depth=depth)
        tree.fit(xTr, yTr)
        pred = tree.predict(xTe)
        Sqdiff=np.power(np.subtract(pred,hbar),2)
        vbar = vbar+Sqdiff
    vbar=vbar/NMODELS
    variance=np.sum(vbar)/float(n)
    return variance
    
 # biasvariancedemo

OFFSET = 1.75
# how big is the training set size N
Nsmall = 75
# how big is a really big data set (approx. infinity)
Nbig = 7500
# how many models do you want to average over
NMODELS = 100
# What regularization constants to evaluate
depths = [0, 1, 2, 3, 4, 5, 6, np.inf]

# we store
Ndepths = len(depths)
lbias = np.zeros(Ndepths)
lvariance = np.zeros(Ndepths)
ltotal = np.zeros(Ndepths)
lnoise = np.zeros(Ndepths)
lsum = np.zeros(Ndepths)

# Different regularization constant classifiers
for i in range(Ndepths):
    depth = depths[i]
    # use this data set as an approximation of the true test set
    xTe,yTe = toydata(OFFSET, Nbig)
    
    # Estimate AVERAGE ERROR (TOTAL)
    total = 0
    for j in range(NMODELS):
        # Set the seed for consistent behavior
        xTr2,yTr2 = toydata(OFFSET, Nsmall)
        model = RegressionTree(depth=depth)
        model.fit(xTr2, yTr2)
        total += np.mean((model.predict(xTe) - yTe) ** 2)
    total /= NMODELS
    
    # Estimate Noise
    noise = computenoise(xTe, yTe, OFFSET)
    
    # Estimate Bias
    bias = computebias(xTe,depth,Nsmall, NMODELS, OFFSET)
    
    # Estimating VARIANCE
    hbar = computehbar(xTe, depth, Nsmall, NMODELS, OFFSET)
    variance = computevariance(xTe, depth, hbar, Nsmall, NMODELS, OFFSET)
    
    # print and store results
    lbias[i] = bias
    lvariance[i] = variance
    ltotal[i] = total
    lnoise[i] = noise
    lsum[i] = lbias[i]+lvariance[i]+lnoise[i]
    
    if np.isinf(depths[i]):
        print('Depth infinite: Bias: %2.4f Variance: %2.4f Noise: %2.4f Bias+Variance+Noise: %2.4f Test error: %2.4f'
          % (lbias[i],lvariance[i],lnoise[i],lsum[i],ltotal[i]))
    else:
        print('Depth: %d: Bias: %2.4f Variance: %2.4f Noise: %2.4f Bias+Variance+Noise: %2.4f Test error: %2.4f'
          % (depths[i],lbias[i],lvariance[i],lnoise[i],lsum[i],ltotal[i]))
        
        
 # plot results
%matplotlib notebook
plt.figure(figsize=(10,6))
plt.plot(lbias[:Ndepths], '*', c='r',linestyle='-',linewidth=2)
plt.plot(lvariance[:Ndepths], '*', c='k', linestyle='-',linewidth=2)
plt.plot(lnoise[:Ndepths], '*', c='g',linestyle='-',linewidth=2)
plt.plot(ltotal[:Ndepths], '*', c='b', linestyle='-',linewidth=2)
plt.plot(lsum[:Ndepths], '*', c='k', linestyle='--',linewidth=2)

plt.legend(["Bias","Variance","Noise","Test error","Bias+Var+Noise"]);
plt.xlabel("Depth",fontsize=18);
plt.ylabel("Squared Error",fontsize=18);
plt.xticks([i for i in range(Ndepths)], depths);

