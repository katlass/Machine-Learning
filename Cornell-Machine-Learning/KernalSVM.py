import numpy as np
from helper import *
import matplotlib.pyplot as plt
import sys

print('You\'re running python %s' % sys.version.split(' ')[0])

%matplotlib notebook
xTr,yTr = generate_data()
visualize_2D(xTr, yTr)

def computeK(kerneltype, X, Z, kpar=1):
    """
    function K = computeK(kernel_type, X, Z)
    computes a matrix K such that Kij=k(x,z);
    for three different function linear, rbf or polynomial.
    
    Input:
    kerneltype: either 'linear','polynomial','rbf'
    X: n input vectors of dimension d (nxd);
    Z: m input vectors of dimension d (mxd);
    kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
    
    OUTPUT:
    K : nxm kernel matrix
    """
    assert kerneltype in ["linear","polynomial","rbf"], "Kernel type %s not known." % kerneltype
    assert X.shape[1] == Z.shape[1], "Input dimensions do not match"
    
    K = None
    
    # YOUR CODE HERE)
    n ,d = X.shape
    if kerneltype == "linear": K=np.dot(X,np.transpose(Z))
    if kerneltype == 'polynomial': K=np.power((np.dot(X,np.transpose(Z)) + 1),kpar)
    if kerneltype == 'rbf': K=np.exp(np.power(l2distance(X,Z),2)*kpar*-1)
    return K



def loss(beta, b, xTr, yTr, xTe, yTe, C, kerneltype, kpar=1):
    """
    INPUT:
    beta : n dimensional vector that stores the linear combination coefficient
    xTr   : nxd dimensional matrix (training set, each row is an input vector)
    yTr   : n   dimensional vector (training label, each entry is a label)
    b     : scalar (bias)
    xTe   : mxd dimensional matrix (test set, each row is an input vector)
    yTe   : m dimensional vector (test label, each entry is a label)
    C     : scalar (constant that controls the tradeoff between l2-regularizer and hinge-loss)
    kerneltype: either 'linear','polynomial','rbf'
    kpar  : kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
    
    OUTPUTS:
    loss     : the total loss obtained with (beta, xTr, yTr, b) on xTe and yTe (scalar)
    """
    
    loss_val = 0.0
    # compute the kernel values between xTr and xTr 
    kernel_train = computeK(kerneltype, xTr, xTr, kpar)
    # compute the kernel values between xTe and xTr
    kernel_test = computeK(kerneltype, xTe, xTr, kpar)
    
    # YOUR CODE HERE
    ###SqLoss=C*np.sum(list(map(lambda x,y: np.power(np.maximum(1-(y*(np.transpose(beta)*x)) + (y*b),0),2),  kernel_test,yTr)))
    SqLoss=(C*np.sum(list(map(lambda x,y: np.power(np.maximum(1-(y*(np.transpose(beta)*x)) + (y*b),0),2),  kernel_test,yTr))))/100
    L2Reg=np.dot(np.dot(np.transpose(beta),kernel_train),beta)
    loss_val=(L2Reg+SqLoss)
    return loss_val

def grad(beta, b, xTr, yTr, xTe, yTe, C, kerneltype, kpar=1):
    """
    INPUT:
    beta : n dimensional vector that stores the linear combination coefficient
    xTr   : nxd dimensional matrix (training set, each row is an input vector)
    yTr   : n   dimensional vector (training label, each entry is a label)
    b     : scalar (bias)
    xTe   : mxd dimensional matrix (test set, each row is an input vector)
    yTe   : m dimensional vector (test label, each entry is a label)
    C     : scalar (constant that controls the tradeoff between l2-regularizer and hinge-loss)
    kerneltype: either 'linear','polynomial','rbf'
    kpar  : kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
    
    
    OUTPUTS:
    beta_grad :  n dimensional vector (the gradient of the hinge loss with respect to the alphas)
    bgrad :  constant (the gradient of he hinge loss with respect to the bias, b)
    """
    
    n, d = xTr.shape
    
    #beta_grad = np.zeros(n)
    bgrad = np.zeros(1)
    
    # compute the kernel values between xTr and xTr 
    kernel_train = computeK(kerneltype, xTr, xTr, kpar)
    # compute the kernel values between xTe and xTr
    kernel_test = computeK(kerneltype, xTe, xTr, kpar)
    
    # YOUR CODE HERE
    beta_grad=(np.dot(kernel_train,beta)*2.0)+(C*np.sum(list(map(lambda x, y: np.multiply(2.0,(np.maximum(1-(((np.dot(np.transpose(beta),x))*y)+(b*y)),0))*-y*x),kernel_test,yTr)),axis=0))
    bgrad=C*np.sum(list(map(lambda x, y: 2.0*(np.maximum(1-(((np.dot(np.transpose(beta),x))*y)+(b*y)),0))*-y,kernel_test,yTr)))
    return beta_grad, bgrad


beta_sol, bias_sol, final_loss = minimize(objective=loss, grad=grad, xTr=xTr, yTr=yTr, C=1000, kerneltype='linear', kpar=1)
print('The Final Loss of your model is: {:0.4f}'.format(final_loss))

svmclassify = lambda x: np.sign(computeK('linear', x, xTr, 1).dot(beta_sol) + bias_sol)

predsTr=svmclassify(xTr)
trainingerr=np.mean(np.sign(predsTr)!=yTr)
print("Training error: %2.4f" % trainingerr)

%matplotlib notebook
visclassifier(svmclassify, xTr, yTr)

xTr_spiral,yTr_spiral,xTe_spiral,yTe_spiral = spiraldata()

%matplotlib notebook
visualize_2D(xTr_spiral, yTr_spiral)


beta_sol_spiral, bias_sol_spiral, final_loss_spiral = minimize(objective=loss, grad=grad, xTr=xTr_spiral, yTr=yTr_spiral, C=100, kerneltype='rbf', kpar=1)
print('The Final Loss of your model is: {:0.4f}'.format(final_loss_spiral))

svmclassify_spiral = lambda x: np.sign(computeK('rbf', xTr_spiral, x, 1).transpose().dot(beta_sol_spiral) + bias_sol_spiral)

predsTr_spiral = svmclassify_spiral(xTr_spiral)
trainingerr_spiral = np.mean(predsTr_spiral != yTr_spiral)
print("Training error: %2.4f" % trainingerr_spiral)

predsTe_spiral = svmclassify_spiral(xTe_spiral)
testerr_spiral = np.mean(predsTe_spiral != yTe_spiral)
print("Test error: %2.4f" % testerr_spiral)


visclassifier(svmclassify_spiral, xTr_spiral, yTr_spiral)

Xdata = []
ldata = []
svmC=10;

fig = plt.figure()
details = {
    'ax': fig.add_subplot(111), 
}

plt.xlim(0,1)
plt.ylim(0,1)
plt.title('Click to add positive point and shift+click to add negative points.')

def vis2(fun,xTr,yTr):
    yTr = np.array(yTr).flatten()
    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    classvals = np.unique(yTr)
    res=150
    xrange = np.linspace(0,1,res)
    yrange = np.linspace(0,1,res)
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T
    testpreds = fun(xTe)
    Z = testpreds.reshape(res, res)
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
           )
    plt.show()


def generate_onclick(Xdata, ldata):    
    global details
    def onclick(event):
        if event.key == 'shift': 
            # add positive point
            details['ax'].plot(event.xdata,event.ydata,'or')
            label = 1
        else: # add negative point
            details['ax'].plot(event.xdata,event.ydata,'ob')
            label = -1    
        pos = np.array([event.xdata, event.ydata])
        ldata.append(label)
        Xdata.append(pos)
        
        X=np.array(Xdata)
        Y=np.array(ldata)
        beta_sol, bias_sol, final_loss = minimize(objective=loss, grad=grad, xTr=X, yTr=Y, C=svmC, kerneltype='rbf', kpar=1)
        svmclassify_demo = lambda x: np.sign(computeK('rbf', X, x, 1).transpose().dot(beta_sol) + bias_sol)
        vis2(svmclassify_demo, X, Y)    
    return onclick


cid = fig.canvas.mpl_connect('button_press_event', generate_onclick(Xdata, ldata))
plt.show()

