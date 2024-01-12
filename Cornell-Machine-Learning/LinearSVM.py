import numpy as np
from numpy.matlib import repmat
import sys
import time

from helper import *

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import linregress

import pylab
from matplotlib.animation import FuncAnimation

%matplotlib notebook
print('You\'re running python %s' % sys.version.split(' ')[0])

xTr, yTr = generate_data()
visualize_2D(xTr, yTr)

def loss(w, b, xTr, yTr, C):
    """
    INPUT:
    w     : d   dimensional weight vector
    b     : scalar (bias)
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    C     : scalar (constant that controls the tradeoff between l2-regularizer and hinge-loss)
    
    OUTPUTS:
    loss     : the total loss obtained with (w, b) on xTr and yTr (scalar)
    """
    
    loss_val = 0.0
    
    # YOUR CODE HERE
    loss_val=C*np.sum(list(map(lambda x, y: np.power(np.maximum(1-(((np.dot(np.transpose(w),x))*y)+(b*y)),0),2),xTr,yTr)))
    Other = np.dot(np.transpose(w),w)
    loss_val=np.asscalar(loss_val+Other)
    return loss_val

def grad(w, b, xTr, yTr, C):
    """
    INPUT:
    w     : d   dimensional weight vector
    b     : scalar (bias)
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    C     : constant (scalar that controls the tradeoff between l2-regularizer and hinge-loss)
    
    OUTPUTS:
    wgrad :  d dimensional vector (the gradient of the hinge loss with respect to the weight, w)
    bgrad :  constant (the gradient of the hinge loss with respect to the bias, b)
    """
    n, d = xTr.shape
    
    wgrad = np.zeros(d)
    bgrad = np.zeros(1)
    
    # YOUR CODE HERE
    bgrad=C*np.sum(list(map(lambda x, y: 2.0*(np.maximum(1-(((np.dot(np.transpose(w),x))*y)+(b*y)),0))*-y,xTr,yTr)))
    wgrad=(2*w)+C*np.sum(list(map(lambda x, y: np.multiply(2.0,(np.maximum(1-(((np.dot(np.transpose(w),x))*y)+(b*y)),0))*-y*x),xTr,yTr)),axis=0)
    return wgrad, bgrad




w, b, final_loss = minimize(objective=loss, grad=grad, xTr=xTr, yTr=yTr, C=1000)
print('The Final Loss of your model is: {:0.4f}'.format(final_loss))

%matplotlib notebook
visualize_classfier(xTr, yTr, w, b)

# Calculate the training error
err=np.mean(np.sign(xTr.dot(w) + b)!=yTr)
print("Training error: {:.2f} %".format (err*100))

Xdata = []
ldata = []

fig = plt.figure()
details = {
    'w': None,
    'b': None,
    'stepsize': 1,
    'ax': fig.add_subplot(111), 
    'line': None
}

plt.xlim(0,1)
plt.ylim(0,1)
plt.title('Click to add positive point and shift+click to add negative points.')

def updateboundary(Xdata, ldata):
    global details
    w_pre, b_pre, _ = minimize(objective=loss, grad=grad, xTr=np.concatenate(Xdata), 
            yTr=np.array(ldata), C=1000)
    details['w'] = np.array(w_pre).reshape(-1)
    details['b'] = b_pre
    details['stepsize'] += 1

def updatescreen():
    global details
    b = details['b']
    w = details['w']
    q = -b / (w**2).sum() * w
    if details['line'] is None:
        details['line'], = details['ax'].plot([q[0] - w[1],q[0] + w[1]],[q[1] + w[0],q[1] - w[0]],'b--')
    else:
        details['line'].set_ydata([q[1] + w[0],q[1] - w[0]])
        details['line'].set_xdata([q[0] - w[1],q[0] + w[1]])


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
        pos = np.array([[event.xdata, event.ydata]])
        ldata.append(label)
        Xdata.append(pos)
        updateboundary(Xdata,ldata)
        updatescreen()
    return onclick


cid = fig.canvas.mpl_connect('button_press_event', generate_onclick(Xdata, ldata))
plt.show()
