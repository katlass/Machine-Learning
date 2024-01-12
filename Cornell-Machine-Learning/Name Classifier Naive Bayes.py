import numpy as np
import sys
sys.path.append('/home/codio/workspace/.guides/hf')
from helper import *

%matplotlib inline
print('You\'re running python %s' % sys.version.split(' ')[0])
def hashfeatures(baby, B, FIX):
    """
    Input:
        baby : a string representing the baby's name to be hashed
        B: the number of dimensions to be in the feature vector
        FIX: the number of chunks to extract and hash from each string
    
    Output:
        v: a feature vector representing the input string
    """
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v
def name2features(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
        X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X
    
def genTrainFeatures(dimension=128):
    """
    Input: 
        dimension: desired dimension of the features
    Output: 
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """
    
    # Load in the data
    Xgirls = name2features("girls.train", B=dimension)
    Xboys = name2features("boys.train", B=dimension)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]
    
    
    
X, Y = genTrainFeatures(128)
def naivebayesPY(X, Y):
    """
    naivebayesPY(Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1,1]])
    n = len(Y)
    # YOUR CODE HERE
    pos=np.count_nonzero(Y == 1)/n
    neg=np.count_nonzero(Y == -1)/n
    return(pos,neg)
    raise NotImplementedError()
    
def naivebayesPXY(X,Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]
    
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)
    
    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    X = np.concatenate([X, np.ones((2,d)), np.zeros((2,d))])
    Y = np.concatenate([Y, [-1,1,-1,1]])
    n, d = X.shape
    
    # YOUR CODE HERE
    IndicesNegative=np.where(Y == Y.min())
    NegativeXs=X[IndicesNegative]
    negprob=np.count_nonzero(NegativeXs == 1,axis=0)/len(NegativeXs)
    IndicesPositive=np.where(Y == Y.max())
    PositiveXs=X[IndicesPositive]
    posprob=np.count_nonzero(PositiveXs == 1,axis=0)/len(PositiveXs)
    return(posprob,negprob)
    raise NotImplementedError()
    

def loglikelihood(posprob, negprob, X_test, Y_test):
    """
    loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_test
    
    Input:
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
        Y_test : labels (-1 or +1) (n)
    
    Output:
        loglikelihood of each point in X_test (n)
    """
    n, d = X_test.shape
    loglikelihood = np.zeros(n)
    #print(X_test)
    # YOUR CODE HERE
    Newy_test=Y_test.reshape(n,1)
    NewProbNeg=negprob.reshape(-1,d)
    NewProbPos=posprob.reshape(-1,d)
    newY = np.where(Newy_test==-1,NewProbNeg,NewProbPos)
    #X_test[X_test==0]=-1
    X_test=np.where(X_test==0,-1,1)
    #print(newY)
    fin=np.array([X_test,newY])
   # print("dog")
   # print(fin)
    fin=np.prod(fin,axis=0)
   # print("cat")
   # print(fin)
    #print(fin)
    fin[fin<0]=fin[fin<0]+1
    fin=np.log(fin)
   # print("final")
    #print(fin)
    #print(np.sum(fin,axis=0))
    #print(fin)
    #print(Y[0])
    ll=np.sum(fin,axis=1)
    return(ll)
    raise NotImplementedError()
    #return loglikelihood

# compute the loglikelihood of the training set
#posprob, negprob = naivebayesPXY(X,Y)
#loglikelihood(posprob,negprob,X,Y) 

def naivebayes_pred(pos, neg, posprob, negprob, X_test):
    """
    naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test
    
    Input:
        pos: class probability for the negative class
        neg: class probability for the positive class
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
    
    Output:
        prediction of each point in X_test (n)
    """
    n, d = X_test.shape
    PosNew=np.where(X_test==1,posprob,1-posprob)
    NegNew=np.where(X_test==1,negprob,1-negprob)
    #print("inside")
   # print(PosNew)
   # print(NegNew)
    PosNew=PosNew*pos
    NegNew=NegNew*neg
    #print(PosNew)
    #print(NegNew)
    PosNew=np.log(PosNew)
    NegNew=np.log(NegNew)
   # print("look here")
    #print(PosNew)
   # print(NegNew)
    ResultPos=np.sum(PosNew, axis=1)
    ResultNeg=np.sum(NegNew, axis=1)
    
  #  print("and then here")
    #print(ResultPos)
    #print(ResultNeg)
    ResultPos=ResultPos+np.log(pos)
    ResultNeg=ResultNeg+np.log(neg)
    #print("here")
    #print(ResultPos)
    #print(ResultNeg)
    End=np.where(ResultPos>ResultNeg,1,-1)
    #print(End)
    return(End)
    # YOUR CODE HERE
    raise NotImplementedError()
    
DIMS = 128
print('Loading data ...')
X,Y = genTrainFeatures(DIMS)
print('Training classifier ...')
pos, neg = naivebayesPY(X, Y)
posprob, negprob = naivebayesPXY(X, Y)
error = np.mean(naivebayes_pred(pos, neg, posprob, negprob, X) != Y)
print('Training error: %.2f%%' % (100 * error))

while True:
    print('Please enter a baby name>')
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname,B=DIMS,LoadFile=False)
    pred = naivebayes_pred(pos, neg, posprob, negprob, xtest)
    if pred > 0:
        print("%s, I am sure you are a baby boy.\n" % yourname)
    else:
        print("%s, I am sure you are a baby girl.\n" % yourname)
        
        
def hashfeatures(baby, B, FIX):
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v

def name2features2(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
        X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
        print("cat")
        
    # YOUR CODE HERE
        #break
    raise NotImplementedError()
    return X   
