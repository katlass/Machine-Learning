#!/usr/bin/env python
# coding: utf-8

# In[8]:


# The goal of the problem is to predict the miles per gallon a car will get using six quantities (features) about that car.
# Data imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw1-data/X_train.csv",header= None)
X_test = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw1-data/X_test.csv",header= None)
y_train = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw1-data/y_train.csv",header= None)
y_test = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw1-data/y_test.csv",header= None)
X_train.head()
y_train.describe()

## Part 1
# Part A
# Ridge regression coefficient calculation
def wrr_calc(X_train,y_train,d,Lambda):
    I = np.identity(d)
    Penalty = I*Lambda
    Squared_Magnitude=np.dot(np.transpose(X_train),X_train)
    XY_Magnitude=np.dot(np.transpose(X_train),y_train)
    Penalty_and_square = np.linalg.inv(Penalty + Squared_Magnitude)
    wrr=np.dot(Penalty_and_square, XY_Magnitude)
    return(wrr)

# Degrees of freedom calculation
def degfree(sing_vals, lambda_ ):
    return(sum(list(map(lambda x: x**2/(lambda_ + x**2),sing_vals))))

# Calculating the coefficients & degrees of freedom for the data set
n,d = X_train.shape
Lambda = 2
u, s, vh = np.linalg.svd(X_train, full_matrices=True)
coefficients=list(map(lambda x: wrr_calc(X_train,y_train,d,x),range(0,5001)))
deg_frees=list(map(lambda x: degfree(s,x),range(0,5001)))

# Visualizing results
colnames = ["cylinders", "displacement", "horsepower","weight","acceleration", "year made","constant"]
fig = plt.figure()
plt.plot(deg_frees,np.transpose(coefficients)[0][0],color="black", label = colnames[0])
plt.plot(deg_frees,np.transpose(coefficients)[0][1],color="red", label = colnames[1])
plt.plot(deg_frees,np.transpose(coefficients)[0][2],color="Green", label = colnames[2])
plt.plot(deg_frees,np.transpose(coefficients)[0][3],color="Blue", label = colnames[3])
plt.plot(deg_frees,np.transpose(coefficients)[0][4],color="Orange", label = colnames[4])
plt.plot(deg_frees,np.transpose(coefficients)[0][5],color="Purple", label = colnames[5])
plt.plot(deg_frees,np.transpose(coefficients)[0][6],color="LightBlue", label = colnames[6])
plt.title('Coefficients as a function of the  effective degrees of freedom')
plt.ylabel('Coefficients')
plt.xlabel('Degrees of freedom')
plt.legend( loc='lower left', ncol=2)
plt.grid()
plt.show()

# Part C
# Root mean square error calculation
def rmse(y,yhat_):
    n=len(y)
    mse=np.sum(list(map(lambda x: (y.values[x]-yhat_[x])**2,range(n))))/n
    return(np.sqrt(mse))

# Calculating the root mean square error for the ridge regression predictions
lambdas=range(0,51)
yhat=list(map(lambda w: np.dot(X_test.values,w),coefficients))
rmses=list(map(lambda yhat_: rmse(y_test,yhat_),yhat))

# Visualizing results
fig = plt.figure()
plt.plot(lambdas,rmses[0:51],color="black")
plt.title('Root Mean Squared Error vs. Lambda')
plt.ylabel('RMSE')
plt.xlabel('Lambda')
plt.grid()
plt.show()


## Part 2
# Part D
# Add an additional column that is the polynomial transform selected by the degree, defined for data frames; standardize columns
def poly_features(X,degree,bias=1): # bias equals one means there is a bias column of ones, so don't include polynomial feature for bias
    d=len(X.columns)  
    X_=X.copy()
    for x in range(d-bias):
        values=X.loc[:,x]**degree
        mean_=values.mean()
        std_=np.std(values)
        X_.loc[:,x]=list(map(lambda x: (x-mean_)/std_,values))
    return(X_.loc[:,0:d-bias-1])

# Polynomial transformation of features
X_train_poly=X_train.copy()
X_train_squared=poly_features(X_train_poly, degree =2)
X_train_cubed=poly_features(X_train_poly, degree =3)
X_train_squared=pd.concat([X_train_poly,X_train_squared],axis=1)
X_train_cubed=pd.concat([X_train_squared,X_train_cubed],axis=1)

X_test_poly=X_test.copy()
X_test_squared=poly_features(X_test_poly, degree =2)
X_test_cubed=poly_features(X_test_poly, degree =3)
X_test_squared=pd.concat([X_test_poly,X_test_squared],axis=1)
X_test_cubed=pd.concat([X_test_squared,X_test_cubed],axis=1)

# Ridge regression coefficient calculations for data set
n,d = X_train_squared.shape
coefficients_squared=list(map(lambda x: wrr_calc(X_train_squared,y_train,d,x),range(0,101)))
n,d = X_train_cubed.shape
coefficients_cubed=list(map(lambda x: wrr_calc(X_train_cubed,y_train,d,x),range(0,101)))

# Calculating the root mean square error for the ridge regression predictions
yhat_squared=list(map(lambda w: np.dot(X_test_squared.values,w),coefficients_squared[0:101]))
rmses_squared=list(map(lambda yhat_: rmse(y_test,yhat_),yhat_squared))
yhat_cubed=list(map(lambda w: np.dot(X_test_cubed.values,w),coefficients_cubed[0:101]))
rmses_cubed=list(map(lambda yhat_: rmse(y_test,yhat_),yhat_cubed))
lambdas=range(0,101)

# Visualizing results
fig = plt.figure()
plt.plot(lambdas,rmses[0:101],color="black",label="p = 1")
plt.plot(lambdas,rmses_squared,color="red",label="p = 2")
plt.plot(lambdas,rmses_cubed,color="Blue",label="p = 3")
plt.title('Polynomial Regression: RMSE vs. Lambda')
plt.ylabel('RMSE')
plt.xlabel('Lambda')
plt.legend()
plt.grid()
plt.show()

