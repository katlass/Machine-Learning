#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import poisson
from tabulate import tabulate
import numpy as np
import math
#Imports
X = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Bayes_classifier/X.csv",header= None)
y = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Bayes_classifier/y.csv",header= None)

# confusion matrix
def confusion_matrix_manual(ypred,ytrue):
    if ypred==1 and ytrue ==1:
        val="TP"
    elif ypred==1 and  ytrue ==0:
        val="FP"
    elif ypred==0 and  ytrue ==0:
        val="TN"
    elif ypred==0 and  ytrue ==1:
        val="FN"
    return(val)

# question 2a 
def poisson_naive_bayes(X,y,iterations=10):
    random.seed(0)
    shuffled_indices = list(range(len(X)))
    random.shuffle(shuffled_indices)
    hold_out=round(len(X)/10)
    start_index=0
    end_index = hold_out
    all_metrics=[]
    lambda_y0s=[]
    lambda_y1s=[]
    for x in range(iterations):
         # Get test and train set
        test_indeces=shuffled_indices[start_index:end_index]
        train_indeces=list(filter(lambda x: x not in test_indeces,shuffled_indices))
        X_train, X_test = X.iloc[train_indeces], X.iloc[test_indeces]
        y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]
        # getting class MLE estimates for Bernoulli Random variable
        prob_y1=sum(y_train.values)/len(y_train)
        prob_y0=1-prob_y1
        # spam
        spam=y_train
        xiyi=[list(map(lambda x_,y_: x_*y_ ,X_train.values,spam.values))]
        xiyi=np.concatenate(xiyi,axis= 0)
        dim_sums_y1=np.sum(xiyi,axis=0)+1
        y1_sum=np.repeat(np.sum(spam)+1,len(dim_sums_y1))
        lambda_y1=dim_sums_y1/y1_sum
        # not spam
        not_spam=1-y_train
        xiyi=[list(map(lambda x_,y_: x_*y_ ,X_train.values,not_spam.values))]
        xiyi=np.concatenate(xiyi,axis= 0)
        dim_sums_y0=np.sum(xiyi,axis=0)+1
        y0_sum=np.repeat(np.sum(not_spam)+1,len(dim_sums_y0))
        lambda_y0=dim_sums_y0/y0_sum
        # predicting new data
        prediction = []
        for x in  range(len(X_test)):
            prob_0 = list(map(lambda x_,lambda_: np.log(poisson.pmf(x_,lambda_)),X_test.iloc[x,:],lambda_y0))
            prob_1 = list(map(lambda x_,lambda_: np.log(poisson.pmf(x_,lambda_)),X_test.iloc[x,:],lambda_y1))
            prob_0=prob_0 +[np.log(prob_y0)]
            prob_1=prob_1 +[np.log(prob_y1)]
            prediction =prediction + [np.argmax(np.array([np.sum(prob_0),np.sum(prob_1)]))]

        Metrics =list(map(lambda y_t,y_p: confusion_matrix_manual(y_p,y_t),y_test.iloc[:,0].values, np.array(prediction)))
        all_metrics=all_metrics+[Metrics]
        lambda_y0s=lambda_y0s+[list(lambda_y0)]
        lambda_y1s=lambda_y1s+[list(lambda_y1)]
        start_index=start_index+hold_out
        end_index=end_index+hold_out
    all_metrics.sort()
    results=np.unique(all_metrics, return_counts = True)
    avg_lambda0=np.mean(lambda_y0s, axis=0)
    avg_lambda1=np.mean(lambda_y1s, axis=0)
    return(results,avg_lambda0,avg_lambda1)

totals,avg_lambda0, avg_lambda1 =poisson_naive_bayes(X,y,iterations=10)

# showing confusion matrix and prediction accuracy
t=np.array([["   ",0,1],[0,totals[1][2],totals[1][0]],[1,totals[1][1],totals[1][3]]])
print(tabulate(t))
print("prediction accuracy:",(totals[1][2]+totals[1][3])/4600)

# question 2b
# stem plot of average lambda for spam and non-spam
num_lambdas=X.shape[1]+1
fig, ax = plt.subplots(nrows=2,  figsize=(20, 20))
ax[0].stem(range(1,num_lambdas),avg_lambda1,"red",markerfmt='ro')
ax[1].stem(range(1,num_lambdas),avg_lambda0,"goldenrod",markerfmt='yo')
ax[0].set_xticks(range(1,num_lambdas,1))
ax[1].set_xticks(range(1,num_lambdas,1))
ax[0].set_ylabel("Lambda")
ax[0].set_xlabel("Feature")
ax[1].set_ylabel("Lambda")
ax[1].set_xlabel("Feature")
ax[1].set_title("Non-spam Poisson Parameters")
ax[0].set_title("Spam Poisson Parameters")

plt.show()


# question 2c

# imports 
X = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Bayes_classifier/X.csv",header= None)
Y = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Bayes_classifier/y.csv",header= None)
# preprocessing
X['54']=1
Y.values[Y.values==0]=-1

def steepest_ascent(X,y):
    random.seed(0)
    shuffled_indices = list(range(len(X)))
    random.shuffle(shuffled_indices)
    hold_out=round(len(X)/10)
    start_index=0
    end_index = hold_out
    learning_rate =0.01/4600
    likelihoods_all=[]
    for x in range(10):
        # Get test and train set
        test_indeces=shuffled_indices[start_index:end_index]
        train_indeces=list(filter(lambda x: x not in test_indeces,shuffled_indices))
        X_train, X_test = X.iloc[train_indeces], X.iloc[test_indeces]
        y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]
        # steepest ascent algorithm
        beta = np.array(np.zeros(X_train.shape[1]))
        likelihoods=[]
        for i in range(1000):
            print(x,i)
            gradient_beta =0
            likelihood=0
            for v in range(len(X_train.values)):
                Part=y_train.values[v]*X_train.values[v]
                Sigmoid =(np.exp(np.sum(Part*beta)))/(1 +np.exp(np.sum(Part*beta)))
                gradient_beta=gradient_beta +((1-Sigmoid)*Part)
                likelihood=likelihood+np.log(Sigmoid)
            likelihoods=likelihoods+[likelihood]
            beta=beta +(learning_rate*gradient_beta)
        likelihoods_all= likelihoods_all+[likelihoods]
    return likelihoods_all

likelihoods_all=steepest_ascent(X,Y)

fig =plt.figure()
plt.plot(range(1,1001),likelihoods_all[0],color="red",label='Fold 1')
plt.plot(range(1,1001),likelihoods_all[1],color="Blue",label='Fold 2')
plt.plot(range(1,1001),likelihoods_all[2],color="Green",label='Fold 3')
plt.plot(range(1,1001),likelihoods_all[3],color="Purple",label='Fold 4')
plt.plot(range(1,1001),likelihoods_all[4],color="Pink",label='Fold 5')
plt.plot(range(1,1001),likelihoods_all[5],color="Brown",label='Fold 6')
plt.plot(range(1,1001),likelihoods_all[6],color="Black",label='Fold 7')
plt.plot(range(1,1001),likelihoods_all[7],color="Teal",label='Fold 8')
plt.plot(range(1,1001),likelihoods_all[8],color="Orange",label='Fold 9')
plt.plot(range(1,1001),likelihoods_all[9],color="Goldenrod",label='Fold 10')
plt.title("Steepest Ascent Log Likelihood")
plt.ylabel("LL")
plt.xlabel("Iteration")
plt.legend()
plt.show()

# question 2d

# attempted to do process with yi= +-1, kept getting the following error "RuntimeWarning: overflow encountered in double_scalars"
# fixed this by simply generalizing yi in the calculations and using yi=0,1, 
# original code that failed
# def gradient(beta,x_,y_):
#     Results =[]
#     Results2 =[]

#     for v in range(len(x_.values)):
#         part=x_.values[v]*y_.values[v]
#         inverse_likelihood=1/np.exp(np.sum(part*beta))
#         Results =Results+[1/(inverse_likelihood+1)]

#     for v in range(len(x_.T.values)):
#         part=x_.T.values[v]*y_.values[v]
#         Results2=Results2 +[np.sum(y_.values.T*x_.T.values[v])-np.sum(Results*part)]
#     return Results2
# def hessian(beta,x_,y_):
#     Results =[]
#     for v in range(len(x_.values)):
#         part=x_.values[v]*y_.values[v]
#         Results =Results+[np.exp(np.sum(part*beta))/((1+np.exp(np.sum(part*beta))))**2]

#     Results2 =[]
#     for v in range(len(x_.values)):
#         part=x_.values[v]*y_.values[v]
#         Results2 =Results2+[Results[v]*part]
#     Results2=np.vstack([Results2])

#     Results4 =[]
#     for v in range(len(x_.T.values)):
#         Results3 =[]
#         for j in range(len(x_.T.values)):
#             part=x_.values.T[v]*y_.values[v]
#             Results3=Results3 +[np.sum(part*Results2[:,j])]
#         Results4=Results4 +[Results3]
#     return Results4

# successful code
# imports
X = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Bayes_classifier/X.csv",header= None)
Y = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Bayes_classifier/y.csv",header= None)
X['54']=1

def gradient(beta,x_,y_):
    Results =[]
    Results2 =[]
    for v in x_.values:
        inverse_likelihood=1/np.exp(np.sum(v*beta))
        Results =Results+[1/(inverse_likelihood+1)]
    for v in range(len(x_.T.values)):
        Results2=Results2 +[np.sum(Results*x_.T.values[v])-np.sum(y_.values.T*x_.T.values[v])]
    return Results2

def hessian(beta,x_):
    Results =[]
    for v in x_.values:
        Results =Results+[np.exp(np.sum(v*beta))/((1+np.exp(np.sum(v*beta)))**2)]
    Results2 =[]
    for v in range(len(x_.values)):
        Results2 =Results2+[Results[v]*x_.values[v]]
    Results2=np.vstack([Results2])
    Results4 =[]
    for v in range(len(x_.T.values)):
        Results3 =[]
        for j in range(len(x_.T.values)):
            Results3=Results3 +[np.sum(x_.T.values[v]*Results2[:,j])]
        Results4=Results4 +[Results3]
    return Results4

def log_likelihood_function(beta,x_,y_):
    total =0
    for v in range(len(x_.values)):
        Part=y_.values[v]*x_.values[v]
        first_term=np.sum(Part*beta)
        second_term =np.log(1 +np.exp(np.sum(x_.values[v]*beta)))
        total=total +(first_term-second_term)
    return total

def newtons_method(X,y):
    random.seed(0)
    shuffled_indices = list(range(len(X)))
    random.shuffle(shuffled_indices)
    hold_out=round(len(X)/10)
    start_index=0
    end_index = hold_out
    log_likelihoods=[]
    all_metrics=[]
    for x in range(10):
        log_likelihood=[]
        # Get test and train set
        test_indeces=shuffled_indices[start_index:end_index]
        train_indeces=list(filter(lambda x: x not in test_indeces,shuffled_indices))
        X_train, X_test = X.iloc[train_indeces], X.iloc[test_indeces]
        y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]
        # perform newtons method weight update
        beta = np.array(np.zeros(X_train.shape[1]))
        for i in range(100):
            beta = -np.dot(np.linalg.pinv(hessian(beta,X_train)), gradient(beta,X_train,y_train))+beta
            log_likelihood=log_likelihood +[log_likelihood_function(beta,X_train,y_train)]

        # predicting test set
        Results =[]
        for v in X_test.values:
            Probability =1/(1+np.exp(-1*np.sum(v*beta)))
            Results =Results+[int(Probability>0.5)]
        Metrics =list(map(lambda y_t,y_p: confusion_matrix_manual(y_p,y_t),y_test.iloc[:,0].values, np.array(Results)))
        all_metrics=all_metrics + [Metrics]
        log_likelihoods=log_likelihoods +[log_likelihood]
        start_index=start_index+hold_out
        end_index=end_index+hold_out
    all_metrics.sort()
    test_results=np.unique(all_metrics, return_counts = True)
    return (log_likelihoods,test_results)

log_likelihoods,test_results =newtons_method(X, Y)

# plotting results
fig =plt.figure()
plt.plot(range(1,101),log_likelihoods[0],color="red",label='Fold 1')
plt.plot(range(1,101),log_likelihoods[1],color="Blue",label='Fold 2')
plt.plot(range(1,101),log_likelihoods[2],color="Green",label='Fold 3')
plt.plot(range(1,101),log_likelihoods[3],color="Purple",label='Fold 4')
plt.plot(range(1,101),log_likelihoods[4],color="Pink",label='Fold 5')
plt.plot(range(1,101),log_likelihoods[5],color="Brown",label='Fold 6')
plt.plot(range(1,101),log_likelihoods[6],color="Black",label='Fold 7')
plt.plot(range(1,101),log_likelihoods[7],color="Teal",label='Fold 8')
plt.plot(range(1,101),log_likelihoods[8],color="Orange",label='Fold 9')
plt.plot(range(1,101),log_likelihoods[9],color="Goldenrod",label='Fold 10')
plt.xticks(np.arange(1,101, 9))
plt.title("Newtons Method Log Likelihood")
plt.ylabel("LL")
plt.xlabel("Iteration")
plt.legend()
plt.show()


# question 2e

# showing confusion matrix and prediction accuracy
t=np.array([["   ",0,1],[0,test_results[1][2],test_results[1][0]],[1,test_results[1][1],test_results[1][3]]])
print(tabulate(t))
print("prediction accuracy:",(test_results[1][2]+test_results[1][3])/4600)


# question 3a

# imports
X_train = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Gaussian_process/X_train.csv",header= None)
X_test = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Gaussian_process/X_test.csv",header= None)
y_train = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Gaussian_process/y_train.csv",header= None)
y_test = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw2-data/Gaussian_process/y_test.csv",header= None)

#  Radial basis function kernel calculation
def rbf_kernel(v,k,b_term=5):
    square_mag_vk=np.dot(v, k.T)
    square_mag_v = np.sum(v**2,axis=1).reshape(-1,1) 
    square_mag_k = np.sum(k**2,axis=1) 
    square_mag= square_mag_v+ square_mag_k- (2*square_mag_vk)
    return np.exp((-1*square_mag)/b_term)

def gaussian_process(X_train,y_train,X_test,var, B):
    Predictions =[]
    RMSES =[]
    for v_  in var:
        RMSE_V =[]
        for b_  in B:
            # calculating kernels
            Kn = rbf_kernel(X_train.values, X_train.values,b_)
            Kxdn=rbf_kernel(X_test.values,  X_train.values,b_)
            k_test=np.diag(rbf_kernel(X_test.values,  X_test.values,b_))
            # calculating mean and variance
            Identity =v_*np.identity(Kn.shape[0])
            kernel_multiplication =np.matmul(Kxdn,np.linalg.inv(Identity + Kn))
            mean =np.matmul(kernel_multiplication,y_train.values)
            variance = v_ + k_test - np.matmul(kernel_multiplication , np.transpose(Kxdn))
            Predictions=Predictions +[mean]
            RMSE =np.sqrt((np.sum((y_test.values-mean)**2)/42))
            RMSE_V=RMSE_V+[RMSE]
        RMSES=RMSES+[RMSE_V]
    return RMSES

Variance = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
b = [5,7,9,11,13,15]
rmse=gaussian_process(X_train,y_train,X_test,Variance, b)

print(tabulate(pd.DataFrame(rmse,index  = Variance, columns = b),headers=b))


# question 3c

# grabbing column 4
x_4tr =X_train.iloc[:,3]
x_4te =X_test.iloc[:,3]

# calculating kernel
b=5
n=[]
for tr in x_4tr.values:
    d=[]
    for tr2 in x_4tr.values:
        square_diff=(tr -tr2)**2
        d=d+[np.sum(square_diff)*np.exp((-1*(1/b)))]
    n=n +[d]
Kn =np.vstack([n])
Kxdn=np.vstack([n])
k_test=np.diag(np.vstack([n]))

# calculating gaussian process for single vector
v_=2
Identity =v_*np.identity(Kn.shape[0])
kernel_multiplication =np.matmul(Kxdn,np.linalg.inv(Identity + Kn))
mean =np.matmul(kernel_multiplication,y_train.values)
variance = v_ + k_test - np.matmul(kernel_multiplication , np.transpose(Kxdn))

# processing results for chart
mean_s=[item for sublist in mean.tolist() for item in sublist]
Results =list(sorted(zip(mean_s,x_4tr.values)))
Values =[]
mean_s=[]
for sublist in Results:
    mean_s=mean_s+[sublist[1]]
    Values=Values+[sublist[0]]

# Visualizing Gaussian process for single dimension
fig =plt.figure()
plt.plot(mean_s,Values,color="red")
plt.scatter(x_4tr.values,y_train,color="Goldenrod")
plt.title("Gaussian Process: Car Weight Versus MPG")
plt.ylabel("MPG")
plt.xlabel("Car Weight")
plt.show()

