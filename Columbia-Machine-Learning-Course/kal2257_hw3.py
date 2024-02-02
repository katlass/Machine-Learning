#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from tabulate import tabulate

#Probablistic Matrix Factorization: Gaussian

# imports
X_train = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw3-data/ratings_train.csv",names=["user_id","movie_id","ratings"])
X_test = pd.read_csv("/Users/katelassiter/Downloads/MLClass/hw3-data/ratings_test.csv",names= ["user_id","movie_id","ratings"])


#Forming the matrix
user_ids=np.unique(X_train["user_id"])
movie_ids=np.unique(X_train["movie_id"])
M =np.full((max(user_ids), max(movie_ids)),np.nan)

for i in  range(len(user_ids)):
    id_ =user_ids[i]
    y_loc =X_train[X_train["user_id"]==id_]["movie_id"].values-1
    rate =X_train[X_train["user_id"]==id_]["ratings"].values 
    M[i][y_loc]=rate

def Gaussian_prior(Dimension,n,Rank=10,λ = 1):
    Array=[]
    for i in range(Rank):
        Array=Array+[multivariate_normal([0], λ**-1 * np.identity(1), n)]
    return np.hstack((Array))


#Probablistic Matrix Factorization Algorithm
def pmf(M,user_ids,movie_ids,X_test,Rank= 10,σ2 = 0.25,λ = 1):
    Run_likelihoods=[]
    RMSES=[]
    V_star=0
    Last_likelihoods=[]
    for i_ in range(10):
        likelihoods=[]
        # initialize U and V as Gaussian
        U = Gaussian_prior(Rank,max(user_ids))
        V = Gaussian_prior(Rank,max(movie_ids))
        V = V.T
        U = U.T
        for i in range(100):    
            for i in range(len(user_ids)):
                # solving ridge regression for U for all user IDs
                Observed =np.argwhere(np.isnan(M[i])== False).flatten()
                Part = np.linalg.inv((λ*σ2*np.identity(Rank)) + np.dot(V[:,Observed], V[:,Observed].T))
                U[:,i]= np.dot(Part,np.dot(M[i,list(Observed)],V[:,Observed].T))
            for i in range(max(movie_ids)):
                # solving ridge regression for U for all Movie IDs
                Observed =np.argwhere(np.isnan(M[:,i])== False).flatten()
                Part = np.linalg.inv((λ*σ2*np.identity(Rank)) + np.dot(U[:,Observed], U[:,Observed].T))
                V[:,i]= np.dot(Part,np.dot(M[list(Observed),i],U[:,Observed].T))
            # solving the objective function for the log joint likelihood
            UT =np.dot(U.T,V)
            x_loc =np.argwhere(np.isnan(M)==False)[:,0]
            y_loc =np.argwhere(np.isnan(M)==False)[:,1]
            sq_diff=np.linalg.norm(M[x_loc,y_loc]-UT[x_loc,y_loc])**2
            normU=sum(list(map(lambda x: np.linalg.norm(U[:,x])**2,range(len(user_ids)))))
            normV=sum(list(map(lambda x: np.linalg.norm(V[:,x])**2,range(max(movie_ids)))))
            likelihood =-(sq_diff*(1/(2*σ2)))-λ/2 *(normU+  normV )
            likelihoods=likelihoods+[likelihood]
        Run_likelihoods =Run_likelihoods+[likelihoods]
        Last_likelihoods=Last_likelihoods+[likelihoods[-1]]
        # saving the best matrix for V
        if np.argmax(Last_likelihoods)==i_:
            V_star =V
        # calculating RMSE
        y_test=np.diagonal(np.dot(U[:,X_test["user_id"].values-1].T, V[:,X_test["movie_id"].values-1]))
        RMSES=RMSES+[np.sqrt((np.sum((y_test-X_test["ratings"].values)**2,axis=0)/len(X_test)))]
    return Run_likelihoods,Last_likelihoods,RMSES,V_star

Run_likelihoods,Last_likelihoods,RMSES,V_star=pmf(M,user_ids,movie_ids,X_test)

# plotting the objective function
fig =plt.figure()
plt.plot(range(1,100),Run_likelihoods[0][1:100],color="red",label='Run 1')
plt.plot(range(1,100),Run_likelihoods[1][1:100],color="Blue",label='Run 2')
plt.plot(range(1,100),Run_likelihoods[2][1:100],color="Green",label='Run 3')
plt.plot(range(1,100),Run_likelihoods[3][1:100],color="Purple",label='Run 4')
plt.plot(range(1,100),Run_likelihoods[4][1:100],color="Pink",label='Run 5')
plt.plot(range(1,100),Run_likelihoods[5][1:100],color="Brown",label='Run 6')
plt.plot(range(1,100),Run_likelihoods[6][1:100],color="Black",label='Run 7')
plt.plot(range(1,100),Run_likelihoods[7][1:100],color="Teal",label='Run 8')
plt.plot(range(1,100),Run_likelihoods[8][1:100],color="Orange",label='Run 9')
plt.plot(range(1,100),Run_likelihoods[9][1:100],color="Goldenrod",label='Run 10')
plt.title("Probablistic Matrix Factorization Log Likelihood")
plt.ylabel("LL")
plt.xlabel("Iteration")
plt.legend()
plt.show()

# showing final likelihood and RMSE
results=list(zip(*sorted(zip(Last_likelihoods,RMSES),reverse=True)))
print(tabulate(pd.DataFrame(results[1],results[0]),headers=['Final Training Objective','Test RMSE']))

# finding most similar movies based on user rating: smallest Euclidean distance
Star_Wars= 50 -1
My_Fair_Lady= 485 -1
Goodfellas= 182 -1
Movies =[Star_Wars,My_Fair_Lady,Goodfellas]
with open("/Users/katelassiter/Downloads/MLClass/hw3-data/movies.txt") as file:
    Movie_names = [line.rstrip() for line in file]

Movie_distances = [] 
Closest_movies = [] 
for i in Movies:
    Distances = []
    movie_dictionary ={}
    for k in range(max(movie_ids)):
        if i!=k:
            Euclidean_distance =np.linalg.norm(V_star[:,i]- V_star[:,k])
            Distances =Distances+[Euclidean_distance]
            movie_dictionary[Euclidean_distance]=Movie_names[k]
    Distances.sort()
    Movie_distances=Movie_distances +[Distances[:10]]
    Closest_movies=Closest_movies+[[movie_dictionary[i] for i in Distances[:10] ]]

# Showing most similar movies to Star Wars
df=pd.DataFrame(Movie_distances[0],columns=["Distance"])
df.index = range(1,11)
df['Title'] =Closest_movies[0]
df = df[['Title',"Distance"]]
print("Star Wars:")
print(tabulate(df,headers=['Title',"Distance"]))

# showing most similar movies to My Fair Lady
df=pd.DataFrame(Movie_distances[1],columns=["Distance"])
df.index = range(1,11)
df['Title'] =Closest_movies[1]
df = df[['Title',"Distance"]]
print("My fair Lady:")
print(tabulate(df,headers=['Title',"Distance"]))

# showing most similar movies to Goodfellas
df=pd.DataFrame(Movie_distances[2],columns=["Distance"])
df.index = range(1,11)
df['Title'] =Closest_movies[2]
df = df[['Title',"Distance"]]
print("Goodfellas:")
print(tabulate(df,headers=['Title',"Distance"]))



# Nonnegative Matrix Factorization

# imports and forming the matrix
with open("/Users/katelassiter/Downloads/MLClass/hw3-data/nyt_vocab.dat") as file:
    Vocabulary = [line.rstrip() for line in file]

with open("/Users/katelassiter/Downloads/MLClass/hw3-data/nyt_data.txt", encoding='utf-8-sig') as file:
    Documents = [line.rstrip().split(",") for line in file]
    X = np.zeros((len(Vocabulary),len(Documents)))
    Index =0
    for Document in Documents:
        Counts =[int(line.rstrip().split(":")[1]) for line in Document]
        Words =[int(line.rstrip().split(":")[0])-1 for line in Document]
        X[Words,Index]=Counts
        Index =Index +1

# Nonnegative Matrix Factorization Algorithm
def nmf(Vocabulary,Documents,Rank = 25,Noise =10**-16):
    # initializing W and H as uniform
    W = np.random.uniform(1,2,(len(Vocabulary),Rank))
    H = np.random.uniform(1,2,(len(Documents),Rank))
    Divergence = []
    for run in range(100):
        # multiplicative update for divergence objective
        normalized_approximation=np.dot((W /(np.sum(W,axis=0)+ Noise)).T,X/(np.dot(W,H.T)+Noise))
        H = (H.T*normalized_approximation).T
        normalized_approximation=np.dot(X/(np.dot(W,H.T)+Noise),H/(np.sum(H,axis=0) + Noise))
        W =  W*normalized_approximation
        # calculating divergence objective
        Divergence =Divergence+[np.sum(X*np.log((np.dot(W,H.T)+ Noise)**-1)+np.dot(W,H.T))] 
    return W, H, Divergence

W, H, Divergence=nmf(Vocabulary,Documents)

# plotting divergence objective
plt.plot(range(0,100),Divergence,color="red")
plt.title("Nonnegative Matrix Factorization Divergence")
plt.ylabel("Divergence")
plt.xlabel("Iteration")
plt.show()

# finding 10 most important words for each of the 25 topics
normalized_W =W /np.sum(W,axis=0)
Weights = [] 
Important_words = [] 
for i in range(25):
    Value_dictionary ={}
    Topic =list(normalized_W[:,i])
    for k in range(len(Vocabulary)):
        Value_dictionary[Topic[k]]=Vocabulary[k]
    Topic.sort(reverse=True)
    Weights =Weights +[Topic[:10]]
    Important_words=Important_words+[[Value_dictionary[i] for i in Topic[:10] ]]

# showing 10 most important words for the topics
df=pd.DataFrame({'T1':Important_words[0],"W1":Weights[0],'T2':Important_words[1],"W2":Weights[1],
                'T3':Important_words[2],"W3":Weights[2],'T4':Important_words[3],"W4":Weights[3],
                'T5':Important_words[4],"W5":Weights[4]})
df2=pd.DataFrame({'T6':Important_words[5],"W6":Weights[5],'T7':Important_words[6],"W7":Weights[6],
                'T8':Important_words[7],"W8":Weights[7],'T9':Important_words[8],"W9":Weights[8],
                'T10':Important_words[9],"W10":Weights[9]})
df3=pd.DataFrame({'T11':Important_words[10],"W11":Weights[10],'T12':Important_words[11],"W12":Weights[11],
                'T13':Important_words[12],"W13":Weights[12],'T14':Important_words[13],"W14":Weights[13],
                'T15':Important_words[14],"W15":Weights[14]})
df4=pd.DataFrame({'T16':Important_words[15],"W16":Weights[15],'T17':Important_words[16],"W17":Weights[16],
                'T18':Important_words[17],"W18":Weights[17],'T19':Important_words[18],"W19":Weights[18],
                'T20':Important_words[19],"W20":Weights[19]})
df5=pd.DataFrame({'T6':Important_words[20],"W6":Weights[20],'T7':Important_words[21],"W7":Weights[21],
                'T8':Important_words[22],"W8":Weights[22],'T9':Important_words[23],"W9":Weights[23],
                'T10':Important_words[24],"W10":Weights[24]})
df.index = range(1,11)
df2.index = range(1,11)
df3.index = range(1,11)
df4.index = range(1,11)
df5.index = range(1,11)

print(tabulate(df,headers=['Topic 1',"Weight",'Topic 2',"Weight",'Topic 3',"Weight",'Topic 4',"Weight",'Topic 5',"Weight"], floatfmt=".3f",stralign="center"))
print("")
print(tabulate(df2,headers=['Topic 6',"Weight",'Topic 7',"Weight",'Topic 8',"Weight",'Topic 9',"Weight",'Topic 10',"Weight"], floatfmt=".3f",stralign="center"))
print("")
print(tabulate(df3,headers=['Topic 11',"Weight",'Topic 12',"Weight",'Topic 13',"Weight",'Topic 14',"Weight",'Topic 15',"Weight"], floatfmt=".3f",stralign="center"))
print("")
print(tabulate(df4,headers=['Topic 16',"Weight",'Topic 17',"Weight",'Topic 18',"Weight",'Topic 19',"Weight",'Topic 20',"Weight"], floatfmt=".3f",stralign="center"))
print("")
print(tabulate(df5,headers=['Topic 21',"Weight",'Topic 22',"Weight",'Topic 23',"Weight",'Topic 24',"Weight",'Topic 25',"Weight"], floatfmt=".3f",stralign="center"))
