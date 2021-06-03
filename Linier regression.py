# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:00:04 2021

@author: Administrator
"""
import numpy as np
import math
import time

# -------------
# - Baselines -
# -------------

def update_dictionary_items(dict1, dict2):
    """
    Replace any common dictionary items in dict1 with the values in dict2
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    if dict2 is None:
        return dict1
    for k in dict1:
        if k in dict2:
            dict1[k]=dict2[k]

    return dict1

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class MomentumSGD(Regressor):
    """
    Stochastic Gradient Descent
    """
    def __init__(self, parameters = {}):
        
        self.params = update_dictionary_items({
            'alpha': 0.01, 
            'beta': 0.9,
            'epochs': 1000,
        }, parameters)

        
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        velocity = np.zeros(Xtrain.shape[1])
        self.weights = np.random.rand(Xtrain.shape[1])
        numsamples = Xtrain.shape[0]
        
        ytrain = np.reshape(ytrain,(numsamples,1))
        loss = []
        times = []
        start = time.time()
        for i in range(self.params['epochs']):
            # shuffle data
            xy = np.concatenate((Xtrain, ytrain), axis=1)
            np.random.shuffle(xy)
            X_shuffle = xy[:,0:xy.shape[1]-1]
            y_shuffle = xy[:,xy.shape[1]-1]
            # compute the cost over epochs
            temp = np.dot(X_shuffle,self.weights) - y_shuffle
            cost = (temp).T.dot(temp) / (2 * numsamples)
            loss.append(cost)
            for t in range(numsamples):
                # for one sample
                xt = X_shuffle[t,:]
                yt = y_shuffle[t]
                grad = (xt.T.dot(self.weights) - yt) * xt
                velocity = self.params['beta'] * velocity + (1 - self.params['beta'] ) * grad
                self.weights = self.weights - self.params['alpha'] * velocity
            times.append(time.time()-start)
            weights = self.weights
        return loss, times, weights


    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
    

import pandas as pd
import numpy as np
import class_algorithms as algs
from utilities.utilities import Change_array_to_list


def give_combined_data(trainx,trainy):
    # combine the trainx and trainy to one file with the trainy at the first column
    d=list()
    for i in trainy:
        if 1 in i:    
            i = 1
            d.append(i)
        else:
            i=0
            d.append(i)
    d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
    
    
    new_data = np.concatenate((d,trainx),axis = 1)  #  new_data = data with label
    
    return new_data

def one_hot(trainy):
    d=list()
    for i in trainy:
        if 1 in i:    
            i = 1
            d.append(i)
        else:
            i=0
            d.append(i)
    d = np.array(d).reshape (len(d),1)   # Make the trainy to 0 or 1
    return d    


from collections import Counter
def get_BSA(trainy):
    d = list ()
    for i in trainy:
        if i[1] == 1:
            i = 1
            d.append(i)
        else:
            i = 0
            d.append(i)
    d = np.array(d).reshape(len(d),1)
    return d

def get_BSA_and_normal (trainy):
    d = list ()
    normal = '[0 0 0 0 0]'
    bsa = '[0 1 0 0 0]'
    for i in trainy:
        if i == bsa:
            i = 1
            d.append(i)
        else:
            i = 0
            d.append(i)
    d = np.array(d).reshape(len(d),1)
    return d

# trainy_bsa_list=trainy_bsa.reshape(len(trainy_bsa),).tolist()
# counter = Counter(trainy_bsa_list)

def get_DTA_and_normal(trainy):
    d = list ()
    normal = '[0 0 0 0 0]'
    bsa = '[1 0 0 0 0]'
    for i in trainy:
        if i == bsa:
            i = 1
            d.append(i)
        else:
            i = 0
            d.append(i)
    d = np.array(d).reshape(len(d),1)
    return d  

#%%
# trainX = pd.read_csv('train_all.csv')
# trainY = np.load('trainy_all.npy')


# here is seperate the BSA and The other

trainX = pd.read_csv('Seperated_data\ALL_data_trainX')
trainY = np.load('Seperated_data\ALL_data_trainY.npy')

# List_trainY = Change_array_to_list(trainY)
# trainy = get_BSA_and_normal(List_trainY)
# counter = Counter(trainy.reshape(len(trainy),).tolist())

''' # here is the comparation of  normal and BSA 
trainX = pd.read_csv('Seperated_data\ALL_Normal_trainX')
trainY = np.load('Seperated_data\ALL_Normal_trainY.npy')

trainX_BSA = pd.read_csv('Seperated_data\ALL_BSA_trainX')
trainY_BSA = np.load('Seperated_data\ALL_BSA_trainY.npy')

trainX = pd.concat([trainX, trainX_BSA])
trainY = np.concatenate ((trainY, trainY_BSA),axis=0)

'''



numruns = 1
n_input = 3
n_output =1

# get the useful feature from the trainX
Useful_data = trainX.loc[:, ['I:E ratio','inst_RR','maxF','minF','maxP','PIP','min_pressure','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC','epAUC']]
# Useful_data = trainX.loc[:, ['I:E ratio','inst_RR','tve:tvi ratio','iTime','eTime','tvi','tve','ipAUC']]
# Useful_data = trainX.loc[:, ['I:E ratio','inst_RR','tve:tvi ratio','iTime','eTime','tvi','tve']]
# Useful_data = trainX.loc[:, ['I:E ratio','inst_RR','tve:tvi ratio','eTime','tve']]
useful_data = np.array(Useful_data)

# normalize the data and see what is going on
# import matplotlib.pyplot as plt
# plt.plot(Useful_data.loc[:,['iTime']])
# plt.show()
# audio /= np.max(np.abs(audio),axis=0)

useful_data = np.nan_to_num(useful_data)

# standardize the array
# useful_data = (useful_data - np.mean(useful_data, axis=0)) / np.std(useful_data, axis=0)

#get the trainy of one-hot
trainy = one_hot(trainY).reshape(len(trainY),)

# trainy_bsa = get_BSA(trainY)  # get BSA data
# get combined dataset
dataset = give_combined_data(useful_data, trainY)


train_loss_epoch, train_loss_time ,weights = MomentumSGD().learn( useful_data, trainy)

ytest = np.dot(useful_data[0:200], weights)
